"""SASRec: a transformer that reads a playlist like a sentence and predicts the next track.

  tracks  ~ words      playlist ~ sentence      next track ~ next word

Each position can only attend to earlier positions (causal mask), exactly like
GPT-style language models. Output scores are dot products between the hidden
state and every track embedding (weights tied with the input embedding).

Sequences are RIGHT-padded with 0, so real tokens never attend to padding.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from mixtape.data.dataset import Dataset
from mixtape.models.base import Recommender


class SASRecNet(nn.Module):
    def __init__(self, n_items: int, max_len: int, dim: int, heads: int, layers: int, dropout: float):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, dim)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.dim = dim
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        L = seq.size(1)
        pos = torch.arange(L, device=seq.device)
        x = self.item_emb(seq) * math.sqrt(self.dim) + self.pos_emb(pos)[None]
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=seq.device), 1)
        h = self.encoder(self.drop(x), mask=causal, is_causal=True)
        return self.norm(h)  # (B, L, dim)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.item_emb.weight.T


def _pad_right(seqs: list[np.ndarray], max_len: int) -> np.ndarray:
    out = np.zeros((len(seqs), max_len), dtype=np.int64)
    for i, s in enumerate(seqs):
        s = s[:max_len]
        out[i, : len(s)] = s
    return out


class SASRec(Recommender):
    name = "sasrec"

    def __init__(self, max_len=50, dim=64, heads=2, layers=2, dropout=0.2, epochs=10,
                 batch_size=256, lr=1e-3, n_neg=None, device=None, seed=42):
        self.cfg = dict(max_len=max_len, dim=dim, heads=heads, layers=layers, dropout=dropout)
        self.epochs, self.batch_size, self.lr = epochs, batch_size, lr
        self.n_neg = n_neg  # None -> full softmax if catalogue is small, else 128 sampled
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.seed = seed
        self.history: list[dict] = []

    def fit(self, data: Dataset) -> "SASRec":
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.n_items = data.n_items
        n_neg = self.n_neg if self.n_neg is not None else (0 if data.n_items <= 50_000 else 128)
        L = self.cfg["max_len"]
        seqs = _pad_right(data.train_sequences(), L + 1)   # +1: input/target shift
        seqs = seqs[(seqs > 0).sum(1) >= 2]
        self.net = SASRecNet(data.n_items, L, **{k: v for k, v in self.cfg.items() if k != "max_len"}) \
            .to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=0.01)
        use_amp = self.device.type == "cuda"
        scaler = torch.amp.GradScaler(enabled=use_amp)

        for epoch in range(self.epochs):
            self.net.train()
            t0, total, n = time.perf_counter(), 0.0, 0
            order = rng.permutation(len(seqs))
            for b in tqdm(range(0, len(order), self.batch_size), desc=f"epoch {epoch + 1}",
                          leave=False):
                batch = torch.from_numpy(seqs[order[b:b + self.batch_size]]).to(self.device)
                inp, tgt = batch[:, :-1], batch[:, 1:]
                mask = tgt > 0
                with torch.autocast(self.device.type, enabled=use_amp):
                    h = self.net(inp)[mask]          # (N, dim) only real positions
                    t = tgt[mask]
                    if n_neg == 0:                   # full softmax over all tracks
                        loss = F.cross_entropy(self.net.logits(h).float(), t)
                    else:                            # sampled softmax
                        neg = torch.randint(1, self.n_items + 1, (n_neg,), device=self.device)
                        W = self.net.item_emb.weight
                        pos_logit = (h * W[t]).sum(-1, keepdim=True)
                        neg_logit = h @ W[neg].T
                        logits = torch.cat([pos_logit, neg_logit], 1).float()
                        loss = F.cross_entropy(logits, torch.zeros(len(t), dtype=torch.long,
                                                                  device=self.device))
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                total += loss.item() * len(t)
                n += len(t)
            self.history.append({"epoch": epoch + 1, "loss": round(total / n, 4),
                                 "seconds": round(time.perf_counter() - t0, 2)})
            print(f"  epoch {epoch + 1}/{self.epochs}  loss={total / n:.4f}  "
                  f"({time.perf_counter() - t0:.1f}s on {self.device})")
        return self

    @torch.no_grad()
    def recommend(self, seeds: list[np.ndarray], k: int) -> np.ndarray:
        self.net.eval()
        L = self.cfg["max_len"]
        out = []
        for b in range(0, len(seeds), 1024):
            batch = seeds[b:b + 1024]
            seq = torch.from_numpy(_pad_right(batch, L)).to(self.device)
            last = (seq > 0).sum(1).clamp(min=1) - 1
            h = self.net(seq)[torch.arange(len(batch), device=self.device), last]
            scores = self.net.logits(h).float()
            scores[:, 0] = -float("inf")
            for i, s in enumerate(batch):
                scores[i, torch.as_tensor(s, device=self.device)] = -float("inf")
            out.append(scores.topk(k, dim=1).indices.cpu().numpy())
        return np.vstack(out)
