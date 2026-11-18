"""SASRec: a transformer that reads a playlist like a sentence and predicts the next track.

  tracks  ~ words      playlist ~ sentence      next track ~ next word

Each position can only attend to earlier positions (causal mask), exactly like
GPT-style language models. Output scores are dot products between the hidden
state and every track embedding (weights tied with the input embedding).

Sequences are RIGHT-padded with 0, so real tokens never attend to padding.
"""
from __future__ import annotations

import math
import sys
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
                 batch_size=256, lr=1e-3, n_neg=None, target_window=1, val_size=0,
                 eval_every=5, patience=3, val_k=50, device=None, seed=42):
        self.target_window = target_window
        # early stopping on a held-out slice of training playlists (val_size=0 disables)
        self.val_size, self.eval_every, self.patience, self.val_k = val_size, eval_every, patience, val_k
        self.best_epoch = None
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
        all_seqs = data.train_sequences()
        val_seeds, val_targets = self._carve_validation(all_seqs, data, rng)
        seqs = _pad_right(all_seqs, L + 1)   # +1: input/target shift
        seqs = seqs[(seqs > 0).sum(1) >= 2]
        best_score, best_state, bad = -1.0, None, 0
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
                          leave=False, disable=not sys.stderr.isatty()):
                batch = torch.from_numpy(seqs[order[b:b + self.batch_size]]).to(self.device)
                inp, tgt, mask = self._targets(batch)
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
            msg = (f"  epoch {epoch + 1}/{self.epochs}  loss={total / n:.4f}  "
                   f"({time.perf_counter() - t0:.1f}s on {self.device})")
            if val_seeds and (epoch + 1) % self.eval_every == 0:
                from mixtape.eval.metrics import recall_at_k
                score = recall_at_k(self.recommend(val_seeds, self.val_k), val_targets, self.val_k)
                self.history[-1][f"val_recall@{self.val_k}"] = round(score, 4)
                msg += f"  val recall@{self.val_k}={score:.4f}"
                if score > best_score:
                    best_score, bad = score, 0
                    best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
                    self.best_epoch = epoch + 1
                else:
                    bad += 1
            print(msg)
            if val_seeds and bad >= self.patience:
                print(f"  early stop: best val recall@{self.val_k}={best_score:.4f} "
                      f"at epoch {self.best_epoch}")
                break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def _carve_validation(self, all_seqs: list[np.ndarray], data: Dataset, rng):
        """Hold out `val_size` training playlists (removed in place from all_seqs).

        Same protocol as the test set: first n_seed tracks in, the rest are targets.
        """
        n_seed = len(data.seeds[0]) if data.seeds else 5
        eligible = [i for i, s in enumerate(all_seqs) if len(s) > n_seed]
        if self.val_size <= 0 or len(eligible) < 2 * self.val_size:
            return [], []
        pick = set(rng.choice(eligible, self.val_size, replace=False).tolist())
        val = [all_seqs[i] for i in sorted(pick)]
        all_seqs[:] = [s for i, s in enumerate(all_seqs) if i not in pick]
        return [s[:n_seed] for s in val], [s[n_seed:] for s in val]

    def _targets(self, batch: torch.Tensor):
        """Inputs, targets and mask for one batch of right-padded sequences.

        target_window=1 is classic next-item SASRec. With window w, position i
        is trained to predict a random track among the next w tracks, which
        matches the task ("what else belongs in this playlist?") better than
        predicting only the very next song.
        """
        inp = batch[:, :-1]
        B, L = inp.shape
        lens = (batch > 0).sum(1, keepdim=True)                      # (B, 1)
        pos = torch.arange(L, device=batch.device).expand(B, L)
        mask = pos < lens - 1
        if self.target_window <= 1:
            return inp, batch[:, 1:], mask
        off = torch.randint(1, self.target_window + 1, (B, L), device=batch.device)
        j = torch.minimum(pos + off, (lens - 1).clamp(min=0))
        return inp, batch.gather(1, j), mask

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
