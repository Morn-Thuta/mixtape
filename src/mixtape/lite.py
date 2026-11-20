"""A small, precomputed version of the demo models for free hosting (~1 GB RAM).

The full app keeps the 63M-row playlist matrix in memory (~5 GB). The lite bundle
keeps only what answering a query needs, computed from ALL 1M playlists:

* co-occurrence -> for each of the `n_items` most popular songs, its `k_nb` nearest
  songs by the same cosine score the full model uses (C[s, j] / sqrt(pop_s * pop_j)).
  A query sums the neighbour scores of its seeds (item-kNN).
* title model   -> each unique playlist name keeps its `k_title` most common songs;
  names are embedded with a static model2vec model (numpy only, no PyTorch).

Build (needs the full prepared data):  python -m mixtape lite [--config configs/mpd.yaml]
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sps

from mixtape.models.title import normalize

STATIC_MODEL = "minishlab/potion-base-8M"


# ----------------------------------------------------------------------------- build

def _neighbours(tr: pd.DataFrame, top_items: np.ndarray, n_all: int, k_nb: int, block: int = 1000):
    """Top-k cosine neighbours of every top item, among the top items."""
    local = np.full(n_all + 1, -1, dtype=np.int64)
    local[top_items] = np.arange(len(top_items))
    sub = tr[local[tr["item"].to_numpy()] >= 0]
    _, rows = np.unique(sub["pid"].to_numpy(), return_inverse=True)
    cols = local[sub["item"].to_numpy()]
    X = sps.csr_matrix((np.ones(len(cols), np.float32), (rows, cols)),
                       shape=(int(rows.max()) + 1, len(top_items)))
    X.data[:] = 1.0
    pop = np.asarray(X.sum(0)).ravel()
    inv = (1.0 / np.sqrt(np.maximum(pop, 1))).astype(np.float32)
    XT = X.T.tocsr()
    nb = np.zeros((len(top_items), k_nb), dtype=np.int32)
    sc = np.zeros((len(top_items), k_nb), dtype=np.float16)
    for b0 in range(0, len(top_items), block):
        b1 = min(b0 + block, len(top_items))
        C = (XT[b0:b1] @ X).toarray()                       # block x n co-occurrence counts
        C *= inv[b0:b1, None] * inv[None, :]
        C[np.arange(b1 - b0), np.arange(b0, b1)] = 0         # not its own neighbour
        idx = np.argpartition(-C, k_nb - 1, axis=1)[:, :k_nb]
        val = np.take_along_axis(C, idx, 1)
        order = np.argsort(-val, axis=1)
        nb[b0:b1] = top_items[np.take_along_axis(idx, order, 1)]
        sc[b0:b1] = np.take_along_axis(val, order, 1)
        print(f"  neighbours {b1:,}/{len(top_items):,}", flush=True)
    return nb, sc


def _title_table(data, tr: pd.DataFrame, k_title: int):
    pids = tr["pid"].unique()
    norm = np.array([normalize(data.titles.get(p, "")) for p in pids])
    uniq, inv = np.unique(norm, return_inverse=True)
    t_of = pd.Series(inv, index=pids)
    counts = (pd.DataFrame({"t": t_of.reindex(tr["pid"].to_numpy()).to_numpy(), "item": tr["item"].to_numpy()})
              .value_counts().rename("n").reset_index())
    counts["w"] = counts["n"] / counts.groupby("t")["n"].transform("sum")
    counts = counts.sort_values(["t", "w"], ascending=[True, False]).groupby("t").head(k_title)
    T = sps.csr_matrix((counts["w"].to_numpy(np.float32), (counts["t"].to_numpy(), counts["item"].to_numpy())),
                       shape=(len(uniq), data.n_items + 1))
    return uniq, T


def build_lite(cfg: dict, out: Path, n_items: int = 100_000, k_nb: int = 100, k_title: int = 200) -> dict:
    from model2vec import StaticModel
    from mixtape.data.dataset import load_dataset

    t0 = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    data = load_dataset(cfg["paths"]["processed"])
    tr = data.train_only
    pop = np.bincount(data.train["item"].to_numpy(), minlength=data.n_items + 1).astype(np.int32)
    pop[0] = 0
    top_items = np.sort(np.argsort(-pop)[:n_items]).astype(np.int64)

    print("co-occurrence neighbours…", flush=True)
    nb, sc = _neighbours(tr, top_items, data.n_items, k_nb)
    print("title table…", flush=True)
    uniq, T = _title_table(data, tr, k_title)
    E = StaticModel.from_pretrained(STATIC_MODEL).encode(list(uniq), normalize=True).astype(np.float16)

    nodes_path = cfg["paths"]["reports"] / "graph_nodes.parquet"
    keep = set(top_items.tolist()) | set(np.unique(T.indices).tolist())
    if nodes_path.exists():
        nodes = pd.read_parquet(nodes_path, columns=["item", "community", "pagerank"])
        nodes.to_parquet(out / "graph_nodes.parquet", index=False)
        keep |= set(nodes["item"].tolist())
    tracks = pd.read_parquet(cfg["paths"]["interim"] / "tracks.parquet",
                             columns=["tid", "track_name", "artist_name"])
    names = data.vocab[data.vocab["item"].isin(keep)].merge(tracks, on="tid")[["item", "artist_name", "track_name"]]
    names.to_parquet(out / "names.parquet", index=False)

    np.savez_compressed(out / "cooccur.npz", items=top_items.astype(np.int32), nb=nb, sc=sc)
    sps.save_npz(out / "title_T.npz", T)
    np.save(out / "title_emb.npy", E)
    np.save(out / "pop.npy", pop)
    (out / "titles.txt").write_text("\n".join(uniq), encoding="utf-8")
    stats = {"tracks": int(data.n_items), "playlists": int(data.train["pid"].nunique()),
             "lite_items": int(n_items), "k_neighbours": int(k_nb), "k_title": int(k_title),
             "static_model": STATIC_MODEL, "build_s": round(time.perf_counter() - t0, 1)}
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


# ----------------------------------------------------------------------------- runtime

def _topk(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest scores, best first, as a NEW small array (a slice of a
    full argsort would keep the whole 857k-long array alive)."""
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx], kind="stable")]


class LiteCooccur:
    """Item-kNN over precomputed cosine neighbours: score(j) = sum over seeds s of sim(s, j)."""

    def __init__(self, path: Path, n_items: int, pop: np.ndarray):
        z = np.load(path)
        self.n_items, self.pop = n_items, pop
        self.row = np.full(n_items + 1, -1, dtype=np.int64)
        self.row[z["items"]] = np.arange(len(z["items"]))
        self.nb, self.sc = z["nb"], z["sc"].astype(np.float32)

    def recommend(self, seeds: list[np.ndarray], k: int) -> np.ndarray:
        out = np.zeros((len(seeds), k), dtype=np.int64)
        fallback = _topk(self.pop.astype(np.float32), k + 64)  # most popular, for padding
        for i, s in enumerate(seeds):
            r = self.row[np.asarray(s, dtype=np.int64)]
            r = r[r >= 0]
            scores = np.zeros(self.n_items + 1, dtype=np.float32)
            if len(r):
                np.add.at(scores, self.nb[r].ravel(), self.sc[r].ravel())
            scores[np.asarray(s, dtype=np.int64)] = 0
            scores[0] = 0
            top = _topk(scores, k)
            top = top[scores[top] > 0]
            if len(top) < k:  # pad with popular songs not already chosen
                extra = fallback[~np.isin(fallback, np.concatenate([top, s, [0]]))][:k - len(top)]
                top = np.concatenate([top, extra])
            out[i] = top
        return out


class _StaticEncoder:
    def __init__(self, model_id: str):
        from model2vec import StaticModel
        self.model, self.name = StaticModel.from_pretrained(model_id), model_id.split("/")[-1]

    def encode(self, texts):
        return self.model.encode(list(texts), normalize=True)


class LiteTitle:
    def __init__(self, d: Path, model_id: str, top_m: int = 50):
        self.uniq = np.array((d / "titles.txt").read_text(encoding="utf-8").split("\n"))
        self.T = sps.load_npz(d / "title_T.npz").tocsr()
        self.E = np.load(d / "title_emb.npy").astype(np.float32)
        self.encoder, self.top_m = _StaticEncoder(model_id), top_m

    def _sim(self, title: str) -> np.ndarray:
        return self.E @ self.encoder.encode([normalize(title)])[0]

    def recommend_titles(self, titles: list[str], k: int, exclude=None) -> np.ndarray:
        out = []
        for i, t in enumerate(titles):
            sim = self._sim(t)
            idx = np.argpartition(-sim, self.top_m - 1)[:self.top_m]
            w = np.clip(sim[idx], 0, None).astype(np.float32)
            scores = np.asarray(self.T[idx].T @ w).ravel()   # similarity-weighted song pool
            scores[0] = 0
            if exclude is not None:
                scores[np.asarray(exclude[i], dtype=np.int64)] = 0
            out.append(_topk(scores, k))
        return np.vstack(out)

    def similar_titles(self, title: str, n: int = 5) -> list[tuple[str, float]]:
        sim = self._sim(title)
        top = _topk(sim, n)
        return [(str(self.uniq[i]), round(float(sim[i]), 3)) for i in top]


def load_lite(d: Path):
    stats = json.loads((d / "stats.json").read_text())
    pop = np.load(d / "pop.npy")
    names = pd.read_parquet(d / "names.parquet").set_index("item").sort_index()
    names["label"] = names["artist_name"] + " - " + names["track_name"]
    co = LiteCooccur(d / "cooccur.npz", stats["tracks"], pop)
    tm = LiteTitle(d, stats["static_model"])
    return stats, names, co, tm, pop
