"""Item co-occurrence recommender ("songs that appear together") - CPU or GPU.

Idea: two tracks are similar if they share many playlists. With X the binary
(playlists x tracks) matrix, X^T X counts shared playlists for every pair of
tracks. We never build that huge matrix; for a batch of seed sets S we compute

    scores = (S' @ X^T) @ X   then scale columns,   with S' = S / sqrt(pop)

which equals summing cosine-normalised co-occurrence rows of the seeds.
The same code runs on scipy (CPU) or cupyx (GPU) via the backend.
"""
from __future__ import annotations

import numpy as np

from mixtape.backend import Backend, get_backend
from mixtape.data.dataset import Dataset
from mixtape.models.base import Recommender, topk_excluding


def build_matrix(be: Backend, pids: np.ndarray, items: np.ndarray, n_items: int):
    _, rows = np.unique(pids, return_inverse=True)
    xp, sp = be.xp, be.sparse
    data = xp.ones(len(items), dtype=xp.float32)
    X = sp.coo_matrix(
        (data, (xp.asarray(rows, dtype=xp.int32), xp.asarray(items, dtype=xp.int32))),
        shape=(int(rows.max()) + 1, n_items + 1),
    ).tocsr()
    X.data[:] = 1.0  # binarise duplicates
    return X


class CoOccurrence(Recommender):
    name = "cooccur"

    def __init__(self, backend: Backend | None = None, batch_size: int = 512, alpha: float = 0.5):
        self.be = backend or get_backend()
        self.batch_size = batch_size
        self.alpha = alpha  # popularity damping; 0.5 = cosine

    def fit(self, data: Dataset) -> "CoOccurrence":
        tr = data.train_only
        self.n_items = data.n_items
        self.X = build_matrix(self.be, tr["pid"].to_numpy(), tr["item"].to_numpy(), data.n_items)
        self.XT = self.X.T.tocsr()
        xp = self.be.xp
        pop = xp.asarray(self.X.sum(axis=0)).ravel()
        self.inv_pop = xp.where(pop > 0, 1.0 / xp.power(xp.maximum(pop, 1), self.alpha), 0.0) \
            .astype(xp.float32)
        return self

    def recommend(self, seeds: list[np.ndarray], k: int) -> np.ndarray:
        xp, sp = self.be.xp, self.be.sparse
        out = []
        for start in range(0, len(seeds), self.batch_size):
            batch = seeds[start:start + self.batch_size]
            rows = np.concatenate([np.full(len(s), i) for i, s in enumerate(batch)])
            cols = np.concatenate(batch)
            cols_x = xp.asarray(cols, dtype=xp.int32)
            S = sp.coo_matrix(
                (self.inv_pop[cols_x], (xp.asarray(rows, dtype=xp.int32), cols_x)),
                shape=(len(batch), self.n_items + 1),
            ).tocsr()
            scores = ((S @ self.XT) @ self.X).toarray() * self.inv_pop[None, :]
            out.append(topk_excluding(scores, batch, k, xp=xp))
        return np.vstack(out)
