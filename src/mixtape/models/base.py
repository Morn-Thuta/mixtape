from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from mixtape.data.dataset import Dataset


class Recommender(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, data: Dataset) -> "Recommender": ...

    @abstractmethod
    def recommend(self, seeds: list[np.ndarray], k: int) -> np.ndarray:
        """Return an (n_playlists, k) array of item ids, excluding the seeds."""


def topk_sparse_rows(S, k: int, exclude: list[np.ndarray] | None = None) -> np.ndarray:
    """Top-k column ids per row of a scipy CSR score matrix (only stored entries are
    candidates), skipping column 0 (padding) and each row's excluded ids.
    Ties go to the lower column id. Memory stays O(nnz) instead of O(rows x items)."""
    S = S.tocsr()
    S.sum_duplicates()
    n = S.shape[0]
    out = np.zeros((n, k), dtype=np.int64)
    for i in range(n):
        a, b = S.indptr[i], S.indptr[i + 1]
        idx, sc = S.indices[a:b], S.data[a:b]
        bad = idx == 0
        if exclude is not None and len(exclude[i]):
            bad |= np.isin(idx, exclude[i])
        idx, sc = idx[~bad], sc[~bad]
        top = idx[np.lexsort((idx, -sc))][:k]
        out[i, : len(top)] = top
    return out


def topk_excluding(scores: np.ndarray, seeds: list[np.ndarray], k: int, xp=np) -> np.ndarray:
    """Top-k columns per row of a dense score matrix, masking padding (0) and seeds.

    Works with numpy or cupy (`xp`). Returns a numpy array.
    """
    scores = scores.astype(xp.float32, copy=True)
    scores[:, 0] = -xp.inf
    rows = np.concatenate([np.full(len(s), i) for i, s in enumerate(seeds)]) if seeds else np.array([])
    cols = np.concatenate(seeds) if seeds else np.array([])
    if len(rows):
        scores[xp.asarray(rows, dtype=xp.int64), xp.asarray(cols, dtype=xp.int64)] = -xp.inf
    k = min(k, scores.shape[1] - 1)
    part = xp.argpartition(-scores, k, axis=1)[:, :k]
    part_scores = xp.take_along_axis(scores, part, axis=1)
    order = xp.argsort(-part_scores, axis=1)
    out = xp.take_along_axis(part, order, axis=1)
    return out.get() if hasattr(out, "get") else out
