"""Hybrid recommender: blend several models with weighted reciprocal rank fusion.

Each model returns its own ranked list. A track at rank r (0-based) in model m's
list earns  w_m / (c + r + 1). Scores are summed across models, so tracks that
several models agree on rise to the top. Rank-based fusion avoids having to
calibrate very different score scales (co-occurrence counts vs transformer logits).

Optionally a TitleRecommender joins the blend, using each playlist's name. That
matters most when a playlist has only one or two seed tracks.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sps

from mixtape.data.dataset import Dataset
from mixtape.models.base import Recommender, topk_sparse_rows


class Hybrid(Recommender):
    name = "hybrid"

    def __init__(self, models: list[Recommender], weights: list[float] | None = None,
                 title_model=None, title_weight: float = 0.5, depth: int = 200, c: float = 60.0):
        self.models = models
        self.weights = weights or [1.0] * len(models)
        self.title_model, self.title_weight = title_model, title_weight
        self.depth, self.c = depth, c

    def fit(self, data: Dataset) -> "Hybrid":
        self.n_items = data.n_items
        for m in self.models:
            m.fit(data)
        if self.title_model is not None:
            self.title_model.fit(data)
        return self

    def recommend(self, seeds: list[np.ndarray], k: int,
                  titles: list[str] | None = None) -> np.ndarray:
        lists = [m.recommend(seeds, self.depth) for m in self.models]
        weights = list(self.weights)
        if self.title_model is not None and titles is not None:
            lists.append(self.title_model.recommend_titles(titles, self.depth, exclude=seeds))
            weights.append(self.title_weight)
        return fuse(lists, weights, self.n_items, k, self.c)


def fuse(lists: list[np.ndarray], weights: list[float], n_items: int, k: int,
         c: float = 60.0) -> np.ndarray:
    """Weighted reciprocal rank fusion. Sparse: only the (row, item) pairs that some
    model actually ranked are stored, so memory is O(rows x depth), not O(rows x items)."""
    n = lists[0].shape[0]
    rows, cols, vals = [], [], []
    for recs, w in zip(lists, weights):
        d = recs.shape[1]
        rank_score = w / (c + np.arange(1, d + 1, dtype=np.float32))
        rows.append(np.repeat(np.arange(n), d))
        cols.append(recs.ravel())
        vals.append(np.tile(rank_score, n))
    rows, cols, vals = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
    keep = cols > 0                                          # 0 = padding
    S = sps.csr_matrix((vals[keep], (rows[keep], cols[keep])), shape=(n, n_items + 1))
    return topk_sparse_rows(S, k)
