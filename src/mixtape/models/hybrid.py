"""Hybrid recommender: blend several models with weighted reciprocal rank fusion.

Each model returns its own ranked list. A track at rank r (0-based) in model m's
list earns  w_m / (c + r + 1). Scores are summed across models, so tracks that
several models agree on rise to the top. Rank-based fusion avoids having to
calibrate very different score scales (co-occurrence counts vs transformer logits).
"""
from __future__ import annotations

import numpy as np

from mixtape.data.dataset import Dataset
from mixtape.models.base import Recommender


class Hybrid(Recommender):
    name = "hybrid"

    def __init__(self, models: list[Recommender], weights: list[float] | None = None,
                 depth: int = 200, c: float = 60.0):
        self.models = models
        self.weights = weights or [1.0] * len(models)
        self.depth, self.c = depth, c

    def fit(self, data: Dataset) -> "Hybrid":
        self.n_items = data.n_items
        for m in self.models:
            m.fit(data)
        return self

    def recommend(self, seeds: list[np.ndarray], k: int) -> np.ndarray:
        lists = [m.recommend(seeds, self.depth) for m in self.models]
        return fuse(lists, self.weights, self.n_items, k, self.c)


def fuse(lists: list[np.ndarray], weights: list[float], n_items: int, k: int,
         c: float = 60.0) -> np.ndarray:
    n, depth = lists[0].shape
    scores = np.zeros((n, n_items + 1), dtype=np.float32)
    rank_score = 1.0 / (c + np.arange(1, depth + 1, dtype=np.float32))
    rows = np.arange(n)[:, None]
    for recs, w in zip(lists, weights):
        np.add.at(scores, (rows, recs), w * rank_score[None, : recs.shape[1]])
    top = np.argsort(-scores, axis=1)[:, :k]
    return top
