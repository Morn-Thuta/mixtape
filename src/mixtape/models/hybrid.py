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

from mixtape.data.dataset import Dataset
from mixtape.models.base import Recommender


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
    n = lists[0].shape[0]
    scores = np.zeros((n, n_items + 1), dtype=np.float32)
    rows = np.arange(n)[:, None]
    for recs, w in zip(lists, weights):
        rank_score = 1.0 / (c + np.arange(1, recs.shape[1] + 1, dtype=np.float32))
        np.add.at(scores, (rows, recs), w * rank_score[None, :])
    scores[:, 0] = -np.inf
    return np.argsort(-scores, axis=1)[:, :k]
