"""Baseline: recommend the most popular tracks. Every model must beat this."""
from __future__ import annotations

import numpy as np

from mixtape.data.dataset import Dataset
from mixtape.models.base import Recommender, topk_excluding


class Popularity(Recommender):
    name = "popularity"

    def fit(self, data: Dataset) -> "Popularity":
        self.counts = np.bincount(data.train["item"].to_numpy(), minlength=data.n_items + 1)
        return self

    def recommend(self, seeds: list[np.ndarray], k: int) -> np.ndarray:
        scores = np.tile(self.counts.astype(np.float32), (len(seeds), 1))
        return topk_excluding(scores, seeds, k)
