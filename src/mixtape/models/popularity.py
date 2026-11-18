"""Baseline: recommend the most popular tracks. Every model must beat this."""
from __future__ import annotations

import numpy as np

from mixtape.data.dataset import Dataset
from mixtape.models.base import Recommender


class Popularity(Recommender):
    name = "popularity"

    def fit(self, data: Dataset) -> "Popularity":
        self.counts = np.bincount(data.train["item"].to_numpy(), minlength=data.n_items + 1)
        self.counts[0] = 0
        return self

    def recommend(self, seeds: list[np.ndarray], k: int) -> np.ndarray:
        # Every playlist gets the same global ranking minus its own seeds, so rank once
        # and filter per row (a dense playlists x tracks matrix would be ~14 GB on MPD).
        longest = max((len(s) for s in seeds), default=0)
        n = min(k + longest, len(self.counts) - 1)
        top = np.argpartition(-self.counts, n)[:n]
        top = top[np.argsort(-self.counts[top], kind="stable")]
        out = np.empty((len(seeds), k), dtype=np.int64)
        for i, s in enumerate(seeds):
            row = top[~np.isin(top, s)][:k] if len(s) else top[:k]
            out[i, : len(row)] = row
            out[i, len(row):] = 0
        return out
