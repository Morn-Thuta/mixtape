"""Ranking metrics for playlist continuation."""
from __future__ import annotations

import numpy as np


def recall_at_k(recs: np.ndarray, targets: list[np.ndarray], k: int) -> float:
    """Fraction of each playlist's held-out tracks found in the top-k, averaged."""
    vals = [len(np.intersect1d(r[:k], t)) / len(t) for r, t in zip(recs, targets) if len(t)]
    return float(np.mean(vals)) if vals else 0.0


def ndcg_at_k(recs: np.ndarray, targets: list[np.ndarray], k: int) -> float:
    """Rewards putting correct tracks near the top of the list."""
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    vals = []
    for r, t in zip(recs, targets):
        if not len(t):
            continue
        hits = np.isin(r[:k], t).astype(float)
        ideal = discounts[: min(len(t), k)].sum()
        vals.append((hits * discounts[: len(hits)]).sum() / ideal)
    return float(np.mean(vals)) if vals else 0.0


def hit_rate_at_k(recs: np.ndarray, targets: list[np.ndarray], k: int) -> float:
    """Share of playlists where at least one correct track is in the top-k."""
    vals = [np.isin(r[:k], t).any() for r, t in zip(recs, targets) if len(t)]
    return float(np.mean(vals)) if vals else 0.0


def evaluate(recs: np.ndarray, targets: list[np.ndarray], k: int) -> dict[str, float]:
    return {
        f"recall@{k}": round(recall_at_k(recs, targets, k), 4),
        f"ndcg@{k}": round(ndcg_at_k(recs, targets, k), 4),
        f"hit@{k}": round(hit_rate_at_k(recs, targets, k), 4),
    }
