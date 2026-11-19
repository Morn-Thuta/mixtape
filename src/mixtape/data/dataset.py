"""In-memory view of the processed split, shared by all models."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    train: pd.DataFrame          # pid, rank, item  (incl. seed part of test playlists)
    n_items: int                 # items are 1..n_items
    test_pids: np.ndarray
    seeds: list[np.ndarray]      # aligned with test_pids
    targets: list[np.ndarray]    # aligned with test_pids
    titles: dict[int, str]       # pid -> playlist name (all playlists)
    vocab: pd.DataFrame          # tid, item  (tid -> interim/tracks.parquet)

    @property
    def train_only(self) -> pd.DataFrame:
        """Train rows excluding the seed rows of test playlists."""
        return self.train[~self.train["pid"].isin(self.test_pids)]

    def train_sequences(self) -> list[np.ndarray]:
        tr = self.train_only.sort_values(["pid", "rank"])
        return [g.to_numpy() for _, g in tr.groupby("pid", sort=False)["item"]]

    def with_validation(self, n_val: int, seed: int = 7) -> "Dataset":
        """A copy where `n_val` training playlists play the role of the test set.

        Same protocol as the real split (first n_seed tracks in, the rest are targets),
        and their target rows are removed from train. Use it to tune hyperparameters
        such as hybrid weights without ever looking at the real test set.
        """
        n_seed = len(self.seeds[0]) if self.seeds else 5
        tr = self.train_only
        lens = tr.groupby("pid").size()
        eligible = lens[lens > n_seed].index.to_numpy()
        rng = np.random.default_rng(seed)
        val_pids = np.sort(rng.choice(eligible, min(n_val, len(eligible)), replace=False))
        is_val = self.train["pid"].isin(val_pids)
        val_rows = self.train[is_val]
        return Dataset(
            train=self.train[~is_val | (self.train["rank"] < n_seed)],
            n_items=self.n_items,
            test_pids=val_pids,
            seeds=_group(val_rows[val_rows["rank"] < n_seed], val_pids),
            targets=_group(val_rows[val_rows["rank"] >= n_seed], val_pids),
            titles=self.titles,
            vocab=self.vocab,
        )


def _group(df: pd.DataFrame, pids: np.ndarray) -> list[np.ndarray]:
    g = {pid: s.to_numpy() for pid, s in df.sort_values(["pid", "rank"]).groupby("pid")["item"]}
    return [g.get(p, np.array([], dtype=np.int32)) for p in pids]


def load_dataset(processed: Path) -> Dataset:
    train = pd.read_parquet(processed / "train.parquet")
    seed = pd.read_parquet(processed / "test_seed.parquet")
    target = pd.read_parquet(processed / "test_target.parquet")
    vocab = pd.read_parquet(processed / "vocab.parquet")
    pls = pd.read_parquet(processed / "playlists.parquet")
    test_pids = np.sort(target["pid"].unique())
    return Dataset(
        train=train,
        n_items=int(vocab["item"].max()),
        test_pids=test_pids,
        seeds=_group(seed, test_pids),
        targets=_group(target, test_pids),
        titles=dict(zip(pls["pid"].to_numpy(), pls["name"].fillna("").to_numpy())),
        vocab=vocab,
    )
