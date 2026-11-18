"""Clean, index and split the interactions - runs on pandas OR cuDF.

This is the step that benefits most from cuDF on the full 66M-row MPD:
group-bys, filters, joins and sorts, all written once against the
pandas/cuDF common API.

Split ("playlist continuation", like the RecSys Challenge 2018):
  * a random `test_frac` of playlists is held out
  * for each held-out playlist the first `n_seed` tracks are the input,
    the remaining tracks are the targets to predict
  * train = all other playlists + the seed part of test playlists
Track ids are 1..n_items (0 is reserved for padding in SASRec).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from mixtape.backend import Backend


def prep(be: Backend, interim: Path, out_dir: Path, min_track_count=3,
         min_playlist_len=10, test_frac=0.05, n_seed=5, seed=42) -> dict:
    xdf = be.df
    timings: dict[str, float] = {}

    def tick(name, t0):
        timings[name] = round(time.perf_counter() - t0, 4)

    t0 = time.perf_counter()
    df = xdf.read_parquet(interim / "interactions.parquet")
    playlists = xdf.read_parquet(interim / "playlists.parquet")
    tick("read", t0)
    n_raw = len(df)

    # 1. drop rare tracks (count distinct playlists per track)
    t0 = time.perf_counter()
    df = df.drop_duplicates(subset=["pid", "track_uri"])
    counts = df.groupby("track_uri").size().reset_index(name="cnt")
    keep = counts[counts["cnt"] >= min_track_count][["track_uri"]]
    df = df.merge(keep, on="track_uri", how="inner")
    tick("filter_tracks", t0)

    # 2. drop short playlists
    t0 = time.perf_counter()
    plen = df.groupby("pid").size().reset_index(name="plen")
    plen = plen[plen["plen"] >= min_playlist_len][["pid"]]
    df = df.merge(plen, on="pid", how="inner")
    tick("filter_playlists", t0)

    # 3. integer ids for tracks (1-based) and re-rank positions
    t0 = time.perf_counter()
    vocab = df[["track_uri"]].drop_duplicates().sort_values("track_uri").reset_index(drop=True)
    vocab["item"] = be.xp.arange(1, len(vocab) + 1, dtype="int32")
    df = df.merge(vocab, on="track_uri", how="inner")
    df = df.sort_values(["pid", "pos"]).reset_index(drop=True)
    df["rank"] = df.groupby("pid").cumcount().astype("int32")
    tick("index", t0)

    # 4. split
    t0 = time.perf_counter()
    pids = df[["pid"]].drop_duplicates()
    test_pids = pids.sample(frac=test_frac, random_state=seed)
    df["is_test"] = df["pid"].isin(test_pids["pid"])
    is_seed = df["rank"] < n_seed
    train = df[~df["is_test"] | is_seed]
    test_seed = df[df["is_test"] & is_seed]
    test_target = df[df["is_test"] & ~is_seed]
    cols = ["pid", "rank", "item"]
    train = train.sort_values(["pid", "rank"])[cols]
    test_seed = test_seed.sort_values(["pid", "rank"])[cols]
    test_target = test_target.sort_values(["pid", "rank"])[cols]
    tick("split", t0)

    # 5. write
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out_dir / "train.parquet", index=False)
    test_seed.to_parquet(out_dir / "test_seed.parquet", index=False)
    test_target.to_parquet(out_dir / "test_target.parquet", index=False)
    vocab.to_parquet(out_dir / "vocab.parquet", index=False)
    playlists.merge(df[["pid", "is_test"]].drop_duplicates(), on="pid", how="inner") \
        .to_parquet(out_dir / "playlists.parquet", index=False)
    tick("write", t0)

    stats = {
        "backend": be.name,
        "raw_interactions": int(n_raw),
        "interactions": int(len(df)),
        "n_items": int(len(vocab)),
        "n_playlists": int(len(pids)),
        "n_test_playlists": int(len(test_pids)),
        "timings_s": timings,
        "total_s": round(sum(timings.values()), 4),
    }
    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    return stats
