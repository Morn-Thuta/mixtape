"""Collect the files the demo app needs into data/space_bundle/, mirroring repo paths.

The bundle is uploaded to a PRIVATE Hugging Face dataset repo (the Spotify MPD terms
don't allow re-sharing the data); the public Space downloads it at start-up with a
read token stored as a Space secret (see deploy/hf-space/README.md).

Run:  python scripts/build_space_bundle.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "space_bundle"

PROCESSED = ["train.parquet", "vocab.parquet", "playlists.parquet", "test_seed.parquet",
             "test_target.parquet", "stats.json"]


def main() -> None:
    src = ROOT / "data" / "mpd" / "processed"
    dst = OUT / "data" / "mpd" / "processed"
    dst.mkdir(parents=True, exist_ok=True)
    for name in PROCESSED + [p.name for p in src.glob("title_emb_*.npy")]:
        shutil.copy2(src / name, dst / name)

    # tracks: only the songs in the vocabulary, only the columns the app shows
    vocab = pd.read_parquet(src / "vocab.parquet", columns=["tid"])
    tracks = pd.read_parquet(ROOT / "data" / "mpd" / "interim" / "tracks.parquet",
                             columns=["tid", "track_name", "artist_name"])
    tracks = tracks[tracks["tid"].isin(vocab["tid"])]
    (OUT / "data" / "mpd" / "interim").mkdir(parents=True, exist_ok=True)
    tracks.to_parquet(OUT / "data" / "mpd" / "interim" / "tracks.parquet", index=False)

    (OUT / "reports" / "mpd").mkdir(parents=True, exist_ok=True)
    nodes = pd.read_parquet(ROOT / "reports" / "mpd" / "graph_nodes.parquet",
                            columns=["item", "community", "pagerank"])
    nodes.to_parquet(OUT / "reports" / "mpd" / "graph_nodes.parquet", index=False)

    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"bundle: {OUT}  ({total / 2**20:.0f} MB)")
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(OUT).as_posix():55s} {p.stat().st_size / 2**20:7.1f} MB")


if __name__ == "__main__":
    main()
