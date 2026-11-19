"""Precompute the MPNet embeddings of all training playlist titles (run once, on the GPU).

The demo app (app/streamlit_app.py) then loads them from <processed>/title_emb_*.npy
instead of encoding ~100k titles on the CPU.

Run:  python scripts/build_demo_cache.py [--config configs/mpd.yaml]
"""
from __future__ import annotations

import argparse
import time

from mixtape.config import load_config
from mixtape.data.dataset import load_dataset
from mixtape.models.title import TitleRecommender

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/mpd.yaml")
args = ap.parse_args()
cfg = load_config(args.config)
t0 = time.perf_counter()
data = load_dataset(cfg["paths"]["processed"])
tm = TitleRecommender(cfg["title"]["model"], cfg["title"]["top_m"]).fit(
    data, cache_dir=cfg["paths"]["processed"])
print(f"{len(tm.uniq):,} unique titles embedded with {tm.encoder.name} "
      f"in {time.perf_counter() - t0:.0f}s -> {cfg['paths']['processed']}")
print("check:", tm.similar_titles("late night drive", 5))
