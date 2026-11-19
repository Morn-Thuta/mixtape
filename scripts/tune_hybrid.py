"""Tune hybrid fusion weights on a validation split, then report the test score once.

1. Carve `n_val` validation playlists out of the training data (Dataset.with_validation).
2. Fit co-occurrence, SASRec and the title model on the remaining training data.
3. Grid-search the SASRec and title weights on validation ndcg@k.
4. Evaluate the chosen weights on the real test set (the only time test is touched).

Run:  python scripts/tune_hybrid.py --config configs/mpd50.yaml [--n-val 2000]
Writes <reports>/hybrid_tuning.json
"""
from __future__ import annotations

import argparse
import itertools
import json

from mixtape.backend import get_backend
from mixtape.config import load_config
from mixtape.data.dataset import load_dataset
from mixtape.eval.metrics import evaluate
from mixtape.models.cooccur import CoOccurrence
from mixtape.models.hybrid import fuse
from mixtape.models.sasrec import SASRec
from mixtape.models.title import TitleRecommender

SASREC_W = [0.0, 0.1, 0.25, 0.5, 1.0]
TITLE_W = [0.0, 0.1, 0.25, 0.5, 1.0]


def ranked_lists(models, d, depth):
    co, sa, tm = models
    titles = [d.titles.get(p, "") for p in d.test_pids]
    return [co.recommend(d.seeds, depth), sa.recommend(d.seeds, depth),
            tm.recommend_titles(titles, depth, exclude=d.seeds)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--depth", type=int, default=200)
    args = ap.parse_args()
    cfg = load_config(args.config)
    k = cfg["eval"]["k"]
    data = load_dataset(cfg["paths"]["processed"])
    val = data.with_validation(args.n_val)
    print(f"validation playlists: {len(val.test_pids):,}   test playlists: {len(data.test_pids):,}")

    models = (CoOccurrence(get_backend(), batch_size=cfg["cooccur"]["batch_size"]).fit(val),
              SASRec(**cfg["sasrec"]).fit(val),
              TitleRecommender(cfg["title"]["model"], cfg["title"]["top_m"]).fit(val))

    L = ranked_lists(models, val, args.depth)
    grid = []
    for ws, wt in itertools.product(SASREC_W, TITLE_W):
        r = evaluate(fuse(L, [1.0, ws, wt], val.n_items, k), val.targets, k)
        grid.append({"w_sasrec": ws, "w_title": wt, **r})
        print(f"  val  w_sasrec={ws:<5} w_title={wt:<5} recall@{k}={r[f'recall@{k}']:.4f} "
              f"ndcg@{k}={r[f'ndcg@{k}']:.4f}")
    best = max(grid, key=lambda g: g[f"ndcg@{k}"])
    print(f"best on validation: w_sasrec={best['w_sasrec']} w_title={best['w_title']}")

    # test: touched once, with the chosen weights (plus the single models for reference)
    T = ranked_lists(models, data, args.depth)
    test = {
        "cooccur": evaluate(T[0][:, :k], data.targets, k),
        "sasrec": evaluate(T[1][:, :k], data.targets, k),
        "hybrid_tuned": evaluate(fuse(T, [1.0, best["w_sasrec"], best["w_title"]],
                                      data.n_items, k), data.targets, k),
    }
    for n, r in test.items():
        print(f"  test {n:13s} recall@{k}={r[f'recall@{k}']:.4f} ndcg@{k}={r[f'ndcg@{k}']:.4f}")
    out = {"n_val": len(val.test_pids), "best": best, "grid": grid, "test": test}
    with open(cfg["paths"]["reports"] / "hybrid_tuning.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
