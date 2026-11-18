"""Does the playlist title help continuation? Compare models with 1 vs 5 seed tracks.

Run:  python scripts/hybrid_title_experiment.py      -> reports/hybrid_title.json
"""
from __future__ import annotations

import json

import numpy as np

from mixtape.backend import get_backend
from mixtape.config import load_config
from mixtape.data.dataset import load_dataset
from mixtape.eval.metrics import evaluate
from mixtape.models.cooccur import CoOccurrence
from mixtape.models.hybrid import fuse
from mixtape.models.sasrec import SASRec
from mixtape.models.title import TitleRecommender


def main():
    cfg = load_config()
    d = load_dataset(cfg["paths"]["processed"])
    k, depth = cfg["eval"]["k"], 200
    co = CoOccurrence(get_backend(), batch_size=cfg["cooccur"]["batch_size"]).fit(d)
    sa = SASRec(**cfg["sasrec"]).fit(d)
    tm = TitleRecommender(cfg["title"]["model"], cfg["title"]["top_m"]).fit(d)
    titles = [d.titles.get(p, "") for p in d.test_pids]
    out = {}
    for n_seed in (1, 5):
        seeds = [s[:n_seed] for s in d.seeds]
        # tracks hidden from the model are targets too
        targets = [np.concatenate([s[n_seed:], t]) for s, t in zip(d.seeds, d.targets)]
        L = {"cooccur": co.recommend(seeds, depth), "sasrec": sa.recommend(seeds, depth),
             "title": tm.recommend_titles(titles, depth, exclude=seeds)}
        res = {name: evaluate(r, targets, k) for name, r in L.items()}
        for tw in (0.25, 0.5, 1.0):
            res[f"cooccur+sasrec+title(w={tw})"] = evaluate(
                fuse([L["cooccur"], L["sasrec"], L["title"]], [1.0, 0.5, tw], d.n_items, k),
                targets, k)
        res["cooccur+sasrec"] = evaluate(
            fuse([L["cooccur"], L["sasrec"]], [1.0, 0.5], d.n_items, k), targets, k)
        out[f"{n_seed}_seed"] = res
        print(f"\n== {n_seed} seed track(s) ==")
        for name, r in res.items():
            print(f"  {name:32s} recall@{k}={r[f'recall@{k}']:.4f}  ndcg@{k}={r[f'ndcg@{k}']:.4f}")
    with open(cfg["paths"]["reports"] / "hybrid_title.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
