"""Cold-start comparison: TF-IDF char n-grams vs sentence-transformer title encoders.

Run:  python scripts/compare_title_encoders.py [model ...]
Writes reports/title_encoders.json.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from mixtape.config import load_config
from mixtape.data.dataset import load_dataset
from mixtape.eval.metrics import evaluate
from mixtape.models.title import TitleRecommender

MODELS = sys.argv[1:] or ["sentence-transformers/all-MiniLM-L6-v2",
                          "sentence-transformers/all-mpnet-base-v2"]

# Unseen phrasings, one per synthetic mood (same order as synthetic.MOOD_TITLES).
# None of them reuse the training title words, so string matching can't cheat.
PARAPHRASES = [
    "weightlifting session", "laid back afternoon", "stormy weather tunes",
    "saturday night out", "concentration music for the library", "driving across the country",
    "songs for a breakup", "hits from when i was a kid", "july heatwave", "falling asleep",
    "songs for my girlfriend", "marathon training", "cowboy hats and pickup trucks",
    "hardest verses", "techno club night", "electric guitar anthems", "saxophone lounge",
    "korean pop groups", "santa and sleigh bells", "sunday service hymns",
    "spanish dance hits", "alternative singer songwriter", "thrash and death growls",
    "mood booster",
]


def mood_precision(tm: TitleRecommender, data, k: int) -> float:
    """Share of top-k tracks that belong to the query's mood (synthetic data only:
    a track's mood is encoded in its artist id as mood*1000 + n)."""
    import pandas as pd
    cfg = load_config()
    tracks = pd.read_parquet(cfg["paths"]["interim"] / "tracks.parquet")
    art = data.vocab.merge(tracks, on="tid").set_index("item")["artist_name"]
    mood_of = (art.str.extract(r"(\d+)$")[0].astype(int) // 1000).to_dict()
    recs = tm.recommend_titles(PARAPHRASES, k)
    return float(np.mean([[mood_of.get(int(i), -1) == m for i in row]
                          for m, row in enumerate(recs)]))


def main():
    cfg = load_config()
    data = load_dataset(cfg["paths"]["processed"])
    k = cfg["eval"]["k"]
    titles = [data.titles.get(p, "") for p in data.test_pids]
    targets = [np.concatenate([s, t]) for s, t in zip(data.seeds, data.targets)]
    out = {}
    for name in [None, *MODELS]:
        t0 = time.perf_counter()
        tm = TitleRecommender(name, cfg["title"]["top_m"], prefer_transformer=name is not None).fit(data)
        recs = tm.recommend_titles(titles, k)
        res = {**evaluate(recs, targets, k),
               f"unseen_phrasing_mood_precision@{k}": round(mood_precision(tm, data, k), 4),
               "seconds": round(time.perf_counter() - t0, 1),
               "unique_titles": int(len(tm.uniq))}
        out[tm.encoder.name] = res
        print(tm.encoder.name, res, flush=True)
        for q in ["workout", "songs for a breakup", "late night drive"]:
            print(f"   {q!r} -> {tm.similar_titles(q, 3)}")
    with open(cfg["paths"]["reports"] / "title_encoders.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
