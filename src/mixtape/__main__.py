"""mixtape command line.

  python -m mixtape synth            # make synthetic MPD-format data
  python -m mixtape ingest           # json slices -> parquet
  python -m mixtape prep             # clean / index / split (pandas or cuDF)
  python -m mixtape train            # train + evaluate models
  python -m mixtape graph            # cuGraph/networkx track communities
  python -m mixtape bench            # CPU vs GPU timings
  python -m mixtape demo --title "rainy sunday"
  python -m mixtape all              # synth -> ingest -> prep -> train -> graph
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np


def cmd_synth(cfg, args):
    from mixtape.data.synthetic import generate
    paths = generate(cfg["paths"]["raw"], **cfg["synthetic"])
    print(f"wrote {len(paths)} synthetic slices to {cfg['paths']['raw']}")


def cmd_ingest(cfg, args):
    from mixtape.data.ingest import ingest
    print(ingest(cfg["paths"]["raw"], cfg["paths"]["interim"], args.max_slices))


def cmd_prep(cfg, args):
    from mixtape.backend import get_backend
    from mixtape.data.prep import prep
    stats = prep(get_backend(), cfg["paths"]["interim"], cfg["paths"]["processed"], **cfg["prep"])
    print(json.dumps(stats, indent=2))


def _build(name, cfg, tfidf: bool = False):
    from mixtape.backend import get_backend
    from mixtape.models.cooccur import CoOccurrence
    from mixtape.models.popularity import Popularity
    from mixtape.models.sasrec import SASRec
    if name == "popularity":
        return Popularity()
    if name == "cooccur":
        return CoOccurrence(get_backend(), batch_size=cfg["cooccur"]["batch_size"])
    if name == "sasrec":
        return SASRec(**cfg["sasrec"])
    if name == "hybrid":
        from mixtape.models.hybrid import Hybrid
        from mixtape.models.title import TitleRecommender
        h = cfg["hybrid"]
        title = None
        if h.get("title_weight", 0) > 0:
            title = TitleRecommender(cfg["title"]["model"], cfg["title"]["top_m"],
                                     prefer_transformer=not tfidf)
        return Hybrid([_build("cooccur", cfg), _build("sasrec", cfg)], h["weights"],
                      title_model=title, title_weight=h.get("title_weight", 0))
    raise ValueError(name)


def cmd_train(cfg, args):
    from mixtape.data.dataset import load_dataset
    from mixtape.eval.metrics import evaluate
    from mixtape.models.popularity import Popularity
    from mixtape.models.title import TitleRecommender

    data = load_dataset(cfg["paths"]["processed"])
    k = cfg["eval"]["k"]
    print(f"items={data.n_items:,}  train rows={len(data.train):,}  test playlists={len(data.test_pids):,}")
    results = {"continuation": {}, "cold_start": {}}

    for name in args.models:
        if name == "title":
            continue
        print(f"\n[{name}] training...")
        t0 = time.perf_counter()
        model = _build(name, cfg, tfidf=args.tfidf).fit(data)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        if name == "hybrid":
            titles = [data.titles.get(p, "") for p in data.test_pids]
            recs = model.recommend(data.seeds, k, titles=titles)
        else:
            recs = model.recommend(data.seeds, k)
        rec_s = time.perf_counter() - t0
        results["continuation"][name] = {**evaluate(recs, data.targets, k),
                                         "fit_s": round(fit_s, 2), "recommend_s": round(rec_s, 2)}
        print("  ", results["continuation"][name])

    if "title" in args.models:
        print("\n[title] cold start: recommend from the playlist name only")
        all_targets = [np.concatenate([s, t]) for s, t in zip(data.seeds, data.targets)]
        titles = [data.titles.get(p, "") for p in data.test_pids]
        empty = [np.array([], dtype=np.int64)] * len(titles)
        pop_recs = Popularity().fit(data).recommend(empty, k)
        results["cold_start"]["popularity"] = evaluate(pop_recs, all_targets, k)
        tm = TitleRecommender(cfg["title"]["model"], cfg["title"]["top_m"],
                              prefer_transformer=not args.tfidf).fit(data)
        results["cold_start"][f"title ({tm.encoder.name})"] = evaluate(
            tm.recommend_titles(titles, k), all_targets, k)
        for n, r in results["cold_start"].items():
            print(f"   {n}: {r}")

    out = cfg["paths"]["reports"] / "results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {out}")


def cmd_graph(cfg, args):
    from mixtape.backend import get_backend
    from mixtape.data.dataset import load_dataset
    from mixtape.graph import analyze
    import pandas as pd

    data = load_dataset(cfg["paths"]["processed"])
    nodes, summary = analyze(get_backend(), data, cfg["graph"]["top_n_tracks"],
                             cfg["graph"]["min_edge_weight"])
    tracks = pd.read_parquet(cfg["paths"]["interim"] / "tracks.parquet")
    nodes = nodes.merge(data.vocab, on="item").merge(tracks, on="tid", how="left")
    nodes.to_parquet(cfg["paths"]["reports"] / "graph_nodes.parquet", index=False)
    print(json.dumps(summary, indent=2))
    big = nodes["community"].value_counts().head(5).index
    for c in big:
        top = nodes[nodes["community"] == c].nlargest(3, "pagerank")
        print(f"community {c} ({(nodes['community'] == c).sum()} tracks): "
              + ", ".join(f"{a} - {t}" for a, t in zip(top["artist_name"], top["track_name"])))


def cmd_bench(cfg, args):
    from mixtape.bench import run
    run(cfg)
    print((cfg["paths"]["reports"] / "benchmark.md").read_text(encoding="utf-8"))


def cmd_demo(cfg, args):
    import pandas as pd
    from mixtape.data.dataset import load_dataset
    from mixtape.models.title import TitleRecommender

    data = load_dataset(cfg["paths"]["processed"])
    tracks = pd.read_parquet(cfg["paths"]["interim"] / "tracks.parquet")
    names = data.vocab.merge(tracks, on="tid").set_index("item")
    tm = TitleRecommender(cfg["title"]["model"], cfg["title"]["top_m"],
                          prefer_transformer=not args.tfidf).fit(data)
    print(f'Similar titles to "{args.title}": {tm.similar_titles(args.title)}')
    recs = tm.recommend_titles([args.title], args.k)[0]
    for i, it in enumerate(recs, 1):
        r = names.loc[it]
        print(f"{i:2d}. {r['artist_name']} - {r['track_name']}")


def cmd_all(cfg, args):
    for fn in (cmd_synth, cmd_ingest, cmd_prep, cmd_train, cmd_graph):
        print(f"\n===== {fn.__name__[4:]} =====")
        fn(cfg, args)


def main(argv=None):
    from mixtape.config import load_config

    ap = argparse.ArgumentParser(prog="mixtape")
    ap.add_argument("command", choices=["synth", "ingest", "prep", "train", "graph",
                                        "bench", "demo", "all"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", nargs="+",
                    default=["popularity", "cooccur", "sasrec", "hybrid", "title"])
    ap.add_argument("--max-slices", type=int, default=None, help="ingest only N MPD slices")
    ap.add_argument("--title", default="rainy sunday")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--tfidf", action="store_true", help="use TF-IDF instead of a transformer")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    globals()[f"cmd_{args.command}"](cfg, args)


if __name__ == "__main__":
    main()
