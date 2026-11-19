"""CPU vs GPU benchmark - the headline numbers for the project.

Runs the same pipeline stages on every available backend and records wall time.
On Windows (no RAPIDS) only the CPU column is filled; run on Colab/WSL2 for both.
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

from mixtape.backend import get_backend, gpu_available
from mixtape.data.dataset import load_dataset
from mixtape.data.prep import prep
from mixtape.graph import analyze
from mixtape.models.cooccur import CoOccurrence


def _sync(be):
    if be.is_gpu:
        be.xp.cuda.Device().synchronize()


def _settings(cfg: dict, repeats: int) -> dict:
    """Optional `bench:` section in the config, for data too big to run everything
    several times on a CPU (the full MPD): fewer passes, a subset of playlists to
    time recommendation on, and a smaller graph (networkx can't hold 11M edges)."""
    b = cfg.get("bench", {})
    return {
        "repeats": {"cpu": b.get("repeats_cpu", repeats), "gpu": b.get("repeats_gpu", repeats)},
        "n_recommend": b.get("n_recommend"),
        "graph_top_n": b.get("graph_top_n", cfg["graph"]["top_n_tracks"]),
        "graph_min_w": b.get("graph_min_edge_weight", cfg["graph"]["min_edge_weight"]),
    }


def _one_pass(be, cfg: dict, s: dict, out_dir: Path) -> tuple[dict[str, float], int]:
    p = cfg["paths"]
    r: dict[str, float] = {}
    stats = prep(be, p["interim"], out_dir, **cfg["prep"])
    _sync(be)
    r.update({f"prep.{k}": v for k, v in stats["timings_s"].items()})
    r["prep.total"] = stats["total_s"]
    print(f"  prep {stats['total_s']:.1f}s", flush=True)

    data = load_dataset(out_dir)
    t0 = time.perf_counter()
    m = CoOccurrence(be, batch_size=cfg["cooccur"]["batch_size"]).fit(data)
    _sync(be)
    r["cooccur.fit"] = round(time.perf_counter() - t0, 4)
    seeds = data.seeds[:s["n_recommend"]] if s["n_recommend"] else data.seeds
    t0 = time.perf_counter()
    m.recommend(seeds, cfg["eval"]["k"])
    _sync(be)
    r["cooccur.recommend"] = round(time.perf_counter() - t0, 4)
    print(f"  cooccur fit {r['cooccur.fit']:.1f}s, recommend {r['cooccur.recommend']:.1f}s", flush=True)
    del m
    gc.collect()

    _, g = analyze(be, data, s["graph_top_n"], s["graph_min_w"])
    r["graph.build"] = g["build_s"]
    r["graph.louvain+pagerank"] = g["algorithms_s"]
    print(f"  graph build {g['build_s']:.1f}s, louvain+pagerank {g['algorithms_s']:.1f}s "
          f"({g['edges']:,} edges)", flush=True)
    return r, stats["interactions"]


def run(cfg: dict, repeats: int = 3) -> dict:
    """Time every stage `repeats` times per backend and keep the fastest run.

    The first GPU pass pays one-off costs (CUDA context, kernel JIT/caching);
    taking the best of several passes measures steady-state throughput, which
    is what matters when the pipeline runs repeatedly or on large data.
    """
    s = _settings(cfg, repeats)
    backends = ["cpu"] + (["gpu"] if gpu_available() else [])
    p = cfg["paths"]
    results: dict[str, dict] = {}
    first_pass: dict[str, dict] = {}
    for name in backends:
        be = get_backend(name)
        runs = []
        n = s["repeats"][name]
        for i in range(n):
            print(f"[bench] backend={name} pass {i + 1}/{n}", flush=True)
            r, rows = _one_pass(be, cfg, s, p["processed"] / f"_bench_{name}")
            runs.append(r)
            gc.collect()
        first_pass[name] = runs[0]
        results[name] = {k: min(run[k] for run in runs) for k in runs[0]}

    report = {"rows": rows, "repeats": s["repeats"], "settings": s, "results": results,
              "first_pass": first_pass}
    write_markdown(report, p["reports"] / "benchmark.md")
    with open(p["reports"] / "benchmark.json", "w") as f:
        json.dump(report, f, indent=2)
    return report


def write_markdown(report: dict, path: Path) -> None:
    res = report["results"]
    stages = list(res["cpu"].keys())
    rep = report.get("repeats", 1)
    if isinstance(rep, dict):
        how = f"Best of {rep['cpu']} CPU / {rep['gpu']} GPU passes."
    else:
        how = f"Best of {rep} passes per backend (first GPU pass includes CUDA start-up and is excluded)."
    s = report.get("settings", {})
    extra = []
    if s.get("n_recommend"):
        extra.append(f"`cooccur.recommend` times {s['n_recommend']:,} test playlists.")
    if s.get("graph_top_n"):
        extra.append(f"The graph uses the top {s['graph_top_n']:,} tracks, edges with weight ≥ "
                     f"{s['graph_min_w']}.")
    lines = [f"# CPU vs GPU benchmark ({report['rows']:,} interactions)", "",
             " ".join([how, *extra]), "",
             "| stage | CPU (s) | GPU (s) | speedup |", "|---|---:|---:|---:|"]
    for s in stages:
        cpu = res["cpu"][s]
        gpu = res.get("gpu", {}).get(s)
        speed = f"{cpu / gpu:.1f}x" if gpu else "-"
        lines.append(f"| {s} | {cpu:.3f} | {gpu if gpu is not None else '-'} | {speed} |")
    if "gpu" not in res:
        lines += ["", "_GPU column empty: RAPIDS not installed here. "
                      "Run `python -m mixtape bench` on Colab or WSL2._"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
