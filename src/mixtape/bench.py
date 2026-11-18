"""CPU vs GPU benchmark - the headline numbers for the project.

Runs the same pipeline stages on every available backend and records wall time.
On Windows (no RAPIDS) only the CPU column is filled; run on Colab/WSL2 for both.
"""
from __future__ import annotations

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


def run(cfg: dict) -> dict:
    backends = ["cpu"] + (["gpu"] if gpu_available() else [])
    p = cfg["paths"]
    results: dict[str, dict] = {}
    for name in backends:
        be = get_backend(name)
        r: dict[str, float] = {}
        print(f"[bench] backend={name}")
        stats = prep(be, p["interim"], p["processed"] / f"_bench_{name}", **cfg["prep"])
        r.update({f"prep.{k}": v for k, v in stats["timings_s"].items()})
        r["prep.total"] = stats["total_s"]

        data = load_dataset(p["processed"])
        t0 = time.perf_counter()
        m = CoOccurrence(be, batch_size=cfg["cooccur"]["batch_size"]).fit(data)
        _sync(be)
        r["cooccur.fit"] = round(time.perf_counter() - t0, 4)
        t0 = time.perf_counter()
        m.recommend(data.seeds, cfg["eval"]["k"])
        _sync(be)
        r["cooccur.recommend"] = round(time.perf_counter() - t0, 4)

        _, g = analyze(be, data, cfg["graph"]["top_n_tracks"], cfg["graph"]["min_edge_weight"])
        r["graph.build"] = g["build_s"]
        r["graph.louvain+pagerank"] = g["algorithms_s"]
        results[name] = r

    report = {"rows": stats["interactions"], "results": results}
    write_markdown(report, p["reports"] / "benchmark.md")
    with open(p["reports"] / "benchmark.json", "w") as f:
        json.dump(report, f, indent=2)
    return report


def write_markdown(report: dict, path: Path) -> None:
    res = report["results"]
    stages = list(res["cpu"].keys())
    lines = [f"# CPU vs GPU benchmark ({report['rows']:,} interactions)", "",
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
