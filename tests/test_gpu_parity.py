"""GPU (RAPIDS) path must give the same answers as the CPU path.

Skipped automatically where RAPIDS isn't installed (e.g. plain Windows).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mixtape.backend import get_backend, gpu_available

pytestmark = pytest.mark.skipif(not gpu_available(), reason="RAPIDS not installed")

from mixtape.data.dataset import load_dataset  # noqa: E402
from mixtape.data.ingest import ingest  # noqa: E402
from mixtape.data.prep import prep  # noqa: E402
from mixtape.data.synthetic import generate  # noqa: E402
from mixtape.graph import analyze  # noqa: E402
from mixtape.models.cooccur import CoOccurrence  # noqa: E402

PREP = dict(min_track_count=2, min_playlist_len=8, test_frac=0.1, n_seed=5, seed=3)


@pytest.fixture(scope="module")
def dirs(tmp_path_factory):
    root = tmp_path_factory.mktemp("parity")
    generate(root / "raw", n_playlists=800, n_tracks=500, n_moods=6, slice_size=400, seed=7)
    ingest(root / "raw", root / "interim")
    prep(get_backend("cpu"), root / "interim", root / "cpu", **PREP)
    prep(get_backend("gpu"), root / "interim", root / "gpu", **PREP)
    return root


def test_prep_same_vocab_and_counts(dirs):
    for f in ["vocab.parquet", "train.parquet"]:
        a = pd.read_parquet(dirs / "cpu" / f)
        b = pd.read_parquet(dirs / "gpu" / f)
        assert len(a) == len(b), f
    va = pd.read_parquet(dirs / "cpu" / "vocab.parquet").sort_values("track_uri")
    vb = pd.read_parquet(dirs / "gpu" / "vocab.parquet").sort_values("track_uri")
    assert (va["item"].to_numpy() == vb["item"].to_numpy()).all()


def test_cooccur_gpu_matches_cpu(dirs):
    data = load_dataset(dirs / "cpu")
    cpu = CoOccurrence(get_backend("cpu")).fit(data).recommend(data.seeds, 10)
    gpu = CoOccurrence(get_backend("gpu")).fit(data).recommend(data.seeds, 10)
    # ties may be ordered differently; compare as sets and require strong overlap
    overlap = np.mean([len(np.intersect1d(a, b)) / 10 for a, b in zip(cpu, gpu)])
    assert overlap > 0.9


def test_graph_gpu_runs(dirs):
    data = load_dataset(dirs / "cpu")
    nodes, summary = analyze(get_backend("gpu"), data, top_n=300, min_w=2)
    _, cpu_summary = analyze(get_backend("cpu"), data, top_n=300, min_w=2)
    assert summary["edges"] == cpu_summary["edges"]
    assert summary["communities"] >= 2
    assert {"item", "community", "pagerank"} <= set(nodes.columns)
