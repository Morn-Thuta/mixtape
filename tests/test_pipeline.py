"""End-to-end smoke tests on a tiny synthetic dataset (CPU backend)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from mixtape.backend import get_backend
from mixtape.data.dataset import load_dataset
from mixtape.data.ingest import ingest
from mixtape.data.prep import prep
from mixtape.data.synthetic import generate
from mixtape.eval.metrics import evaluate, ndcg_at_k, recall_at_k
from mixtape.graph import analyze
from mixtape.models.cooccur import CoOccurrence
from mixtape.models.popularity import Popularity
from mixtape.models.sasrec import SASRec, SASRecNet
from mixtape.models.title import TitleRecommender


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp("mpd")
    generate(root / "raw", n_playlists=600, n_tracks=400, n_moods=6, slice_size=200, seed=1)
    ingest(root / "raw", root / "interim")
    stats = prep(get_backend("cpu"), root / "interim", root / "processed",
                 min_track_count=2, min_playlist_len=8, test_frac=0.1, n_seed=5)
    assert stats["n_test_playlists"] > 0
    return load_dataset(root / "processed")


def test_split_is_consistent(data):
    assert len(data.seeds) == len(data.targets) == len(data.test_pids)
    assert all(len(s) == 5 for s in data.seeds)
    assert data.train["item"].min() >= 1
    # targets of test playlists never leak into train
    leak = data.train[data.train["pid"].isin(data.test_pids)]
    assert len(leak) == 5 * len(data.test_pids)


def test_metrics():
    recs = np.array([[1, 2, 3], [4, 5, 6]])
    targets = [np.array([1, 9]), np.array([7])]
    assert recall_at_k(recs, targets, 3) == pytest.approx(0.25)
    assert ndcg_at_k(np.array([[1, 2]]), [np.array([1])], 2) == pytest.approx(1.0)


@pytest.mark.parametrize("model", [Popularity(), CoOccurrence(get_backend("cpu"))])
def test_models_exclude_seeds(data, model):
    recs = model.fit(data).recommend(data.seeds, 20)
    assert recs.shape == (len(data.seeds), 20)
    for r, s in zip(recs, data.seeds):
        assert not np.isin(r, s).any() and 0 not in r


def test_cooccur_beats_popularity(data):
    pop = evaluate(Popularity().fit(data).recommend(data.seeds, 20), data.targets, 20)
    co = evaluate(CoOccurrence(get_backend("cpu")).fit(data).recommend(data.seeds, 20),
                  data.targets, 20)
    assert co["recall@20"] > pop["recall@20"]


def test_sasrec_is_causal():
    net = SASRecNet(50, 8, 16, 2, 1, 0.0).eval()
    a = torch.tensor([[3, 4, 5, 6, 0, 0, 0, 0]])
    b = a.clone()
    b[0, 3] = 40
    assert torch.allclose(net(a)[0, :3], net(b)[0, :3])


def test_sasrec_runs(data):
    m = SASRec(max_len=20, dim=16, heads=2, layers=1, epochs=1, batch_size=64, device="cpu").fit(data)
    recs = m.recommend(data.seeds, 10)
    assert recs.shape == (len(data.seeds), 10)


def test_sasrec_window_targets_stay_in_playlist():
    m = SASRec(target_window=5, device="cpu")
    batch = torch.tensor([[1, 2, 3, 4, 0, 0], [5, 6, 7, 8, 9, 10]])
    inp, tgt, mask = m._targets(batch)
    assert mask.sum().item() == 3 + 5
    for row, t, mk in zip(batch, tgt, mask):
        real = set(row[row > 0].tolist())
        assert set(t[mk].tolist()) <= real
    # each target comes strictly after its input position
    assert (tgt[1][mask[1]] > inp[1][mask[1]]).all()


def test_sasrec_early_stopping_restores_best(data):
    m = SASRec(max_len=20, dim=16, heads=2, layers=1, epochs=6, batch_size=64, device="cpu",
               val_size=20, eval_every=2, patience=1).fit(data)
    assert m.best_epoch is not None
    assert any("val_recall@50" in h for h in m.history)


def test_fuse_prefers_agreement():
    from mixtape.models.hybrid import fuse
    a = np.array([[1, 2, 3]])
    b = np.array([[3, 4, 5]])
    assert fuse([a, b], [1.0, 1.0], n_items=5, k=1)[0, 0] == 3


def test_hybrid_with_title_excludes_seeds(data):
    from mixtape.models.hybrid import Hybrid
    h = Hybrid([Popularity(), CoOccurrence(get_backend("cpu"))], [0.5, 1.0],
               title_model=TitleRecommender(prefer_transformer=False), title_weight=1.0).fit(data)
    titles = [data.titles.get(p, "") for p in data.test_pids]
    recs = h.recommend(data.seeds, 20, titles=titles)
    assert recs.shape == (len(data.seeds), 20)
    for r, s in zip(recs, data.seeds):
        assert not np.isin(r, s).any() and 0 not in r


def test_with_validation_is_disjoint(data):
    val = data.with_validation(20)
    assert len(val.test_pids) == 20
    assert not set(val.test_pids) & set(data.test_pids)
    # validation targets are gone from train; their seeds stay
    tr = val.train[val.train["pid"].isin(val.test_pids)]
    assert (tr["rank"] < 5).all() and len(tr) == 5 * 20
    assert all(len(s) == 5 and len(t) > 0 for s, t in zip(val.seeds, val.targets))


def test_title_cold_start(data):
    tm = TitleRecommender(prefer_transformer=False).fit(data)
    recs = tm.recommend_titles(["gym 💪", "chill vibes"], 10)
    assert recs.shape == (2, 10)
    assert tm.similar_titles("GYM!!")[0][0] == "gym"


def test_graph(data):
    nodes, summary = analyze(get_backend("cpu"), data, top_n=200, min_w=2)
    assert summary["communities"] >= 2 and len(nodes) > 0
