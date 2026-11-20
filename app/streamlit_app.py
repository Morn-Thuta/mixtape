"""mixtape demo: build a playlist from a name, continue one from a few songs,
or browse the music scenes cuGraph found.

Run from the repo root:
    pip install -r requirements-app.txt -r requirements-nlp.txt
    streamlit run app/streamlit_app.py

Two modes:
* full: the prepared Spotify MPD (configs/mpd.yaml) or the synthetic data, models fitted
  at start-up (~5 GB RAM for the MPD). Run `python scripts/build_demo_cache.py` once on
  the GPU first so MPNet doesn't encode every playlist title on the CPU.
* lite: a precomputed bundle in data/lite (`python -m mixtape lite`, ~1 GB RAM), used
  when it exists. Hosted copies set MIXTAPE_DATA_REPO (a private Hugging Face dataset
  holding lite/*) and HF_TOKEN, and the bundle is downloaded on first start.
MIXTAPE_CONFIG=configs/<name>.yaml forces full mode with that config.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # hosted copies run from a checkout without installing

from mixtape.models.hybrid import fuse  # noqa: E402

st.set_page_config(page_title="mixtape", page_icon="🎧", layout="wide")
N_PICKABLE = 20_000  # most popular tracks offered in the song picker
LITE = ROOT / "data" / "lite"


def _secret(name: str) -> str | None:
    if os.environ.get(name):
        return os.environ[name]
    try:
        return st.secrets.get(name)
    except Exception:  # no secrets file
        return None


def fetch_lite() -> None:
    """Hosted deployment: download the lite bundle from the private dataset repo."""
    repo = _secret("MIXTAPE_DATA_REPO")
    if not repo or (LITE / "stats.json").exists():
        return
    from huggingface_hub import snapshot_download
    with st.spinner("Downloading the playlist data (first start only)…"):
        snapshot_download(repo, repo_type="dataset", allow_patterns=["lite/*"],
                          local_dir=ROOT / "data", token=_secret("HF_TOKEN"))


def choose_mode() -> tuple[str, str]:
    if os.environ.get("MIXTAPE_CONFIG"):
        return "full", os.environ["MIXTAPE_CONFIG"]
    fetch_lite()
    if (LITE / "stats.json").exists():
        return "lite", str(LITE)
    mpd = ROOT / "data" / "mpd" / "processed" / "train.parquet"
    return "full", str(ROOT / "configs" / ("mpd.yaml" if mpd.exists() else "default.yaml"))


@st.cache_resource(show_spinner="Loading playlists and models…")
def load(mode: str, where: str, tfidf_only: bool):
    """Returns (info, names, co, tm, pop, scenes_path); co / tm share one interface in
    both modes: co.recommend(seeds, k), tm.recommend_titles(...), tm.similar_titles(...)."""
    if mode == "lite":
        from mixtape.lite import load_lite
        stats, names, co, tm, pop = load_lite(Path(where))
        info = {**stats, "title_weight": 0.5, "label": "lite (precomputed from all playlists)"}
        return info, names, co, tm, pop, str(Path(where) / "graph_nodes.parquet")

    from mixtape.backend import get_backend
    from mixtape.config import load_config
    from mixtape.data.dataset import load_dataset
    from mixtape.models.cooccur import CoOccurrence
    from mixtape.models.title import TitleRecommender
    cfg = load_config(where)
    data = load_dataset(cfg["paths"]["processed"])
    tracks = pd.read_parquet(cfg["paths"]["interim"] / "tracks.parquet",
                             columns=["tid", "track_name", "artist_name"])
    names = data.vocab.merge(tracks, on="tid").set_index("item").sort_index()
    names["label"] = names["artist_name"] + " - " + names["track_name"]
    co = CoOccurrence(get_backend(), batch_size=cfg["cooccur"]["batch_size"]).fit(data)
    tm = TitleRecommender(cfg["title"]["model"], cfg["title"]["top_m"],
                          prefer_transformer=not tfidf_only).fit(data, cache_dir=cfg["paths"]["processed"])
    pop = np.bincount(data.train["item"].to_numpy(), minlength=data.n_items + 1)
    info = {"tracks": data.n_items, "playlists": int(data.train["pid"].nunique()),
            "title_weight": cfg["hybrid"].get("title_weight", 0.5),
            "label": f"full · {os.path.basename(where)} · {get_backend().name}"}
    return info, names, co, tm, pop, str(cfg["paths"]["reports"] / "graph_nodes.parquet")


@st.cache_resource(show_spinner=False)
def load_scenes(path: str, _names):
    """Louvain communities from `python -m mixtape graph`, named after their three most
    central artists (summed PageRank of their tracks): one artist alone can mislabel a
    big mixed community."""
    if not Path(path).exists():
        return None
    nodes = pd.read_parquet(path, columns=["item", "community", "pagerank"])
    nodes = nodes[nodes["item"].isin(_names.index)]
    nodes["artist"] = _names["artist_name"].reindex(nodes["item"]).to_numpy()
    size = nodes["community"].value_counts()
    by_artist = nodes.groupby(["community", "artist"])["pagerank"].sum().reset_index() \
        .sort_values("pagerank", ascending=False)
    label = {c: " · ".join(g["artist"].head(3)) for c, g in by_artist.groupby("community", sort=False)}
    nodes["scene"] = nodes["community"].map(label)
    nodes["size"] = nodes["community"].map(size)
    return nodes.set_index("item")


def show(items: np.ndarray, names: pd.DataFrame, pop: np.ndarray, scenes) -> None:
    items = np.asarray([i for i in items if i in names.index], dtype=np.int64)
    df = names.loc[items, ["artist_name", "track_name"]].reset_index(drop=True)
    df.columns = ["Artist", "Track"]
    df.index += 1
    df["Playlists containing it"] = pop[items]
    if scenes is not None:
        df["Scene (cuGraph)"] = scenes["scene"].reindex(items).fillna("").to_numpy()
    st.dataframe(df, use_container_width=True)


st.title("🎧 mixtape")
st.caption("Playlist continuation on the Spotify Million Playlist Dataset · RAPIDS (cuDF, cuGraph) + transformers · "
           "[code and results](https://github.com/Morn-Thuta/mixtape)")
mode, where = choose_mode()
tfidf_only = mode == "full" and st.sidebar.toggle("TF-IDF only (no transformer)", value=False)
k = st.sidebar.slider("Songs to recommend", 5, 50, 20)
info, names, co, tm, pop, scenes_path = load(mode, where, tfidf_only)
scenes = load_scenes(scenes_path, names)
st.sidebar.markdown(
    f"**Mode:** {info['label']}  \n**Tracks:** {info['tracks']:,}  \n"
    f"**Playlists:** {info['playlists']:,}  \n**Title encoder:** {tm.encoder.name}")
st.sidebar.caption("Live demo uses co-occurrence + the title model, fused by reciprocal rank. "
                   "SASRec is evaluated offline (reports/mpd/results.json).")

tabs = st.tabs(["✍️ From a name", "🎵 From songs"] + (["🗺️ Scenes"] if scenes is not None else []))

with tabs[0]:
    st.write("Type any playlist name, even one nobody used: the encoder matches meaning, not just spelling.")
    title = st.text_input("Name your playlist", "late night drive")
    if title:
        st.write("Closest playlist names among 1M real playlists: "
                 + ", ".join(f"`{t}` ({s:.2f})" for t, s in tm.similar_titles(title, 6)))
        show(tm.recommend_titles([title], k)[0], names, pop, scenes)

with tabs[1]:
    order = np.argsort(-pop)
    order = order[(order > 0) & np.isin(order, names.index)][:N_PICKABLE]
    pickable = names.loc[order]
    picked = st.multiselect(f"Pick a few songs (search the {N_PICKABLE:,} most popular)",
                            pickable["label"].tolist(), max_selections=10)
    name_hint = st.text_input("Optional: playlist name, blended in with rank fusion", "")
    if picked:
        seed = pickable.index[pickable["label"].isin(picked)].to_numpy()
        lists, weights = [co.recommend([seed], 200)], [1.0]
        if name_hint:
            lists.append(tm.recommend_titles([name_hint], 200, exclude=[seed]))
            weights.append(info["title_weight"])
        show(fuse(lists, weights, info["tracks"], k)[0], names, pop, scenes)
    else:
        st.info("Pick at least one song to get recommendations.")

if scenes is not None:
    with tabs[2]:
        st.write("Louvain communities on the track co-occurrence graph, ranked by size. "
                 "No genre labels were used: the scenes come purely from which songs share playlists.")
        summary = scenes.groupby("scene")["size"].first().sort_values(ascending=False)
        pick = st.selectbox("Scene", summary.index,
                            format_func=lambda s: f"{s} ({summary[s]:,} tracks)")
        top = scenes[scenes["scene"] == pick].nlargest(k, "pagerank")
        df = names.loc[top.index, ["artist_name", "track_name"]].reset_index(drop=True)
        df.columns = ["Artist", "Track"]
        df.index += 1
        df["PageRank"] = top["pagerank"].to_numpy()
        st.dataframe(df, use_container_width=True)
