"""mixtape demo: build a playlist from a name, continue one from a few songs,
or browse the music scenes cuGraph found.

Run from the repo root:
    pip install -r requirements-app.txt -r requirements-nlp.txt
    streamlit run app/streamlit_app.py

Uses the real Spotify MPD (configs/mpd.yaml) when it has been prepared, else the
synthetic data. Override with MIXTAPE_CONFIG=configs/<name>.yaml. For the MPD, run
`python scripts/build_demo_cache.py` once on the GPU so the app doesn't spend
~15 min encoding 100k playlist titles with MPNet on the CPU.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import streamlit as st

from mixtape.backend import get_backend
from mixtape.config import ROOT, load_config
from mixtape.data.dataset import load_dataset
from mixtape.models.cooccur import CoOccurrence
from mixtape.models.hybrid import fuse
from mixtape.models.title import TitleRecommender

st.set_page_config(page_title="mixtape", page_icon="🎧", layout="wide")
N_PICKABLE = 20_000  # most popular tracks offered in the song picker


def default_config() -> str:
    if os.environ.get("MIXTAPE_CONFIG"):
        return os.environ["MIXTAPE_CONFIG"]
    mpd = ROOT / "data" / "mpd" / "processed" / "train.parquet"
    return str(ROOT / "configs" / ("mpd.yaml" if mpd.exists() else "default.yaml"))


@st.cache_resource(show_spinner="Loading playlists and fitting models (≈3–4 min on a CPU for the full MPD, once)…")
def load(config: str, tfidf_only: bool):
    cfg = load_config(config)
    data = load_dataset(cfg["paths"]["processed"])
    tracks = pd.read_parquet(cfg["paths"]["interim"] / "tracks.parquet",
                             columns=["tid", "track_name", "artist_name"])
    names = data.vocab.merge(tracks, on="tid").set_index("item").sort_index()
    names["label"] = names["artist_name"] + " - " + names["track_name"]
    co = CoOccurrence(get_backend(), batch_size=cfg["cooccur"]["batch_size"]).fit(data)
    tm = TitleRecommender(cfg["title"]["model"], cfg["title"]["top_m"],
                          prefer_transformer=not tfidf_only).fit(data, cache_dir=cfg["paths"]["processed"])
    pop = np.bincount(data.train["item"].to_numpy(), minlength=data.n_items + 1)
    stats = {"tracks": data.n_items, "playlists": int(data.train["pid"].nunique())}
    return cfg, stats, names, co, tm, pop


@st.cache_resource(show_spinner=False)
def load_scenes(config: str, _cfg, _names):
    """Louvain communities from `python -m mixtape graph`, named after their three most
    central artists (summed PageRank of their tracks): one artist alone can mislabel a
    big mixed community."""
    path = _cfg["paths"]["reports"] / "graph_nodes.parquet"
    if not path.exists():
        return None
    nodes = pd.read_parquet(path, columns=["item", "community", "pagerank"])
    nodes["artist"] = _names["artist_name"].reindex(nodes["item"]).to_numpy()
    size = nodes["community"].value_counts()
    by_artist = nodes.groupby(["community", "artist"])["pagerank"].sum().reset_index() \
        .sort_values("pagerank", ascending=False)
    label = {c: " · ".join(g["artist"].head(3)) for c, g in by_artist.groupby("community", sort=False)}
    nodes["scene"] = nodes["community"].map(label)
    nodes["size"] = nodes["community"].map(size)
    return nodes.set_index("item")


def show(items: np.ndarray, names: pd.DataFrame, pop: np.ndarray, scenes) -> None:
    df = names.loc[items, ["artist_name", "track_name"]].reset_index(drop=True)
    df.columns = ["Artist", "Track"]
    df.index += 1
    df["Playlists containing it"] = pop[items]
    if scenes is not None:
        df["Scene (cuGraph)"] = scenes["scene"].reindex(items).fillna("").to_numpy()
    st.dataframe(df, use_container_width=True)


st.title("🎧 mixtape")
st.caption("Playlist continuation on the Spotify Million Playlist Dataset · RAPIDS (cuDF, cuGraph) + transformers")
config = default_config()
tfidf_only = st.sidebar.toggle("TF-IDF only (no transformer)", value=False)
k = st.sidebar.slider("Songs to recommend", 5, 50, 20)
cfg, stats, names, co, tm, pop = load(config, tfidf_only)
scenes = load_scenes(config, cfg, names)
st.sidebar.markdown(
    f"**Data:** `{os.path.basename(config)}`  \n**Tracks:** {stats['tracks']:,}  \n"
    f"**Playlists:** {stats['playlists']:,}  \n**Backend:** {get_backend().name}  \n"
    f"**Title encoder:** {tm.encoder.name}")
st.sidebar.caption("Live demo uses co-occurrence + the title model, fused by reciprocal rank. "
                   "SASRec is evaluated offline (reports/mpd/results.json).")

tabs = st.tabs(["✍️ From a name", "🎵 From songs"] + (["🗺️ Scenes"] if scenes is not None else []))

with tabs[0]:
    st.write("Type any playlist name, even one nobody used: the transformer matches meaning, "
             "TF-IDF matches spelling.")
    title = st.text_input("Name your playlist", "late night drive")
    if title:
        st.write("Closest playlist names among 1M real playlists: "
                 + ", ".join(f"`{t}` ({s:.2f})" for t, s in tm.similar_titles(title, 6)))
        show(tm.recommend_titles([title], k)[0], names, pop, scenes)

with tabs[1]:
    order = np.argsort(-pop)
    pickable = names.loc[order[order > 0][:N_PICKABLE]]
    picked = st.multiselect(f"Pick a few songs (search the {N_PICKABLE:,} most popular)",
                            pickable["label"].tolist(), max_selections=10)
    name_hint = st.text_input("Optional: playlist name, blended in with rank fusion", "")
    if picked:
        seed = pickable.index[pickable["label"].isin(picked)].to_numpy()
        lists, weights = [co.recommend([seed], 200)], [1.0]
        if name_hint:
            lists.append(tm.recommend_titles([name_hint], 200, exclude=[seed]))
            weights.append(cfg["hybrid"].get("title_weight", 0.5))
        show(fuse(lists, weights, stats["tracks"], k)[0], names, pop, scenes)
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
