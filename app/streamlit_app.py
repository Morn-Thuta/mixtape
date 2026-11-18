"""mixtape demo: build a playlist from a name, or continue one from a few songs.

Run (from the repo root, after `python -m mixtape prep`):
    pip install -r requirements-app.txt
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from mixtape.backend import get_backend
from mixtape.config import load_config
from mixtape.data.dataset import load_dataset
from mixtape.models.cooccur import CoOccurrence
from mixtape.models.hybrid import fuse
from mixtape.models.title import TitleRecommender

st.set_page_config(page_title="mixtape", page_icon="🎧", layout="centered")


@st.cache_resource(show_spinner="Loading data and fitting models…")
def load(tfidf_only: bool):
    cfg = load_config()
    data = load_dataset(cfg["paths"]["processed"])
    tracks = pd.read_parquet(cfg["paths"]["interim"] / "tracks.parquet")
    names = data.vocab.merge(tracks, on="tid").set_index("item")
    names["label"] = names["artist_name"] + " - " + names["track_name"]
    co = CoOccurrence(get_backend(), batch_size=cfg["cooccur"]["batch_size"]).fit(data)
    tm = TitleRecommender(cfg["title"]["model"], cfg["title"]["top_m"],
                          prefer_transformer=not tfidf_only).fit(data)
    pop = np.bincount(data.train["item"].to_numpy(), minlength=data.n_items + 1)
    return cfg, data, names, co, tm, pop


def show(items: np.ndarray, names: pd.DataFrame, pop: np.ndarray) -> None:
    df = names.loc[items, ["artist_name", "track_name"]].reset_index(drop=True)
    df.index += 1
    df["playlists containing it"] = pop[items]
    st.dataframe(df, use_container_width=True)


st.title("🎧 mixtape")
st.caption("Playlist continuation with RAPIDS (cuDF / cuGraph) and transformers.")
tfidf_only = st.sidebar.toggle("TF-IDF only (no transformer download)", value=False)
k = st.sidebar.slider("Songs to recommend", 5, 50, 15)
cfg, data, names, co, tm, pop = load(tfidf_only)
st.sidebar.write(f"**Backend:** {get_backend().name}  \n**Title encoder:** {tm.encoder.name}  \n"
                 f"**Tracks:** {data.n_items:,}  \n**Playlists:** {data.train['pid'].nunique():,}")

tab_title, tab_seeds = st.tabs(["✍️ From a name", "🎵 From songs"])

with tab_title:
    title = st.text_input("Name your playlist", "late night drive")
    if title:
        st.write("Most similar playlist names in the data:")
        st.write(", ".join(f"`{t}` ({s:.2f})" for t, s in tm.similar_titles(title, 5)))
        show(tm.recommend_titles([title], k)[0], names, pop)

with tab_seeds:
    popular = names.loc[np.argsort(-pop)[:3000]]
    popular = popular[popular.index > 0]
    picked = st.multiselect("Pick a few songs", popular["label"].tolist(), max_selections=10)
    name_hint = st.text_input("Optional: playlist name (blended in with rank fusion)", "")
    if picked:
        seed = popular.index[popular["label"].isin(picked)].to_numpy()
        lists = [co.recommend([seed], 200)]
        weights = [1.0]
        if name_hint:
            lists.append(tm.recommend_titles([name_hint], 200, exclude=[seed]))
            weights.append(cfg["hybrid"].get("title_weight", 1.0))
        show(fuse(lists, weights, data.n_items, k)[0], names, pop)
    else:
        st.info("Pick at least one song to get recommendations.")
