"""Cold start: build a playlist from its title alone ("rainy sunday 🌧️" -> tracks).

1. Embed every training playlist title with a sentence transformer
   (falls back to TF-IDF character n-grams if sentence-transformers is missing).
2. For a new title, find the `top_m` most similar titles (cosine similarity).
3. Pool the tracks of those playlists, weighted by similarity.
"""
from __future__ import annotations

import re

import numpy as np
import scipy.sparse as sps

from mixtape.data.dataset import Dataset
from mixtape.models.base import topk_excluding


def normalize(title: str) -> str:
    t = title.lower().strip()
    t = re.sub(r"[^\w\s']", " ", t)       # drop emoji / punctuation
    return re.sub(r"\s+", " ", t).strip()


class _TfidfEncoder:
    name = "tfidf-char"

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)

    def fit(self, texts):
        self.vec.fit(texts)
        return self

    def encode(self, texts) -> np.ndarray:
        m = self.vec.transform(texts)
        return m.toarray().astype(np.float32)  # rows are already L2-normalised


class _SentenceEncoder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.name = model_name

    def fit(self, texts):
        return self

    def encode(self, texts) -> np.ndarray:
        return self.model.encode(list(texts), batch_size=256, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)


class _HybridEncoder:
    """Concatenate L2-normalised TF-IDF and transformer vectors, each scaled by
    sqrt(w), so the dot product is w_tfidf*cos_tfidf + w_st*cos_st: exact-wording
    matches from TF-IDF plus meaning-level matches from the transformer."""

    def __init__(self, model_name: str, w_tfidf: float = 0.5):
        self.tfidf, self.st = _TfidfEncoder(), _SentenceEncoder(model_name)
        self.a, self.b = np.sqrt(w_tfidf), np.sqrt(1 - w_tfidf)
        self.name = f"hybrid(tfidf+{model_name.split('/')[-1]})"

    def fit(self, texts):
        self.tfidf.fit(texts)
        return self

    def encode(self, texts) -> np.ndarray:
        return np.hstack([self.a * self.tfidf.encode(texts), self.b * self.st.encode(texts)])


def make_encoder(model_name: str | None, prefer_transformer: bool = True):
    """model_name: a sentence-transformers id, 'hybrid:<id>' to blend it with TF-IDF,
    or None for TF-IDF only. Falls back to TF-IDF if sentence-transformers is missing."""
    if prefer_transformer and model_name:
        try:
            if model_name.startswith("hybrid:"):
                return _HybridEncoder(model_name.split(":", 1)[1])
            return _SentenceEncoder(model_name)
        except ImportError:
            print("  sentence-transformers not installed -> TF-IDF fallback "
                  "(pip install -r requirements-nlp.txt)")
    return _TfidfEncoder()


class TitleRecommender:
    name = "title"

    def __init__(self, model_name: str | None = None, top_m: int = 50, prefer_transformer=True):
        self.model_name, self.top_m, self.prefer_transformer = model_name, top_m, prefer_transformer

    def fit(self, data: Dataset) -> "TitleRecommender":
        tr = data.train_only
        pids = tr["pid"].unique()
        norm = np.array([normalize(data.titles.get(p, "")) for p in pids])
        # one row per unique normalised title -> track counts (titles x items)
        self.uniq, inv = np.unique(norm, return_inverse=True)
        pid_to_title = dict(zip(pids, inv))
        rows = tr["pid"].map(pid_to_title).to_numpy()
        self.T = sps.csr_matrix(
            (np.ones(len(rows), np.float32), (rows, tr["item"].to_numpy())),
            shape=(len(self.uniq), data.n_items + 1),
        )
        # normalise each title's track distribution so big groups don't dominate
        sums = np.asarray(self.T.sum(1)).ravel()
        self.T = sps.diags(1.0 / np.maximum(sums, 1)) @ self.T
        self.encoder = make_encoder(self.model_name, self.prefer_transformer).fit(self.uniq)
        self.E = self.encoder.encode(self.uniq)
        return self

    def recommend_titles(self, titles: list[str], k: int,
                         exclude: list[np.ndarray] | None = None) -> np.ndarray:
        q = self.encoder.encode([normalize(t) for t in titles])
        sim = q @ self.E.T                                     # (n, n_titles)
        m = min(self.top_m, sim.shape[1])
        idx = np.argpartition(-sim, m - 1, axis=1)[:, :m]
        w = np.take_along_axis(sim, idx, 1).clip(min=0)
        W = sps.csr_matrix((w.ravel(), (np.repeat(np.arange(len(titles)), m), idx.ravel())),
                           shape=sim.shape)
        scores = (W @ self.T).toarray()
        exclude = exclude or [np.array([], dtype=np.int64)] * len(titles)
        return topk_excluding(scores, exclude, k)

    def similar_titles(self, title: str, n: int = 5) -> list[tuple[str, float]]:
        sim = (self.encoder.encode([normalize(title)]) @ self.E.T).ravel()
        top = np.argsort(-sim)[:n]
        return [(str(self.uniq[i]), round(float(sim[i]), 3)) for i in top]
