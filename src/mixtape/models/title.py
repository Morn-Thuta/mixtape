"""Cold start: build a playlist from its title alone ("rainy sunday 🌧️" -> tracks).

1. Embed every training playlist title with a sentence transformer
   (falls back to TF-IDF character n-grams if sentence-transformers is missing).
2. For a new title, find the `top_m` most similar titles (cosine similarity).
3. Pool the tracks of those playlists, weighted by similarity.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import scipy.sparse as sps

from mixtape.data.dataset import Dataset
from mixtape.models.base import topk_sparse_rows


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

    def encode(self, texts):
        # stays sparse: a dense TF-IDF matrix of ~100k titles x ~50k char n-grams
        # would be ~20 GB. Rows are already L2-normalised.
        return self.vec.transform(texts).astype(np.float32)

    @staticmethod
    def sim(q, E) -> np.ndarray:
        return (q @ E.T).toarray()


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

    @staticmethod
    def sim(q, E) -> np.ndarray:
        return q @ E.T


class _HybridEncoder:
    """Similarity = w_tfidf * cos_tfidf + (1 - w_tfidf) * cos_transformer:
    exact-wording matches from TF-IDF plus meaning-level matches from the transformer."""

    def __init__(self, model_name: str, w_tfidf: float = 0.5):
        self.tfidf, self.st = _TfidfEncoder(), _SentenceEncoder(model_name)
        self.w = w_tfidf
        self.name = f"hybrid(tfidf+{model_name.split('/')[-1]})"

    def fit(self, texts):
        self.tfidf.fit(texts)
        return self

    def encode(self, texts):
        return self.tfidf.encode(texts), self.st.encode(texts)

    def sim(self, q, E) -> np.ndarray:
        return self.w * self.tfidf.sim(q[0], E[0]) + (1 - self.w) * self.st.sim(q[1], E[1])


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


def _encode_cached(encoder, texts: np.ndarray, cache_dir: Path | None):
    """encoder.encode(texts), loading/saving the transformer part from cache_dir.
    The cache key covers the model name and the exact list of titles."""
    st = encoder.st if isinstance(encoder, _HybridEncoder) else encoder
    if cache_dir is None or not isinstance(st, _SentenceEncoder):
        return encoder.encode(texts)
    h = hashlib.sha1(st.name.encode())
    h.update("\n".join(texts.tolist()).encode())
    path = Path(cache_dir) / f"title_emb_{h.hexdigest()[:16]}.npy"
    if path.exists():
        dense = np.load(path).astype(np.float32)
    else:
        dense = st.encode(texts)
        np.save(path, dense.astype(np.float16))
    if isinstance(encoder, _HybridEncoder):
        return encoder.tfidf.encode(texts), dense
    return dense


class TitleRecommender:
    name = "title"

    def __init__(self, model_name: str | None = None, top_m: int = 50, prefer_transformer=True):
        self.model_name, self.top_m, self.prefer_transformer = model_name, top_m, prefer_transformer

    def fit(self, data: Dataset, cache_dir: Path | None = None) -> "TitleRecommender":
        """cache_dir: where to keep the transformer embeddings of the training titles.
        Encoding ~100k titles with MPNet takes seconds on a GPU but ~15 min on a CPU,
        so the demo app reuses embeddings computed once on the GPU."""
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
        self.E = _encode_cached(self.encoder, self.uniq, cache_dir)
        return self

    def recommend_titles(self, titles: list[str], k: int,
                         exclude: list[np.ndarray] | None = None,
                         batch_size: int = 512) -> np.ndarray:
        """Batched and sparse end to end: memory is O(batch x n_titles), never
        O(queries x tracks) (that would be ~14 GB for 10k queries on the MPD)."""
        out = []
        for b in range(0, len(titles), batch_size):
            chunk = titles[b:b + batch_size]
            sim = self.encoder.sim(self.encoder.encode([normalize(t) for t in chunk]), self.E)
            m = min(self.top_m, sim.shape[1])
            idx = np.argpartition(-sim, m - 1, axis=1)[:, :m]
            w = np.take_along_axis(sim, idx, 1).clip(min=0)
            W = sps.csr_matrix((w.ravel(), (np.repeat(np.arange(len(chunk)), m), idx.ravel())),
                               shape=sim.shape)
            ex = exclude[b:b + batch_size] if exclude is not None else None
            out.append(topk_sparse_rows(W @ self.T, k, ex))
        return np.vstack(out) if out else np.zeros((0, k), dtype=np.int64)

    def similar_titles(self, title: str, n: int = 5) -> list[tuple[str, float]]:
        sim = self.encoder.sim(self.encoder.encode([normalize(title)]), self.E).ravel()
        top = np.argsort(-sim)[:n]
        return [(str(self.uniq[i]), round(float(sim[i]), 3)) for i in top]
