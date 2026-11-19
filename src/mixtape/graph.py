"""Track graph analysis with cuGraph (GPU) or networkx (CPU).

Nodes = tracks, edge weight = number of playlists two tracks share.
  * PageRank   -> "central" tracks that glue many playlists together
  * Louvain    -> communities of tracks = emergent scenes / moods
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from mixtape.backend import Backend
from mixtape.data.dataset import Dataset
from mixtape.models.cooccur import build_matrix


def build_edges(be: Backend, data: Dataset, top_n: int, min_w: int) -> pd.DataFrame:
    tr = data.train_only
    pop = np.bincount(tr["item"].to_numpy(), minlength=data.n_items + 1)
    pop[0] = 0
    top = np.argsort(-pop)[:top_n]
    sub = tr[tr["item"].isin(top)]
    X = build_matrix(be, sub["pid"].to_numpy(), sub["item"].to_numpy(), data.n_items)
    top_sorted = np.sort(top)
    Xt = X[:, be.xp.asarray(top_sorted)]
    if be.is_gpu:
        r, c, w = _upper_edges_gpu(Xt, min_w)
    else:
        C = (Xt.T @ Xt).tocoo()
        r, c, w = C.row, C.col, C.data
        keep = (r < c) & (w >= min_w)
        r, c, w = r[keep], c[keep], w[keep]
    return pd.DataFrame({"src": top_sorted[r], "dst": top_sorted[c],
                         "weight": w.astype(np.float32)})


def _upper_edges_gpu(Xt, min_w: int, chunk: int = 4096):
    """Xt^T Xt as a dense matrix built from playlist chunks with cuBLAS.

    cuSPARSE SpGEMM needs a scratch buffer proportional to the number of
    intermediate products (sum of len^2 over playlists): ~21 GB on the full MPD,
    far beyond a 4 GB card. A dense top_n x top_n accumulator (1.6 GB at 20k)
    plus one dense chunk keeps memory bounded and is fast on tensor-free GPUs too.
    """
    import cupy as cp
    n = Xt.shape[1]
    C = cp.zeros((n, n), dtype=cp.float32)
    for s in range(0, Xt.shape[0], chunk):
        Xb = Xt[s:s + chunk].toarray()
        C += Xb.T @ Xb
        del Xb
    C = cp.triu(C, k=1)
    r, c = cp.nonzero(C >= min_w)
    w = C[r, c]
    del C
    return cp.asnumpy(r), cp.asnumpy(c), cp.asnumpy(w)


def analyze(be: Backend, data: Dataset, top_n=5000, min_w=3, seed=42) -> tuple[pd.DataFrame, dict]:
    t0 = time.perf_counter()
    edges = build_edges(be, data, top_n, min_w)
    t_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    if be.is_gpu:
        import cudf
        import cugraph
        G = cugraph.Graph()
        G.from_cudf_edgelist(cudf.from_pandas(edges), source="src", destination="dst",
                             edge_attr="weight", store_transposed=True)
        parts, modularity = cugraph.louvain(G)
        pr = cugraph.pagerank(G)
        nodes = parts.merge(pr, on="vertex").to_pandas() \
            .rename(columns={"vertex": "item", "partition": "community"})
    else:
        import networkx as nx
        G = nx.Graph()
        G.add_weighted_edges_from(edges.itertuples(index=False, name=None))
        comms = nx.community.louvain_communities(G, weight="weight", seed=seed)
        modularity = nx.community.modularity(G, comms, weight="weight")
        pr = nx.pagerank(G, weight="weight")
        nodes = pd.DataFrame([(n, ci, pr[n]) for ci, cset in enumerate(comms) for n in cset],
                             columns=["item", "community", "pagerank"])
    t_algo = time.perf_counter() - t0

    summary = {
        "backend": be.name,
        "nodes": int(len(nodes)),
        "edges": int(len(edges)),
        "communities": int(nodes["community"].nunique()),
        "modularity": round(float(modularity), 4),
        "build_s": round(t_build, 3),
        "algorithms_s": round(t_algo, 3),
    }
    return nodes, summary
