# mixtape 🎧

**GPU-accelerated playlist continuation**: given the first few songs of a playlist (or just its title), predict what comes next.

Built to combine two NVIDIA DLI skill sets:

| Part | What it does | Skill |
|---|---|---|
| `data/prep.py` | Clean, index and split 66M playlist rows | **RAPIDS cuDF** (Accelerated Data Science) |
| `models/cooccur.py` | "Songs that appear together" via sparse matrix products | **CuPy / cuDF** (Accelerated Data Science) |
| `graph.py` | Track graph → Louvain communities + PageRank | **RAPIDS cuGraph** (Accelerated Data Science) |
| `models/sasrec.py` | Transformer reads a playlist like a sentence, predicts the next song | **Self-attention** (Transformer NLP) |
| `models/title.py` | "rainy sunday 🌧️" → playlist, via sentence embeddings | **Sentence transformers** (Transformer NLP) |
| `bench.py` | Same pipeline on CPU vs GPU → speedup table | both |

```
MPD json ──ingest──► parquet ──prep (pandas|cuDF)──► train / test split
                                         │
          ┌──────────────┬───────────────┼────────────────┐
     popularity      co-occurrence     SASRec          title encoder
     (baseline)      (CuPy sparse)     (PyTorch)       (cold start)
                          │
                    cuGraph communities + PageRank
```

## Quick start (Windows / any CPU, ~1 min)

```powershell
pip install -r requirements.txt
pip install -e .
python -m mixtape all          # synthetic data -> ingest -> prep -> train -> graph
python -m mixtape demo --title "rainy sunday"
python -m pytest
```

The code picks its backend automatically: **cuDF/CuPy/cuGraph when RAPIDS is installed, pandas/SciPy/networkx otherwise.** Force it with `MIXTAPE_BACKEND=cpu|gpu`. PyTorch (SASRec) uses CUDA whenever it's available, Windows included.

## GPU (RAPIDS) - Colab or WSL2

RAPIDS doesn't run natively on Windows. Two options:

- **Colab (easiest):** open `notebooks/colab_quickstart.ipynb` and pick a T4 GPU runtime.
- **WSL2:** `wsl --install -d Ubuntu`, then inside Ubuntu:
  `pip install -r requirements.txt -r requirements-gpu.txt --extra-index-url=https://pypi.nvidia.com`

Then `python -m mixtape bench` fills both columns of `reports/benchmark.md`.

## Real data: Spotify Million Playlist Dataset

1. Register and download the MPD from the AIcrowd *Spotify Million Playlist Dataset Challenge* page.
2. Copy the `mpd.slice.*.json` files into `data/raw/` (and remove the synthetic ones).
3. Start small: `python -m mixtape ingest --max-slices 50` (50k playlists), then `prep`, `train`.
4. For the full 1M playlists, set `sasrec.n_neg: 128` (sampled softmax) so training fits in GPU memory.

The synthetic generator (`data/synthetic.py`) writes the **exact same JSON format**, so nothing else changes.

## Results so far (synthetic data, 20k playlists, k=50)

| Task | Model | recall@50 | ndcg@50 |
|---|---|---:|---:|
| Continuation (5 seed tracks) | popularity | 0.062 | 0.058 |
| | co-occurrence | **0.398** | **0.402** |
| | SASRec, next-item, 30 epochs | 0.246 | 0.203 |
| | SASRec, window-5 targets, 60 epochs | 0.310 | 0.284 |
| | SASRec, window-5 + early stopping (best epoch 65) | 0.324 | 0.302 |
| | hybrid (co-occurrence + 0.25 × SASRec) | 0.398 | 0.400 |
| Cold start (title only) | popularity | 0.069 | 0.071 |
| | title encoder (TF-IDF) | **0.370** | **0.420** |

Louvain recovered all 24 planted moods (modularity 0.87).

**Reading these numbers.** The synthetic generator builds playlists mostly by sampling from a mood bucket, which is exactly what co-occurrence captures. The only order signal is a weak drift between sub-scenes, so SASRec and the hybrid can't beat co-occurrence here. Real playlists have stronger order effects (album runs, energy arcs), so the real comparison needs MPD data. Training SASRec to predict *any of the next 5 tracks*, instead of only the next one, raised its recall@50 from 0.25 to 0.31.

## Roadmap

- [ ] Run on real MPD slices (start with 50, then all 1,000)
- [ ] Fill the GPU column of the benchmark on Colab
- [ ] Sentence-transformer title encoder vs TF-IDF
- [x] SASRec future-window targets (`sasrec.target_window`) and longer training
- [x] Hybrid model: reciprocal rank fusion of co-occurrence + SASRec
- [x] Validation split + early stopping for SASRec (`sasrec.val_size`, `patience`)
- [ ] Add the title model to the hybrid for playlists with few seed tracks
- [ ] Use cuGraph communities/PageRank as features
- [ ] Small web demo (Streamlit): type a title or pick seed songs → playlist
