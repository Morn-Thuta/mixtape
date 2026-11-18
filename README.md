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

**Demo app**: build a playlist from a name, or continue one from songs you pick:

```powershell
pip install -r requirements-app.txt
streamlit run app/streamlit_app.py
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

## CPU vs GPU (RTX 3050 Laptop, 4 GB, WSL2, RAPIDS 26.10)

Same code, same data; best of 3 passes per backend. Full tables: [reports/benchmark.md](reports/benchmark.md) and [reports/large/benchmark.md](reports/large/benchmark.md).

| Stage | 0.7M rows: CPU → GPU | speedup | 7M rows: CPU → GPU | speedup |
|---|---|---:|---|---:|
| Data prep (cuDF) | 0.73 s → 0.36 s | 2.0× | 9.38 s → 1.10 s | **8.5×** |
| Co-occurrence scoring (CuPy sparse) | 0.32 s → 0.18 s | 1.8× | 32.7 s → 8.5 s | **3.9×** |
| Louvain + PageRank (cuGraph) | 8.69 s → 0.30 s | 28.7× | 17.8 s → 0.37 s | **47.7×** |

**The speedup grows with data size.** At 0.7M rows the GPU's fixed costs (kernel launches, host↔device copies) eat most of the gain, and a few tiny steps are even slower on GPU. At 7M rows the per-row work dominates and cuDF's group-bys and joins pull ahead (the track-filter step alone is 22.6× faster). The full MPD is 66M rows, about 10× larger again.

## Model results (synthetic data, 20k playlists, k=50, prepared and trained on GPU)

| Task | Model | recall@50 | ndcg@50 |
|---|---|---:|---:|
| Continuation (5 seed tracks) | popularity | 0.061 | 0.058 |
| | co-occurrence | **0.405** | **0.409** |
| | SASRec, next-item, 30 epochs | 0.246 | 0.203 |
| | SASRec, window-5 targets + early stopping | 0.329 | 0.306 |
| | hybrid (co-occurrence + 0.5 × SASRec) | 0.402 | 0.405 |
| Cold start (title only) | popularity | 0.070 | 0.073 |
| | title encoder (TF-IDF) | **0.374** | **0.428** |

Louvain recovered all 24 planted moods (modularity 0.87) on both CPU and GPU.

### Adding the playlist title to the hybrid

`scripts/hybrid_title_experiment.py`: co-occurrence + SASRec + title model, fused by reciprocal rank (title weight 1.0).

| Seed tracks given | Best single model (recall / ndcg @50) | Full hybrid (recall / ndcg @50) |
|---|---|---|
| 1 | 0.359 / 0.384 (co-occurrence) | **0.389 / 0.425** (+8% / +11%) |
| 5 | 0.405 / 0.409 (co-occurrence) | 0.404 / **0.419** |

The name helps most when a playlist has only one or two songs, which is exactly when the track-based models know least.

### Title encoders: spelling vs meaning (cold start)

`scripts/compare_title_encoders.py`. *Seen titles* are test playlists whose names also occur in training. *Unseen phrasings* are 24 hand-written queries that share no words with any training title (e.g. "marathon training" → running); the score is the share of the top 50 tracks that come from the right mood.

| Encoder | Seen titles recall@50 | Unseen phrasings precision@50 |
|---|---:|---:|
| TF-IDF character n-grams | **0.374** | 0.220 |
| all-MiniLM-L6-v2 | 0.318 | 0.471 |
| all-mpnet-base-v2 | 0.313 | **0.551** |
| TF-IDF + MiniLM | 0.358 | 0.490 |
| **TF-IDF + MPNet (default)** | 0.355 | 0.532 |

TF-IDF matches spelling, so it wins when the exact title was seen before but maps "late night drive" to "date night". Transformers match meaning and are 2.5× better on new phrasings. Concatenating both embeddings keeps most of each strength.

**Reading these numbers.** The synthetic generator builds playlists mostly by sampling from a mood bucket, which is exactly what co-occurrence captures. The only order signal is a weak drift between sub-scenes, so SASRec and the hybrid can't beat co-occurrence here. Real playlists have stronger order effects (album runs, energy arcs), so the real comparison needs MPD data. Training SASRec to predict *any of the next 5 tracks*, instead of only the next one, plus early stopping, raised its recall@50 from 0.25 to 0.33.

**Reproducibility.** The train/test split uses a deterministic hash of the playlist id, so pandas and cuDF hold out exactly the same playlists (`tests/test_gpu_parity.py` checks that both backends agree).

## Roadmap

- [ ] Run on real MPD slices (start with 50, then all 1,000)
- [x] Fill the GPU column of the benchmark (WSL2, RTX 3050), at 0.7M and 7M rows
- [ ] Benchmark at full MPD scale (66M rows) on Colab T4 (16 GB)
- [x] Sentence-transformer title encoder vs TF-IDF (+ hybrid encoder, unseen-phrasing test)
- [x] SASRec future-window targets (`sasrec.target_window`) and longer training
- [x] Hybrid model: reciprocal rank fusion of co-occurrence + SASRec
- [x] Validation split + early stopping for SASRec (`sasrec.val_size`, `patience`)
- [x] Add the title model to the hybrid for playlists with few seed tracks
- [ ] Use cuGraph communities/PageRank as features
- [x] Small web demo (Streamlit): type a title or pick seed songs → playlist
