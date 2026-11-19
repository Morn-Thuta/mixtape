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

**Demo app** (Streamlit, runs on a plain CPU). It has three tabs:
- **From a name:** type any playlist name and get tracks.
- **From songs:** pick a few of the 20k most popular tracks and get a continuation (co-occurrence, optionally fused with a name).
- **Scenes:** browse the cuGraph communities.

```powershell
pip install -r requirements-app.txt -r requirements-nlp.txt
streamlit run app/streamlit_app.py
```

The app uses the full MPD automatically once `data/mpd/processed` exists, otherwise the synthetic data; override with `MIXTAPE_CONFIG=configs/<name>.yaml`. For the MPD, first run `python scripts/build_demo_cache.py` on the GPU: it embeds all 16,976 unique playlist names with MPNet in 36 s, which would take ~15 min on a CPU. The app then loads in ~3–4 min on a laptop CPU (~5 GB RAM). Examples on the real data:
- "late night drive" → Kavinsky *Nightcall* and Electric Youth *A Real Hero*, both from the *Drive* soundtrack.
- Bon Iver + Fleet Foxes as seeds → *Skinny Love*, *Mykonos*, Ben Howard.

SASRec isn't in the live demo: it's evaluated offline, since its 10.8 h training run isn't checkpointed.

The code picks its backend automatically: **cuDF/CuPy/cuGraph when RAPIDS is installed, pandas/SciPy/networkx otherwise.** Force it with `MIXTAPE_BACKEND=cpu|gpu`. PyTorch (SASRec) uses CUDA whenever it's available, Windows included.

## GPU (RAPIDS) - Colab or WSL2

RAPIDS doesn't run natively on Windows. Two options:

- **Colab (easiest):** open `notebooks/colab_quickstart.ipynb` and pick a T4 GPU runtime.
- **WSL2:** `wsl --install -d Ubuntu`, then inside Ubuntu:
  `pip install -r requirements.txt -r requirements-gpu.txt --extra-index-url=https://pypi.nvidia.com`

Then `python -m mixtape bench` fills both columns of `reports/benchmark.md`.

## Real data: Spotify Million Playlist Dataset

1. Register and download the MPD from the AIcrowd *Spotify Million Playlist Dataset Challenge* page.
2. `python scripts/extract_mpd.py <zip>` extracts and md5-checks the slices into `data/mpd/raw/`.
3. Start small: `bash scripts/run_mpd50.sh` (50 slices, `configs/mpd50.yaml`).
4. Full 1M playlists: `CFG=configs/mpd.yaml bash scripts/full_scale_run.sh ingest prep train-all graph` (logs time, peak RAM and GPU memory per step). `sasrec.n_neg: 128` (sampled softmax) keeps training inside 4 GB.

The synthetic generator (`data/synthetic.py`) writes the **exact same JSON format**, so nothing else changes.

## CPU vs GPU (RTX 3050 Laptop, 4 GB, WSL2, RAPIDS 26.10)

Same code, same data; best of 3 passes per backend. Full tables: [reports/benchmark.md](reports/benchmark.md) and [reports/large/benchmark.md](reports/large/benchmark.md).

| Stage | 0.7M rows: CPU → GPU | speedup | 7M rows: CPU → GPU | speedup |
|---|---|---:|---|---:|
| Data prep (cuDF) | 0.73 s → 0.36 s | 2.0× | 9.38 s → 1.10 s | **8.5×** |
| Co-occurrence scoring (CuPy sparse) | 0.32 s → 0.18 s | 1.8× | 32.7 s → 8.5 s | **3.9×** |
| Louvain + PageRank (cuGraph) | 8.69 s → 0.30 s | 28.7× | 17.8 s → 0.37 s | **47.7×** |

**The speedup grows with data size.** At 0.7M rows the GPU's fixed costs (kernel launches, host↔device copies) eat most of the gain, and a few tiny steps are even slower on GPU. At 7M rows the per-row work dominates and cuDF's group-bys and joins pull ahead (the track-filter step alone is 22.6× faster). The full MPD is 66M rows, about 10× larger again.

## Results on the real Spotify Million Playlist Dataset

All 1,000 slices: 66.3M rows → 63.4M after filtering, 857,746 tracks, 957,253 playlists, 9,549 held-out test playlists (first 5 tracks in, predict the rest; k=50). Everything ran on one 4 GB RTX 3050 Laptop GPU in WSL2. Raw numbers: [reports/mpd/results.json](reports/mpd/results.json), timings: [reports/mpd/run_log.txt](reports/mpd/run_log.txt).

| Task | Model | recall@50 | ndcg@50 | hit@50 |
|---|---|---:|---:|---:|
| Continuation (5 seed tracks) | popularity | 0.024 | 0.034 | 0.387 |
| | co-occurrence | 0.158 | 0.208 | 0.862 |
| | SASRec (transformer) | 0.125 | 0.166 | 0.880 |
| | **hybrid (co-occurrence + 0.5 × SASRec + 0.5 × title)** | **0.169** | **0.225** | **0.895** |
| Cold start (title only) | popularity | 0.025 | 0.037 | 0.414 |
| | **title encoder (TF-IDF + MPNet)** | **0.076** | **0.107** | **0.646** |

- **The hybrid beats the best single model by 7% recall and 8% ndcg.** Its weights were tuned on validation playlists carved out of training (`scripts/tune_hybrid.py` on the 50-slice subset); the test set was never used for tuning.
- **SASRec alone trails co-occurrence on recall, but it puts at least one correct track in the top 50 for more playlists (hit@50 0.880 vs 0.862).** That's because the two models get different playlists right, and that difference is what the fusion exploits.
- **From the name alone, the title model finds 3× more of the playlist than popularity.**

| Stage (full MPD, GPU) | Time | Peak GPU memory |
|---|---:|---:|
| Ingest (1,000 JSON slices → parquet) | 9 min | 1.4 GB |
| Prep in cuDF (filter, index, split 66M rows) | 15 s | 3.9 GB |
| Co-occurrence fit | 17 s | |
| SASRec (early-stopped at epoch 50, best 35; ~12–17 min/epoch) | 10.8 h | 3.9 GB |

**Track communities (cuGraph Louvain + PageRank, top 20k tracks, edges shared by ≥ 25 playlists: 11.5M edges).** The graph builds in 3.5 min and Louvain + PageRank take 16 s on the GPU. It splits into 10 communities (modularity 0.51) that line up with real scenes, even though the model never sees a genre label. The five largest are below; the smaller five are Latin (J Balvin, Don Omar), worship (Hillsong), Broadway (Hamilton, Wicked), Christmas (Michael Bublé, Bing Crosby) and K-pop (BTS, EXO).

| Community | Size | Most central tracks (PageRank) |
|---|---:|---|
| Pop / EDM | 7,004 | The Chainsmokers - Closer · The Chainsmokers - Roses · Vance Joy - Riptide |
| Classic rock | 3,734 | Queen - Bohemian Rhapsody · Journey - Don't Stop Believin' · Lynyrd Skynyrd - Sweet Home Alabama |
| Current hip-hop | 3,446 | Kendrick Lamar - HUMBLE. · Drake - One Dance · DRAM - Broccoli |
| 2000s R&B / hip-hop | 2,458 | R. Kelly - Ignition (Remix) · Kanye West - Gold Digger · Usher - Yeah! |
| Country | 1,397 | Sam Hunt - Body Like A Back Road · Thomas Rhett - Die A Happy Man · Zac Brown Band - Chicken Fried |

The co-occurrence product is computed as dense, chunked cuBLAS matrix multiplies (`graph._upper_edges_gpu`). cuSPARSE SpGEMM would need a 21 GB scratch buffer at this scale.

On the 50-slice subset (46k playlists) the same pipeline gives co-occurrence 0.155, tuned hybrid 0.164 recall@50, so more data helped every model, SASRec most (0.105 → 0.125).

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

- [x] Run on real MPD slices (50, then all 1,000)
- [x] Fill the GPU column of the benchmark (WSL2, RTX 3050), at 0.7M and 7M rows
- [ ] Benchmark at full MPD scale (66M rows) on Colab T4 (16 GB)
- [x] Sentence-transformer title encoder vs TF-IDF (+ hybrid encoder, unseen-phrasing test)
- [x] SASRec future-window targets (`sasrec.target_window`) and longer training
- [x] Hybrid model: reciprocal rank fusion of co-occurrence + SASRec
- [x] Validation split + early stopping for SASRec (`sasrec.val_size`, `patience`)
- [x] Add the title model to the hybrid for playlists with few seed tracks
- [ ] Use cuGraph communities/PageRank as features
- [x] Small web demo (Streamlit): type a title or pick seed songs → playlist
