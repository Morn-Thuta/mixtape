# CPU vs GPU benchmark (696,845 interactions)

| stage | CPU (s) | GPU (s) | speedup |
|---|---:|---:|---:|
| prep.read | 0.136 | - | - |
| prep.filter_tracks | 0.189 | - | - |
| prep.filter_playlists | 0.048 | - | - |
| prep.index | 0.183 | - | - |
| prep.split | 0.073 | - | - |
| prep.write | 0.091 | - | - |
| prep.total | 0.720 | - | - |
| cooccur.fit | 0.042 | - | - |
| cooccur.recommend | 0.272 | - | - |
| graph.build | 0.249 | - | - |
| graph.louvain+pagerank | 11.119 | - | - |

_GPU column empty: RAPIDS not installed here. Run `python -m mixtape bench` on Colab or WSL2._
