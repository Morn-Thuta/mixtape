# CPU vs GPU benchmark (696,845 interactions)

Best of 3 passes per backend (first GPU pass includes CUDA start-up and is excluded).

| stage | CPU (s) | GPU (s) | speedup |
|---|---:|---:|---:|
| prep.read | 0.046 | 0.0619 | 0.7x |
| prep.filter_tracks | 0.160 | 0.0373 | 4.3x |
| prep.filter_playlists | 0.080 | 0.0192 | 4.2x |
| prep.index | 0.168 | 0.073 | 2.3x |
| prep.split | 0.042 | 0.0457 | 0.9x |
| prep.write | 0.161 | 0.1242 | 1.3x |
| prep.total | 0.731 | 0.3613 | 2.0x |
| cooccur.fit | 0.036 | 0.0477 | 0.8x |
| cooccur.recommend | 0.315 | 0.1767 | 1.8x |
| graph.build | 0.210 | 0.151 | 1.4x |
| graph.louvain+pagerank | 8.694 | 0.303 | 28.7x |
