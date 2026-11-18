# CPU vs GPU benchmark (7,006,085 interactions)

Best of 3 passes per backend (first GPU pass includes CUDA start-up and is excluded).

| stage | CPU (s) | GPU (s) | speedup |
|---|---:|---:|---:|
| prep.read | 0.481 | 0.2133 | 2.3x |
| prep.filter_tracks | 3.442 | 0.1523 | 22.6x |
| prep.filter_playlists | 0.854 | 0.061 | 14.0x |
| prep.index | 2.907 | 0.2468 | 11.8x |
| prep.split | 0.598 | 0.1562 | 3.8x |
| prep.write | 0.967 | 0.2651 | 3.6x |
| prep.total | 9.383 | 1.1011 | 8.5x |
| cooccur.fit | 0.397 | 0.5761 | 0.7x |
| cooccur.recommend | 32.733 | 8.4899 | 3.9x |
| graph.build | 1.292 | 0.711 | 1.8x |
| graph.louvain+pagerank | 17.846 | 0.374 | 47.7x |
