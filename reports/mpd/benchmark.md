# CPU vs GPU benchmark (63,444,639 interactions)

Best of 1 CPU / 2 GPU passes. `cooccur.recommend` times 2,000 test playlists. The graph uses the top 5,000 tracks, edges with weight ≥ 25.

| stage | CPU (s) | GPU (s) | speedup |
|---|---:|---:|---:|
| prep.read | 1.877 | 1.5821 | 1.2x |
| prep.filter_tracks | 44.613 | 0.9124 | 48.9x |
| prep.filter_playlists | 3.052 | 0.2679 | 11.4x |
| prep.index | 18.484 | 5.1484 | 3.6x |
| prep.split | 10.064 | 2.3767 | 4.2x |
| prep.write | 10.564 | 2.4487 | 4.3x |
| prep.total | 88.653 | 12.7362 | 7.0x |
| cooccur.fit | 6.082 | 16.7399 | 0.4x |
| cooccur.recommend | 63.301 | 33.9374 | 1.9x |
| graph.build | 19.244 | 25.433 | 0.8x |
| graph.louvain+pagerank | 118.152 | 3.277 | 36.1x |
