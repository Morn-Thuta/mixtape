---
title: mixtape
emoji: 🎧
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: Playlist continuation on 1M Spotify playlists
---

# mixtape demo

Build a playlist from a name, continue one from a few songs, or browse the music scenes
that cuGraph found in the Spotify Million Playlist Dataset.

Models were trained and evaluated on a GPU with RAPIDS (cuDF, CuPy, cuGraph) and transformers
(SASRec, MPNet); this Space serves co-occurrence + the MPNet/TF-IDF title model on CPU.
Code, results and benchmarks: https://github.com/Morn-Thuta/mixtape

The playlist data is not in this repo (the dataset's terms don't allow re-sharing it).
The app downloads it at start-up from a private dataset repo:
- Space variable `MIXTAPE_DATA_REPO` = that dataset repo's id
- Space secret `HF_TOKEN` = a read token that can access it

First start takes a few minutes (download + loading 1M playlists); after that it's instant
until the Space goes to sleep.
