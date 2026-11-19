#!/usr/bin/env bash
# Real Spotify MPD, first 50 slices (50k playlists): ingest -> GPU prep -> all models -> graph.
set -euo pipefail
source ~/.venvs/mixtape/bin/activate
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
CFG=configs/mpd50.yaml
python -m mixtape ingest --config $CFG --max-slices 50
python -m mixtape prep --config $CFG
python -m mixtape train --config $CFG --models popularity cooccur sasrec hybrid title
python -m mixtape graph --config $CFG
