"""Flatten MPD json slices (real or synthetic) into compact parquet tables.

Built for the full Million Playlist Dataset (66M rows) on a machine with ~8 GB RAM
and a 4 GB GPU:
  * slices are processed in chunks and streamed to parquet part files, so memory
    stays flat no matter how many slices there are
  * track URIs (~36-char strings) are replaced by integer ids (`tid`) at read time,
    so the interactions table is pure int32/int16 - ~10 bytes per row, which fits
    the whole MPD (~0.7 GB) into GPU memory for cuDF

Outputs in data/interim/:
  interactions/part-*.parquet  pid:int32, pos:int16, tid:int32  (one row per playlist entry)
  playlists.parquet            pid, name
  tracks.parquet               tid, track_uri, track_name, artist_name, album_name
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def ingest(raw_dir: Path, out_dir: Path, max_slices: int | None = None,
           slices_per_part: int = 50) -> dict[str, int]:
    files = sorted(raw_dir.glob("mpd.slice.*.json"),
                   key=lambda p: int(p.name.split(".")[2].split("-")[0]))
    if not files:
        raise FileNotFoundError(
            f"No mpd.slice.*.json files in {raw_dir}. Run `python -m mixtape synth` "
            "or copy the Spotify MPD 'data/' folder there."
        )
    if max_slices:
        files = files[:max_slices]

    out_dir.mkdir(parents=True, exist_ok=True)
    part_dir = out_dir / "interactions"
    if part_dir.exists():
        shutil.rmtree(part_dir)
    part_dir.mkdir()
    legacy = out_dir / "interactions.parquet"
    if legacy.exists():
        legacy.unlink()

    uri_to_tid: dict[str, int] = {}
    track_meta: list[tuple[str, str, str]] = []
    pl_ids: list[int] = []
    pl_names: list[str] = []
    n_rows = 0

    for p_idx, start in enumerate(tqdm(range(0, len(files), slices_per_part), desc="ingest")):
        pids, poss, tids = [], [], []
        for fp in files[start:start + slices_per_part]:
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)
            for p in data["playlists"]:
                pid = p["pid"]
                pl_ids.append(pid)
                pl_names.append(p.get("name", ""))
                for t in p["tracks"]:
                    uri = t["track_uri"]
                    tid = uri_to_tid.get(uri)
                    if tid is None:
                        tid = uri_to_tid[uri] = len(track_meta)
                        track_meta.append((t.get("track_name", ""), t.get("artist_name", ""),
                                           t.get("album_name", "")))
                    pids.append(pid)
                    poss.append(t["pos"])
                    tids.append(tid)
        pd.DataFrame({
            "pid": np.asarray(pids, dtype=np.int32),
            "pos": np.asarray(poss, dtype=np.int16),
            "tid": np.asarray(tids, dtype=np.int32),
        }).to_parquet(part_dir / f"part-{p_idx:05d}.parquet", index=False)
        n_rows += len(pids)

    pd.DataFrame({"pid": np.asarray(pl_ids, dtype=np.int32), "name": pl_names}) \
        .to_parquet(out_dir / "playlists.parquet", index=False)
    meta = pd.DataFrame(track_meta, columns=["track_name", "artist_name", "album_name"])
    meta.insert(0, "track_uri", list(uri_to_tid.keys()))   # dict keeps insertion order
    meta.insert(0, "tid", np.arange(len(meta), dtype=np.int32))
    meta.to_parquet(out_dir / "tracks.parquet", index=False)
    return {"slices": len(files), "playlists": len(pl_ids), "interactions": n_rows,
            "tracks": len(meta)}
