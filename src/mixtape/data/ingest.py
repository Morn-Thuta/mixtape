"""Flatten MPD json slices (real or synthetic) into parquet tables.

Outputs in data/interim/:
  interactions.parquet  pid, pos, track_uri       (one row per playlist entry)
  playlists.parquet     pid, name
  tracks.parquet        track_uri, track_name, artist_name, album_name
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def ingest(raw_dir: Path, out_dir: Path, max_slices: int | None = None) -> dict[str, int]:
    files = sorted(raw_dir.glob("mpd.slice.*.json"))
    if not files:
        raise FileNotFoundError(
            f"No mpd.slice.*.json files in {raw_dir}. Run `python -m mixtape synth` "
            "or copy the Spotify MPD 'data/' folder there."
        )
    if max_slices:
        files = files[:max_slices]

    inter, pls, tracks = [], [], {}
    for fp in tqdm(files, desc="ingest"):
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        for p in data["playlists"]:
            pid = p["pid"]
            pls.append((pid, p.get("name", "")))
            for t in p["tracks"]:
                uri = t["track_uri"]
                inter.append((pid, t["pos"], uri))
                if uri not in tracks:
                    tracks[uri] = (t.get("track_name", ""), t.get("artist_name", ""),
                                   t.get("album_name", ""))

    out_dir.mkdir(parents=True, exist_ok=True)
    df_i = pd.DataFrame(inter, columns=["pid", "pos", "track_uri"])
    df_i[["pid", "pos"]] = df_i[["pid", "pos"]].astype("int32")
    df_i.to_parquet(out_dir / "interactions.parquet", index=False)
    pd.DataFrame(pls, columns=["pid", "name"]).to_parquet(out_dir / "playlists.parquet", index=False)
    pd.DataFrame(
        [(u, *v) for u, v in tracks.items()],
        columns=["track_uri", "track_name", "artist_name", "album_name"],
    ).to_parquet(out_dir / "tracks.parquet", index=False)
    return {"slices": len(files), "playlists": len(pls), "interactions": len(df_i),
            "tracks": len(tracks)}
