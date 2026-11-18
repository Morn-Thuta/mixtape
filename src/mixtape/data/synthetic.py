"""Generate a synthetic dataset in the exact Spotify Million Playlist Dataset format.

Lets the whole pipeline run before you have the real data. The generator plants
structure the models can learn:
  * moods      -> which tracks co-occur (graph / co-occurrence signal)
  * sub-scenes -> playlists drift between neighbouring sub-scenes in order
                  (sequential signal for SASRec)
  * titles     -> mood-specific vocabulary (signal for the title encoder)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MOOD_TITLES = [
    ["gym", "workout", "lift", "beast mode", "gains"],
    ["chill", "chill vibes", "relax", "lofi", "mellow"],
    ["rainy day", "rainy sunday", "cozy", "rain", "grey skies"],
    ["party", "pregame", "turn up", "lit", "friday night"],
    ["study", "focus", "deep work", "exam szn", "homework"],
    ["road trip", "drive", "highway", "car jams", "long drive"],
    ["sad", "heartbreak", "crying", "in my feels", "sad boi hours"],
    ["throwback", "2000s", "nostalgia", "old school", "tbt"],
    ["summer", "beach", "sunshine", "pool party", "hot girl summer"],
    ["sleep", "bedtime", "night", "insomnia", "calm"],
    ["love", "romance", "date night", "crush", "valentines"],
    ["running", "cardio", "run", "5k", "jog"],
    ["country", "boots", "southern", "honky tonk", "backroads"],
    ["rap", "hip hop", "bars", "trap", "rap caviar"],
    ["edm", "rave", "festival", "drops", "house"],
    ["rock", "classic rock", "guitar", "headbang", "grunge"],
    ["jazz", "smooth jazz", "coffee shop", "brunch", "sunday morning"],
    ["kpop", "k-pop", "bias", "stan", "idols"],
    ["christmas", "xmas", "holiday", "winter", "snow"],
    ["worship", "gospel", "church", "praise", "faith"],
    ["latin", "reggaeton", "fiesta", "perreo", "salsa"],
    ["indie", "indie folk", "acoustic", "coffee", "bedroom pop"],
    ["metal", "heavy", "mosh", "screamo", "doom"],
    ["happy", "good vibes", "feel good", "smile", "positive"],
]
DECOR = ["", "", "", " 💪", " 🌧️", " 🔥", " ✨", " vibes", " mix", " playlist", "!!", " 2017"]


def _title(rng: np.random.Generator, mood: int) -> str:
    words = MOOD_TITLES[mood % len(MOOD_TITLES)]
    t = words[rng.integers(len(words))] + DECOR[rng.integers(len(DECOR))]
    return t.upper() if rng.random() < 0.1 else t


def generate(out_dir: Path, n_playlists=20_000, n_tracks=8_000, n_moods=24,
             n_sub=4, slice_size=1_000, seed=42) -> list[Path]:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- tracks: mood, sub-scene, artist, Zipf popularity ---
    track_mood = rng.integers(n_moods, size=n_tracks)
    track_sub = rng.integers(n_sub, size=n_tracks)
    track_artist = track_mood * 1000 + rng.integers(40, size=n_tracks)
    pop = 1.0 / (rng.permutation(n_tracks) + 1) ** 0.8

    buckets: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for m in range(n_moods):
        for s in range(n_sub):
            idx = np.where((track_mood == m) & (track_sub == s))[0]
            p = pop[idx] / pop[idx].sum()
            buckets[(m, s)] = (idx, p)

    def track_obj(t: int, pos: int) -> dict:
        return {
            "pos": pos,
            "artist_name": f"Artist {track_artist[t]}",
            "track_uri": f"spotify:track:syn{t:07d}",
            "artist_uri": f"spotify:artist:syn{track_artist[t]:07d}",
            "track_name": f"Track {t}",
            "album_uri": f"spotify:album:syn{t // 10:07d}",
            "duration_ms": int(rng.integers(120_000, 300_000)),
            "album_name": f"Album {t // 10}",
        }

    paths = []
    for start in range(0, n_playlists, slice_size):
        end = min(start + slice_size, n_playlists)
        playlists = []
        for pid in range(start, end):
            mood = int(rng.integers(n_moods))
            alt = int(rng.integers(n_moods)) if rng.random() < 0.3 else None
            length = int(np.clip(rng.lognormal(3.4, 0.6), 5, 150))
            sub = int(rng.integers(n_sub))
            seen: set[int] = set()
            tracks = []
            tries = 0
            while len(tracks) < length and tries < length * 10:
                tries += 1
                if rng.random() < 0.15:  # drift to neighbouring sub-scene
                    sub = (sub + rng.choice([-1, 1])) % n_sub
                r = rng.random()
                if r < 0.05:      # noise: any track by popularity
                    t = int(rng.choice(n_tracks, p=pop / pop.sum()))
                else:
                    m = alt if (alt is not None and r < 0.25) else mood
                    idx, p = buckets[(m, sub)]
                    if len(idx) == 0:
                        continue
                    t = int(rng.choice(idx, p=p))
                if t in seen:
                    continue
                seen.add(t)
                tracks.append(track_obj(t, len(tracks)))
            playlists.append({
                "name": _title(rng, mood),
                "collaborative": "false",
                "pid": pid,
                "num_tracks": len(tracks),
                "tracks": tracks,
            })
        path = out_dir / f"mpd.slice.{start}-{end - 1}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"info": {"generated_on": "synthetic"}, "playlists": playlists}, f)
        paths.append(path)
    return paths
