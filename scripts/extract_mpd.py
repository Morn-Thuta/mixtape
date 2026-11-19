"""Extract the Spotify MPD zip into data/mpd/raw and verify every slice's md5.

Run:  python scripts/extract_mpd.py <path/to/spotify_million_playlist_dataset.zip> [out_dir]
"""
from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    zpath = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "data" / "mpd" / "raw"
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        expected = {}
        for line in z.read("md5sums").decode().splitlines():
            parts = line.split()
            if len(parts) == 2:
                expected[Path(parts[1]).name] = parts[0]
        for name in ("README.md", "license.txt", "stats.txt"):
            if name in z.namelist():
                (out.parent / name).write_bytes(z.read(name))
        slices = sorted(n for n in z.namelist() if "mpd.slice." in n and n.endswith(".json"))
        bad = []
        for i, n in enumerate(slices, 1):
            target = out / Path(n).name
            h = hashlib.md5()
            with z.open(n) as src, open(target, "wb") as dst:
                for chunk in iter(lambda: src.read(1 << 20), b""):
                    h.update(chunk)
                    dst.write(chunk)
            want = expected.get(target.name)
            if want and want != h.hexdigest():
                bad.append(target.name)
            if i % 100 == 0:
                print(f"extracted {i}/{len(slices)}", flush=True)
    print(f"done: {len(slices)} slices -> {out}; md5 checked {sum(1 for s in slices if Path(s).name in expected)}, "
          f"mismatches: {bad or 'none'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
