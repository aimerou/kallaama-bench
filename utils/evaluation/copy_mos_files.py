#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Copy MOS Evaluation Audio Files
===============================
Reads the medoids JSON containing MOS evaluation files and copies every
matching audio file from the source dataset into a flat destination folder.

Usage
-----
uv run copy_mos_files.py \
    --medoids  ../../data/mos-evaluation/clustering/medoids.json \
    --audiodir ../../submissions/omnivoice-ft-cs/audio \
    --destdir  ../../data/mos-evaluation/outputs/omnivoice-ft-cs/
"""

import argparse
import json
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Copy medoid audio files to a destination folder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--medoids",  type=Path, required=True,
                   help="Path to medoids.json produced by cluster_tts_samples.py.")
    p.add_argument("--audiodir", type=Path, required=True,
                   help="Source audio directory (data/audio/).")
    p.add_argument("--destdir",  type=Path, required=True,
                   help="Destination folder for copied audio files.")
    return p.parse_args()


def resolve_audio_path(audiodir: Path, filename: str) -> Path | None:
    """
    Try to locate the audio file:
      1. audiodir / filename  (as stored in the JSON)
      2. audiodir / stem + each common extension  (in case extension differs)
    """
    direct = audiodir / filename
    if direct.exists():
        return direct

    stem = Path(filename).stem
    for ext in (".wav", ".mp3", ".flac", ".ogg", ".m4a"):
        candidate = audiodir / (stem + ext)
        if candidate.exists():
            return candidate

    return None


def main() -> None:
    args = parse_args()

    # ── Load medoids ──────────────────────────────────────────────────────────
    if not args.medoids.exists():
        sys.exit(f"Error: medoids file not found – {args.medoids}")

    with args.medoids.open(encoding="utf-8") as fh:
        medoids: list[dict] = json.load(fh)

    print(f"Loaded {len(medoids)} medoid(s) from {args.medoids}")

    # ── Prepare destination ───────────────────────────────────────────────────
    args.destdir.mkdir(parents=True, exist_ok=True)

    # ── Copy files ────────────────────────────────────────────────────────────
    copied  = 0
    missing = []

    for entry in medoids:
        filename = entry["filename"].strip()
        src = resolve_audio_path(args.audiodir, filename)

        if src is None:
            print(f"  [WARN] Not found: {filename}")
            missing.append(filename)
            continue

        dest = args.destdir / src.name
        shutil.copy2(src, dest)
        print(f"  Copied [{entry['speaker']} | cluster {entry['cluster']}]  "
              f"{src.name}  →  {dest}")
        copied += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n✓ {copied}/{len(medoids)} file(s) copied to {args.destdir}")
    if missing:
        print(f"✗ {len(missing)} file(s) not found:")
        for f in missing:
            print(f"    {f}")


if __name__ == "__main__":
    main()