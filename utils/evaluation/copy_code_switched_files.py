#!/usr/bin/env python3
"""
Copy Code-Switched Audio Files
===============================
Reads the code-switch JSON file (which records original row indexes from the
metadata CSV), looks up the corresponding filename for each index, and copies
matching audio files to a destination folder.

Use --lang to restrict to a specific language:
  --lang french   → only entries with french_word_count > 0
  --lang english  → only entries with english_word_count > 0
  --lang both     → entries that have BOTH french and english words (default: any)

Dataset structure expected
--------------------------
  <root>/metadata.csv   – pipe-separated: filename | transcription | speaker | gender
  <root>/audio/         – audio files

Usage
-----
python copy_code_switched_files.py \
    --json      ../../data/cs-evaluation/stats/code_switch_counts.json \
    --metadata  ../../data/corpus.csv \
    --audiodir  ../../submissions/omnivoice-ft-cs/audio/ \
    --destdir   ../../data/cs-evaluation/outputs/omnivoice-ft-cs/ \
    [--delimiter "|"] \
    [--lang french|english|both] \
    [--include-zero]
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Copy code-switched audio files identified by original row index.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--json",         type=Path, required=True,
                   help="Path to the code-switch JSON file.")
    p.add_argument("--metadata",     type=Path, required=True,
                   help="Path to metadata CSV (pipe-separated by default).")
    p.add_argument("--audiodir",     type=Path, required=True,
                   help="Source audio directory.")
    p.add_argument("--destdir",      type=Path, required=True,
                   help="Destination folder for copied audio files.")
    p.add_argument("--delimiter",    default="|",
                   help="CSV delimiter (default: pipe |).")
    p.add_argument("--encoding",     default="utf-8")
    p.add_argument("--lang",
                   choices=["french", "english", "both"],
                   default=None,
                   help=(
                       "Filter by code-switching language: "
                       "'french' (fr > 0), 'english' (en > 0), "
                       "'both' (fr > 0 AND en > 0). "
                       "Default: any foreign word (fr > 0 OR en > 0)."
                   ))
    p.add_argument("--include-zero", action="store_true",
                   help="Ignore all language filters and copy every entry in the JSON.")
    return p.parse_args()


def is_code_switched(entry: dict, lang: str | None) -> bool:
    """Return True if the entry passes the language filter."""
    counts = entry["counts"]
    fr = counts.get("french_word_count", 0)
    en = counts.get("english_word_count", 0)

    if lang == "french":
        return fr > 0
    if lang == "english":
        return en > 0
    if lang == "both":
        return fr > 0 and en > 0
    # default: any foreign word
    return (fr + en) > 0


def resolve_audio_path(audiodir: Path, filename: str) -> Path | None:
    """
    Try to locate the audio file:
      1. audiodir / filename  (as stored in the metadata)
      2. audiodir / stem + common extensions  (in case extension differs)
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


def lang_tag(entry: dict) -> str:
    """Build a short human-readable tag showing language counts."""
    counts = entry["counts"]
    fr = counts.get("french_word_count", 0)
    en = counts.get("english_word_count", 0)
    parts = []
    if fr:
        words = ", ".join(counts.get("french_words", []))
        parts.append(f"fr={fr} ({words})")
    if en:
        words = ", ".join(counts.get("english_words", []))
        parts.append(f"en={en} ({words})")
    return " | ".join(parts) if parts else "no foreign words"


def main() -> None:
    args = parse_args()

    # ── Load JSON ─────────────────────────────────────────────────────────────
    if not args.json.exists():
        sys.exit(f"Error: JSON file not found – {args.json}")

    with args.json.open(encoding="utf-8") as fh:
        all_entries: list[dict] = json.load(fh)

    # Apply filter
    if args.include_zero:
        entries = all_entries
        filter_desc = "all entries (--include-zero)"
    else:
        entries = [e for e in all_entries if is_code_switched(e, args.lang)]
        lang_desc = args.lang if args.lang else "french or english"
        filter_desc = f"code-switched ({lang_desc})"

    if not entries:
        sys.exit(
            f"No entries match the current filter ({filter_desc}).\n"
            "Try --include-zero or a different --lang value."
        )

    target_indexes: dict[int, dict] = {e["original_index"]: e for e in entries}
    print(f"Targeting {len(target_indexes)} index(es)  [{filter_desc}]")

    # ── Load metadata CSV → map row index → filename ──────────────────────────
    if not args.metadata.exists():
        sys.exit(f"Error: metadata file not found – {args.metadata}")

    index_to_filename: dict[int, str] = {}
    with args.metadata.open(encoding=args.encoding, newline="") as fh:
        reader = csv.DictReader(fh, delimiter=args.delimiter)
        if "filename" not in (reader.fieldnames or []):
            sys.exit(
                f"'filename' column not found in CSV.\n"
                f"Available columns: {', '.join(reader.fieldnames or [])}"
            )
        for row_idx, row in enumerate(reader):
            if row_idx in target_indexes:
                index_to_filename[row_idx] = row["filename"].strip()

    missing_index = set(target_indexes.keys()) - set(index_to_filename.keys())
    if missing_index:
        print(
            f"  [WARN] {len(missing_index)} index(es) not found in CSV "
            f"(CSV may be shorter than expected): {sorted(missing_index)}"
        )

    # ── Prepare destination ───────────────────────────────────────────────────
    args.destdir.mkdir(parents=True, exist_ok=True)

    # ── Copy files ────────────────────────────────────────────────────────────
    copied = 0
    missing_audio = []

    for row_idx, filename in sorted(index_to_filename.items()):
        entry = target_indexes[row_idx]
        tag   = lang_tag(entry)

        src = resolve_audio_path(args.audiodir, filename)
        if src is None:
            print(f"  [WARN] Audio not found: {filename}  (index {row_idx})")
            missing_audio.append(filename)
            continue

        dest = args.destdir / src.name
        shutil.copy2(src, dest)
        print(f"  Copied [idx {row_idx:>4} | {tag}]  {src.name}")
        copied += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n✓ {copied}/{len(target_indexes)} file(s) copied to {args.destdir}")
    if missing_audio:
        print(f"✗ {len(missing_audio)} audio file(s) not found on disk:")
        for f in missing_audio:
            print(f"    {f}")


if __name__ == "__main__":
    main()