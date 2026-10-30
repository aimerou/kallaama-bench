#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Extract Code-Switched Samples to CSV
======================================
Reads the code-switch JSON file, collects every original_index where at least
one French or English word was detected, then filters the metadata CSV to keep
only those rows and writes them to a new CSV file.

Usage
-----
uv run extract_code_switched_csv.py \
    --json      ../../data/cs-evaluation/stats/code_switch_counts.json \
    --metadata  ../../data/corpus.csv \
    --output    ../../data/cs-evaluation/code_switch_corpus.csv \
    [--delimiter "|"] \
    [--lang french|english|both]
"""

import argparse
import csv
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Filter metadata CSV to keep only code-switched samples.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--json",      type=Path, required=True,
                   help="Path to the code-switch JSON file.")
    p.add_argument("--metadata",  type=Path, required=True,
                   help="Path to the source metadata CSV.")
    p.add_argument("--output",    type=Path, required=True,
                   help="Path for the output CSV (code-switched samples only).")
    p.add_argument("--delimiter", default="|",
                   help="CSV delimiter used in both input and output (default: pipe |).")
    p.add_argument("--encoding",  default="utf-8")
    p.add_argument("--lang",
                   choices=["french", "english", "both"],
                   default=None,
                   help=(
                       "Language filter: "
                       "'french' (fr > 0), 'english' (en > 0), "
                       "'both' (fr > 0 AND en > 0). "
                       "Default: any foreign word (fr > 0 OR en > 0)."
                   ))
    return p.parse_args()


def is_code_switched(entry: dict, lang: str | None) -> bool:
    counts = entry["counts"]
    fr = counts.get("french_word_count", 0)
    en = counts.get("english_word_count", 0)
    if lang == "french":
        return fr > 0
    if lang == "english":
        return en > 0
    if lang == "both":
        return fr > 0 and en > 0
    return (fr + en) > 0


def lang_summary(entry: dict) -> str:
    """Short annotation added to each row: detected foreign words."""
    counts = entry["counts"]
    fr_words = counts.get("french_words", [])
    en_words = counts.get("english_words", [])
    parts = []
    if fr_words:
        parts.append(f"fr: {', '.join(fr_words)}")
    if en_words:
        parts.append(f"en: {', '.join(en_words)}")
    return " | ".join(parts)


def main() -> None:
    args = parse_args()

    # ── Load & filter JSON ────────────────────────────────────────────────────
    if not args.json.exists():
        sys.exit(f"Error: JSON file not found – {args.json}")

    with args.json.open(encoding="utf-8") as fh:
        all_entries: list[dict] = json.load(fh)

    switched = [e for e in all_entries if is_code_switched(e, args.lang)]

    if not switched:
        lang_desc = args.lang if args.lang else "french or english"
        sys.exit(f"No code-switched entries found for filter: {lang_desc}")

    # Map original_index → entry for O(1) lookup
    index_map: dict[int, dict] = {e["original_index"]: e for e in switched}

    lang_desc = args.lang if args.lang else "french or english"
    print(f"Found {len(index_map)} code-switched index(es)  [{lang_desc}]")

    # ── Read metadata CSV ─────────────────────────────────────────────────────
    if not args.metadata.exists():
        sys.exit(f"Error: metadata file not found – {args.metadata}")

    matched_rows: list[dict] = []   # original CSV fields
    matched_meta: list[dict] = []   # parallel code-switch annotations

    with args.metadata.open(encoding=args.encoding, newline="") as fh:
        reader = csv.DictReader(fh, delimiter=args.delimiter)
        fieldnames = reader.fieldnames or []

        if not fieldnames:
            sys.exit("CSV appears to have no header row.")

        for row_idx, row in enumerate(reader):
            if row_idx in index_map:
                matched_rows.append(dict(row))
                matched_meta.append(index_map[row_idx])

    print(f"Matched {len(matched_rows)} row(s) in metadata CSV "
          f"(out of {len(index_map)} target indexes)")

    unmatched = len(index_map) - len(matched_rows)
    if unmatched:
        print(f"  [WARN] {unmatched} index(es) exceed the CSV row count and were skipped.")

    # ── Write output CSV ──────────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding=args.encoding, newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="|",
                                extrasaction="ignore")
        writer.writeheader()

        for row, entry in zip(matched_rows, matched_meta):
            writer.writerow(row)
            print(f"  Row {entry['original_index']:>4}  {row.get('filename', ''):30s}"
                  f"  {lang_summary(entry)}")

    print(f"\n✓ {len(matched_rows)} code-switched row(s) saved to {args.output}")


if __name__ == "__main__":
    main()