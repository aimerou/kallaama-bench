"""
extract_corpus.py
-----------------
Extracts transcriptions from a pipe-separated metadata file into a corpus.txt,
one transcription per line.

Usage
-----
python extract_corpus.py --metadata /root/kallaama-bench/data/metadata.csv --output /root/kallaama-bench/data/corpus.txt
"""

import argparse
import csv
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract transcriptions from a metadata file into corpus.txt."
    )
    parser.add_argument(
        "--metadata", default="metadata.csv",
        help="Path to the pipe-separated metadata file (default: metadata.csv)",
    )
    parser.add_argument(
        "--output", default="corpus.txt",
        help="Path to the output corpus file (default: corpus.txt)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    meta_path = Path(args.metadata)
    if not meta_path.exists():
        sys.exit(
            f"[ERROR] Metadata file not found: {meta_path}\n"
            f"Run create_metadata.py first to generate it."
        )

    transcriptions = []
    with meta_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="|")

        if "transcription" not in reader.fieldnames:
            sys.exit(
                f"[ERROR] No 'transcription' column found in {meta_path}.\n"
                f"Available columns: {', '.join(reader.fieldnames)}"
            )

        for row in reader:
            text = row["transcription"].strip()
            if text:
                transcriptions.append(text)

    if not transcriptions:
        sys.exit("[ERROR] No transcriptions found in the metadata file.")

    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(transcriptions) + "\n")

    print(f"[INFO] Extracted {len(transcriptions)} transcriptions → {output_path}")


if __name__ == "__main__":
    main()