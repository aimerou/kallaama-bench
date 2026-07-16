#!/usr/bin/env python3
"""
French and English word counter per sentence (Wolof dataset).

Usage
-----
python  counter_code_switched.py ../../data/corpus.csv \
    --outdir ../../data/cs-evaluation/stats/ \
    --column transcription \
    --delimiter "|" \
    --window 1 \
    --confidence 0.5
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import fasttext
from huggingface_hub import hf_hub_download


# ── FastText model loading ──────────────────────────────────────────────────

def load_lid_model():
    lid_model = fasttext.load_model(
        hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin")
    )
    return lid_model


# ── Helpers ─────────────────────────────────────────────────────────────────

def tokenize_words(sentence: str) -> list[str]:
    """
    Tokenize into whitespace-separated tokens while preserving simple forms.
    We keep tokens as-is (FastText context uses raw tokens).
    """
    # Split on whitespace; keep punctuation attached (FastText is robust to n-grams)
    # But later we’ll only count tokens whose “clean” form is alphabetic.
    return sentence.split()


def clean_token_alpha(token: str) -> bool:
    """Whether token is alphabetic (after stripping common punctuation edges)."""
    t = token.strip()
    # If you want stricter behavior, change this.
    return bool(re.fullmatch(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", t))


def identify_languages_contextual(
    words: list[str],
    lid_model,
    window: int,
    confidence: float,
    k: int = 2,
) -> list[str | None]:
    """
    Classify each word as 'fr', 'wo', or None using a sliding context window.

    Returns a list aligned with `words`, containing:
      - "fr" if French label is top-1 and score >= confidence
      - "wo" if Wolof label is top-1 and score >= confidence
      - None otherwise
    """
    results: list[str | None] = []
    for i, word in enumerate(words):
        if not clean_token_alpha(word):
            results.append(None)
            continue

        start = max(0, i - window)
        end = min(len(words), i + window + 1)
        context = " ".join(words[start:end])

        labels, scores = lid_model.predict(context, k=k)

        lang = None
        if labels[0] == "__label__fra_Latn" and scores[0] >= confidence:
            lang = "fr"
        elif labels[0] == "__label__eng_Latn" and scores[0] >= confidence:
            lang = "en"
        elif labels[0] == "__label__wol_Latn" and scores[0] >= confidence:
            lang = "wo"

        results.append(lang)
    return results


def analyse_sentence_code_switched_words(
    sentence: str,
    lid_model,
    window: int,
    confidence: float,
) -> dict:
    """
    Returns per-sentence counts and optional breakdown for French and English.
    """
    words = tokenize_words(sentence)
    langs = identify_languages_contextual(
        words=words,
        lid_model=lid_model,
        window=window,
        confidence=confidence,
        k=3,
    )

    fr_words = [words[i] for i, lang in enumerate(langs) if lang == "fr"]
    en_words = [words[i] for i, lang in enumerate(langs) if lang == "en"]

    return {
        "french_word_count": len(fr_words),
        "french_words": fr_words,
        "english_word_count": len(en_words),
        "english_words": en_words,
    }


def plot_distribution(
    fr_counts: list[int],
    en_counts: list[int],
    output_path: Path,
) -> None:
    """
    Plot side-by-side distribution of French and English word counts across sentences.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for ax, counts, label, color in (
        (axes[0], fr_counts, "French", "steelblue"),
        (axes[1], en_counts, "English", "darkorange"),
    ):
        if not counts:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        else:
            dist = Counter(counts)
            x_vals = sorted(dist.keys())
            y_vals = [dist[x] for x in x_vals]
            ax.bar([str(x) for x in x_vals], y_vals, color=color)
            ax.tick_params(axis="x", rotation=0)

        ax.set_title(f"Distribution of {label} words per sentence")
        ax.set_xlabel(f"{label} word count")
        ax.set_ylabel("Number of sentences")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count occurrences of French words per sentence using FastText language identification.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "csv_file",
        type=Path,
        help="Path to the input CSV file.",
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Output directory containing both JSON results and the plot image.",
    )

    parser.add_argument(
        "-c",
        "--column",
        default="transcription",
        help="CSV column name containing sentences (default: transcription).",
    )

    parser.add_argument(
        "--delimiter",
        default=",",
        help="CSV delimiter character (default: comma).",
    )

    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="CSV file encoding (default: utf-8).",
    )

    parser.add_argument(
        "--window",
        type=int,
        default=2,
        help="Sliding context window size (number of tokens on each side). Default: 2.",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.6,
        help="Minimum FastText top-1 score required to label a token as French. Default: 0.6.",
    )

    parser.add_argument(
        "--no-words-list",
        dest="no_french_words_list",
        action="store_true",
        help="Do not store the detected French/English word lists per sentence (reduces JSON size).",
    )

    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plot generation (still writes JSON).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    if not args.csv_file.exists():
        sys.exit(f"Error: file not found – {args.csv_file}")

    lid_model = load_lid_model()

    # Read + analyse
    records: list[dict] = []
    total_sentences = 0
    sentences_with_fr = 0
    sentences_with_en = 0
    french_counts: list[int] = []
    english_counts: list[int] = []

    with args.csv_file.open(encoding=args.encoding, newline="") as fh:
        reader = csv.DictReader(fh, delimiter=args.delimiter)

        if args.column not in (reader.fieldnames or []):
            available = ", ".join(reader.fieldnames or [])
            sys.exit(
                f"Error: column '{args.column}' not found in CSV.\n"
                f"Available columns: {available}"
            )

        for idx, row in enumerate(reader):
            sentence = (row.get(args.column) or "").strip()
            if not sentence:
                continue

            total_sentences += 1

            analysis = analyse_sentence_code_switched_words(
                sentence=sentence,
                lid_model=lid_model,
                window=args.window,
                confidence=args.confidence,
            )

            fr_count = analysis["french_word_count"]
            en_count = analysis["english_word_count"]
            french_counts.append(fr_count)
            english_counts.append(en_count)

            if fr_count > 0:
                sentences_with_fr += 1
            if en_count > 0:
                sentences_with_en += 1

            rec = {
                "original_index": idx,  # 0-based row index (header not counted)
                "sentence": sentence,
                "counts": {
                    "french_word_count": fr_count,
                    "english_word_count": en_count,
                },
            }
            if not args.no_french_words_list:
                rec["counts"]["french_words"] = analysis.get("french_words", [])
                rec["counts"]["english_words"] = analysis.get("english_words", [])

            records.append(rec)

    # Log percentage
    fr_pct = (sentences_with_fr / total_sentences * 100.0) if total_sentences else 0.0
    en_pct = (sentences_with_en / total_sentences * 100.0) if total_sentences else 0.0
    print(
        f"Sentences processed: {total_sentences}\n"
        f"Sentences containing >=1 French word:  {sentences_with_fr} ({fr_pct:.2f}%)\n"
        f"Sentences containing >=1 English word: {sentences_with_en} ({en_pct:.2f}%)"
    )

    # Rank by combined code-switched word count desc
    records.sort(
        key=lambda r: r["counts"]["french_word_count"] + r["counts"]["english_word_count"],
        reverse=True,
    )

    # Write JSON
    json_path = args.outdir / "code_switch_counts.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"JSON written to: {json_path}")

    # Plot distribution
    if not args.no_plot:
        plot_path = args.outdir / "code_switch_distribution.png"
        plot_distribution(french_counts, english_counts, plot_path)
        print(f"Plot written to: {plot_path}")


if __name__ == "__main__":
    main()