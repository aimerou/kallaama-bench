#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets==3.6.0",
#     "soundfile",
#     "librosa",
#     "tqdm",
# ]
# ///
"""
hf_dataset_download.py
-----------------------
Download a TTS dataset from Hugging Face and save audio files + metadata.
Supports a single split or multiple splits (which get merged into one dataset).

Requirements
------------
Dependencies are declared in the inline script metadata above (installed by `uv run`).
export HF_TOKEN=your_huggingface_token

Example Usage
-------------
# Single split
uv run hf_dataset_download.py --dataset user/repo-name --splits train --dest /path/to/dest/

# Multiple splits (merged)
uv run hf_dataset_download.py --dataset user/repo-name --splits train test --dest /path/to/dest/
"""

import os
import csv
import argparse
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
import soundfile as sf

# Defaults (can be overridden via CLI args)
DEFAULT_DATASET = "user/repo-name"
DEFAULT_DEST    = "/root/kallaama-bench/data/"
DEFAULT_SPLITS  = ["train", "test"]

# Dataset field names
AUDIO_FIELD   = "audio"
TEXT_FIELD    = "transcription"
SPEAKER_FIELD = "speaker"
GENDER_FIELD  = "gender"


def load_splits(dataset_name, splits, hf_token):
    """Load one or more splits and return a single merged dataset."""
    loaded = []
    for split in splits:
        print(f"Loading '{split}' split...")
        ds = load_dataset(dataset_name, split=split, token=hf_token)
        print(f"  → {len(ds)} samples")
        loaded.append(ds)

    if len(loaded) == 1:
        return loaded[0]

    print(f"\nMerging {len(splits)} splits: {splits}...")
    merged = concatenate_datasets(loaded)
    print(f"Total samples after merge: {len(merged)}")
    return merged


def download_tts_dataset(dataset_name, dest_folder, splits, max_samples=None):
    """
    Download a TTS dataset from Hugging Face and save audio files + metadata.

    Args:
        dataset_name: HF dataset identifier (e.g. "org/dataset-name")
        dest_folder:  Root output directory
        splits:       List of split names to load (e.g. ["train"] or ["train", "test"])
        max_samples:  Optional cap on the number of samples to process
    """
    hf_token = os.getenv("HF_TOKEN")

    # Build a descriptive output subfolder: "train", "test", or "train+test"
    split_tag   = "+".join(splits)
    dest_folder = os.path.join(dest_folder, split_tag)
    audio_folder = os.path.join(dest_folder, "wavs")
    os.makedirs(audio_folder, exist_ok=True)

    dataset = load_splits(dataset_name, splits, hf_token)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"Capped to {len(dataset)} samples")

    print(f"\nDataset features: {dataset.features}")

    available_fields = dataset.features.keys()
    has_speaker = SPEAKER_FIELD in available_fields
    has_gender  = GENDER_FIELD  in available_fields

    csv_path = os.path.join(dest_folder, "metadata.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["filename", "transcription", "speaker", "gender"],
            delimiter="|"
        )
        writer.writeheader()

        for idx, sample in enumerate(tqdm(dataset, desc="Saving audio files")):
            filename   = f"audio_{idx:05d}.wav"
            audio_data = sample[AUDIO_FIELD]
            sf.write(
                os.path.join(audio_folder, filename),
                audio_data["array"],
                audio_data["sampling_rate"],
            )
            writer.writerow({
                "filename":      filename,
                "transcription": str(sample[TEXT_FIELD]),
                "speaker":       str(sample[SPEAKER_FIELD]) if has_speaker else "",
                "gender":        str(sample[GENDER_FIELD]) if has_gender else "",
            })

    print(f"\n✓ Done! Splits processed : {splits}")
    print(f"✓ Audio files : {audio_folder}")
    print(f"✓ Metadata    : {csv_path}")
    print(f"✓ Total files : {len(dataset)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download a HuggingFace TTS dataset (one or more splits)."
    )
    parser.add_argument(
        "--dataset", default=DEFAULT_DATASET,
        help="HuggingFace dataset identifier (default: %(default)s)"
    )
    parser.add_argument(
        "--dest", default=DEFAULT_DEST,
        help="Root output directory (default: %(default)s)"
    )
    parser.add_argument(
        "--splits", nargs="+", default=DEFAULT_SPLITS,
        help="One or more split names, e.g. --splits train test (default: %(default)s)"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Cap the number of samples to process (default: no limit)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_tts_dataset(
        dataset_name=args.dataset,
        dest_folder=args.dest,
        splits=args.splits,
        max_samples=args.max_samples,
    )