# /// script
# requires-python = ">=3.10,<3.12"
# dependencies = [
#     "torch==2.5.1",
#     "torchaudio==2.5.1",
#     "TTS==0.19.0",
#     "pandas",
#     "pyyaml",
#     "soundfile",
#     "tqdm",
#     "huggingface_hub",
# ]
# ///
"""
Kiriku-TTS Inference Script
===========================

Single-speaker Coqui-TTS Wolof model: https://huggingface.co/AIHubSN/Kiriku-Wolof-TTS
The repo is gated: accept its conditions on the Hub before running.
The model has no voice cloning, so every corpus row is read in the same voice.

Produces a KallaamaBench submission folder:
    <output_dir>/
    ├── metadata.json
    └── audio/
        ├── <filename from corpus.csv row 1>
        ├── <filename from corpus.csv row 2>
        └── ...

Requirements
------------
1. Dependencies are declared in the inline script metadata above; `uv run` installs them
   in an isolated environment (Coqui TTS 0.19 needs Python < 3.12, uv fetches one).
   TTS 0.19 pins transformers 4.33 and numpy 1.24.3 itself.
2. Set up environment variable for Hugging Face token
    export HF_TOKEN=your_huggingface_token

Usage
-----
uv run infer_kiriku.py --config-path configs/infer_kiriku.yaml
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
import yaml
from huggingface_hub import snapshot_download
from tqdm import tqdm
from TTS.utils.synthesizer import Synthesizer
from verbalizer import WolofTTSVerbalizer


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_corpus(corpus_file: str, separator: str) -> pd.DataFrame:
    """Loads the KallaamaBench test corpus (filename|transcription|speaker|gender)."""
    corpus = pd.read_csv(corpus_file, sep=separator, header=0, on_bad_lines="skip", dtype=str)
    required_cols = {"filename", "transcription", "speaker", "gender"}
    missing = required_cols - set(corpus.columns)
    if missing:
        raise ValueError(f"corpus_file is missing required column(s): {missing}")
    return corpus


def save_submission_audio(audio, native_sampling_rate, output_sampling_rate, output_path):
    """Resamples generated audio to the submission sample rate and writes it as
    16-bit PCM mono WAV, per SUBMISSION_FORMAT.md."""
    waveform = torch.as_tensor(np.asarray(audio), dtype=torch.float32).reshape(1, -1)
    if native_sampling_rate != output_sampling_rate:
        # On many-core machines the default thread count makes this CPU resample
        # orders of magnitude slower; cap it here (this runs outside RTF timing).
        num_threads = torch.get_num_threads()
        torch.set_num_threads(min(num_threads, 8))
        waveform = torchaudio.functional.resample(waveform, native_sampling_rate, output_sampling_rate)
        torch.set_num_threads(num_threads)
    sf.write(
        output_path,
        waveform.squeeze(0).cpu().numpy(),
        output_sampling_rate,
        subtype="PCM_16",
    )
    return waveform.shape[1] / output_sampling_rate


def load_model(model_repo, checkpoint_dir, token):
    snapshot_download(repo_id=model_repo, local_dir=checkpoint_dir, token=token)
    return Synthesizer(
        tts_checkpoint=os.path.join(checkpoint_dir, "model.pth"),
        tts_config_path=os.path.join(checkpoint_dir, "config.json"),
        use_cuda=DEVICE == "cuda",
    )


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kiriku-TTS inference script")
    parser.add_argument("--config-path", required=True, help="Path to config.yaml")
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────────────────
    cfg = load_config(args.config_path)

    token = os.getenv("HF_TOKEN")

    # Corpus
    corpus_file      = cfg["inference"]["corpus_file"]
    corpus_separator = cfg["inference"].get("corpus_separator", "|")

    # Submission output
    output_dir           = cfg["inference"]["output_dir"]
    output_audio_dir     = os.path.join(output_dir, "audio")
    output_sampling_rate = cfg["inference"]["output_sampling_rate"]
    num_sentences        = cfg["inference"].get("num_sentences")

    # Model
    model_cfg      = cfg["model"]
    model_repo     = model_cfg["repo"]
    checkpoint_dir = model_cfg["checkpoint_dir"]

    # Submission metadata
    sub_cfg = cfg.get("submission", {})
    # ─────────────────────────────────────────────────────────────────────────────

    os.makedirs(output_audio_dir, exist_ok=True)

    # Load the KallaamaBench test corpus
    corpus = load_corpus(corpus_file, corpus_separator)
    if num_sentences is not None:
        corpus = corpus.iloc[:num_sentences]
    print(f"Loaded {len(corpus)} rows from corpus.")

    # Initialize model
    synthesizer = load_model(model_repo, checkpoint_dir, token)
    synthesis_sampling_rate = synthesizer.output_sample_rate
    print(f"Model loaded successfully! Sampling rate: {synthesis_sampling_rate}")

    # Initialize verbalizer
    verbalizer = WolofTTSVerbalizer()

    # Warm up the model
    for _ in range(3):
        _ = synthesizer.tts(text=verbalizer.verbalize("salamalekum, na nga def"))
    sync()
    print("Model warm-up complete.")

    # Generate and save audio for each corpus row
    rtf_values = []
    n_empty = 0
    for _, row in tqdm(corpus.iterrows(), total=len(corpus), desc="Generating audio"):
        filename = row["filename"]
        transcription = row["transcription"]

        sync()
        start_time = time.time()
        sentence = verbalizer.verbalize(transcription)
        audio = synthesizer.tts(text=sentence)
        sync()
        inference_time = time.time() - start_time

        output_path = os.path.join(output_audio_dir, filename)
        if len(audio) == 0:
            # Keep one file per corpus row: write 0.1 s of silence and exclude it from RTF.
            print(f"Warning: no audio generated for {filename}; writing silence.")
            save_submission_audio(np.zeros(output_sampling_rate // 10), output_sampling_rate,
                                  output_sampling_rate, output_path)
            n_empty += 1
            continue

        audio_duration = save_submission_audio(
            audio, synthesis_sampling_rate, output_sampling_rate, output_path
        )
        rtf = inference_time / audio_duration if audio_duration > 0 else float("inf")
        rtf_values.append(rtf)

    avg_rtf = sum(rtf_values) / len(rtf_values) if rtf_values else 0.0
    print(f"Audio files saved to: {output_audio_dir}")
    print(f"Average RTF: {avg_rtf:.3f}")
    if n_empty:
        print(f"Warning: {n_empty} row(s) produced no audio and were written as silence.")

    # Write metadata.json, per SUBMISSION_FORMAT.md
    metadata = {
        "team": sub_cfg.get("team", "Kallaama-Bench Team"),
        "system": sub_cfg.get("system", model_cfg["name"]),
        "submission_date": time.strftime("%Y-%m-%d"),
        "contact": sub_cfg.get("contact", ""),
        "rtf": round(avg_rtf, 3),
        "notes": sub_cfg.get("notes", ""),
    }
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    print(f"metadata.json written to: {metadata_path}")
    print(f"Submission ready: {output_dir} ({len(corpus)} audio files)")
