"""
Requirements
------------
pip install --upgrade omnivoice

Usage
-----
python infer_omni.py --config-path configs/infer_omni_ft.yaml

Produces a KallaamaBench submission folder:
    <output_dir>/
    ├── metadata.json
    └── audio/
        ├── <filename from corpus.csv row 1>
        ├── <filename from corpus.csv row 2>
        └── ...
"""

import argparse
import json
import os
import time

import pandas as pd
import torch
import torchaudio
import yaml
from tqdm import tqdm
import soundfile as sf
from huggingface_hub import login
from omnivoice import OmniVoice
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


def resolve_reference_audio_path(speaker: str, reference_audio_dir: str) -> str:
    """Resolves the reference/prompt audio for a given speaker."""
    speaker = speaker.strip()
    return os.path.join(reference_audio_dir, f"reference_{speaker}.wav")


def load_reference_transcriptions(reference_transcriptions_file: str) -> dict:
    """Loads reference transcriptions from a JSON file shaped as a list of
    {"speaker", "reference", "transcription"} records, keyed by speaker id."""
    with open(reference_transcriptions_file, "r", encoding="utf-8") as f:
        records = json.load(f)
    return {record["speaker"].strip(): record["transcription"] for record in records}


def resolve_reference_transcription(speaker: str, reference_transcriptions: dict) -> str:
    """Resolves the reference transcription for a given speaker."""
    speaker = speaker.strip()
    return reference_transcriptions[speaker]


def save_submission_audio(audio, native_sampling_rate, output_sampling_rate, output_path):
    """Resamples generated audio to the submission sample rate and writes it as
    16-bit PCM mono WAV, per SUBMISSION_FORMAT.md."""
    waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    if native_sampling_rate != output_sampling_rate:
        waveform = torchaudio.functional.resample(waveform, native_sampling_rate, output_sampling_rate)
    sf.write(
        output_path,
        waveform.squeeze(0).cpu().numpy(),
        output_sampling_rate,
        subtype="PCM_16",
    )
    return waveform.shape[1] / output_sampling_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OmniVoice inference script")
    parser.add_argument("--config-path", required=True, help="Path to config.yaml")
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────────────────
    cfg = load_config(args.config_path)

    token = os.getenv("HF_TOKEN")

    # Corpus / reference audio
    corpus_file                  = cfg["inference"]["corpus_file"]
    corpus_separator             = cfg["inference"].get("corpus_separator", "|")
    reference_audio_dir          = cfg["inference"]["reference_audio_dir"]
    reference_transcriptions_file = cfg["inference"]["reference_transcriptions_file"]

    # Submission output
    output_dir              = cfg["inference"]["output_dir"]
    output_audio_dir        = os.path.join(output_dir, "audio")
    synthesis_sampling_rate = cfg["inference"]["synthesis_sampling_rate"]
    output_sampling_rate    = cfg["inference"]["output_sampling_rate"]
    num_sentences           = cfg["inference"].get("num_sentences")

    # Model
    model_cfg   = cfg["model"]
    model_repo  = model_cfg["repo"]
    model_dtype = getattr(torch, model_cfg["dtype"])

    # Submission metadata
    sub_cfg = cfg.get("submission", {})
    # ─────────────────────────────────────────────────────────────────────────────

    login(token)
    os.makedirs(output_audio_dir, exist_ok=True)

    # Load the KallaamaBench test corpus
    corpus = load_corpus(corpus_file, corpus_separator)
    if num_sentences is not None:
        corpus = corpus.iloc[:num_sentences]
    print(f"Loaded {len(corpus)} rows from corpus.")

    # Initialize model
    model = OmniVoice.from_pretrained(
        model_repo,
        device_map=DEVICE,
        dtype=model_dtype,
    )

    # Initialize verbalizer
    verbalizer = WolofTTSVerbalizer()

    # Load reference transcriptions once from the JSON file
    reference_transcriptions = load_reference_transcriptions(reference_transcriptions_file)

    # Resolve and cache (ref_audio_path, ref_text) per speaker so each unique
    # reference is only resolved once.
    reference_cache = {}

    def get_reference(speaker):
        speaker = speaker.strip()
        if speaker not in reference_cache:
            ref_audio_path = resolve_reference_audio_path(speaker, reference_audio_dir)
            ref_text = resolve_reference_transcription(speaker, reference_transcriptions)
            reference_cache[speaker] = (ref_audio_path, ref_text)
        return reference_cache[speaker]

    # Generate and save audio for each corpus row
    rtf_values = []
    for _, row in tqdm(corpus.iterrows(), total=len(corpus), desc="Generating audio"):
        filename = row["filename"]
        transcription = row["transcription"]
        speaker = row["speaker"].strip()

        ref_audio_path, ref_text = get_reference(speaker)

        start_time = time.time()
        sentence = verbalizer.verbalize(transcription)
        audio = model.generate(
            text=sentence,
            ref_audio=ref_audio_path,
            ref_text=ref_text,
        )
        inference_time = time.time() - start_time

        output_path = os.path.join(output_audio_dir, filename)
        audio_duration = save_submission_audio(
            audio[0], synthesis_sampling_rate, output_sampling_rate, output_path
        )
        rtf = inference_time / audio_duration if audio_duration > 0 else float("inf")
        rtf_values.append(rtf)

    avg_rtf = sum(rtf_values) / len(rtf_values) if rtf_values else 0.0
    print(f"Audio files saved to: {output_audio_dir}")
    print(f"Average RTF: {avg_rtf:.3f}")

    # Write metadata.json, per SUBMISSION_FORMAT.md
    metadata = {
        "team": sub_cfg.get("team", "Baamtu"),
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
    print(f"Submission ready: {output_dir} ({len(rtf_values)} audio files)")