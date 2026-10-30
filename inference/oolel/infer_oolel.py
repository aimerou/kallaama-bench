# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch==2.8.0",
#     "torchaudio==2.8.0",
#     "torchcodec==0.7.0",
#     "librosa>=0.10.2",
#     "transformers==4.46.3",
#     "diffusers==0.29.0",
#     "conformer==0.3.2",
#     "numpy>=2.0.0",
#     "s3tokenizer",
#     "num2words",
#     "pandas",
#     "tqdm",
#     "huggingface_hub",
# ]
# ///
"""
Oolel TTS Inference Script
==========================

Requirements
------------
Dependencies are declared in the inline script metadata above; `uv run` installs them
in an isolated environment. ffmpeg must be installed on the system:
apt update && apt install -y ffmpeg

Usage
-----
uv run infer_oolel.py
"""

import os
import json
import time
import torch
import torchaudio
import pandas as pd
from transformers import AutoModel
from huggingface_hub import snapshot_download
from tqdm import tqdm
from verbalizer import WolofTTSVerbalizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CORPUS_FILE            = "/root/kallaama-bench/data/corpus.csv"
REFERENCE_AUDIO_FOLDER = "/root/kallaama-bench/data/references/audio"
OUTPUT_DIR             = "/root/kallaama-bench/submissions/oolel"
OUTPUT_AUDIO_DIR       = os.path.join(OUTPUT_DIR, "audio")
OUTPUT_METADATA        = os.path.join(OUTPUT_DIR, "metadata.json")

MODEL_REPO = "soynade-research/Oolel-Voices"

# Team information written to metadata.json
METADATA = {
    "team"            : "Kallaama-Bench Team",
    "system"          : "Oolel Voices",
    "submission_date" : "2026-05-19",
    "contact"         : "contact@example.com",
    "notes"           : "RTF measured on a single NVIDIA A100 40 GB",
}

# Voice expressivity parameters
CFG_WEIGHT   = 0.5
EXAGGERATION = 0.2
TEMPERATURE  = 0.3


def generate_audio(text, model, reference_audio_path):
    wav = model.generate(
        text,
        audio_prompt_path=reference_audio_path,
        cfg_weight=CFG_WEIGHT,
        exaggeration=EXAGGERATION,
        temperature=TEMPERATURE,
    )
    return wav


if __name__ == "__main__":
    os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)

    # Load corpus
    corpus = pd.read_csv(CORPUS_FILE, sep="|", header=0, on_bad_lines="skip", dtype=str)
    corpus = corpus.dropna(subset=["filename", "transcription"]).reset_index(drop=True)
    print(f"Loaded {len(corpus)} sentences from corpus.")

    # Load model
    model_path = snapshot_download(repo_id=MODEL_REPO)
    model      = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.to(DEVICE)
    output_sampling_rate = model.sr
    print("Model loaded successfully!")
    print(f"Model sampling rate: {output_sampling_rate}")

    # Initialize verbalizer
    verbalizer = WolofTTSVerbalizer()

    # Warm up the model
    for _ in range(3):
        reference_audio_file = os.path.join(REFERENCE_AUDIO_FOLDER, "reference_afy.wav")
        _ = generate_audio(verbalizer.verbalize("This is a warm-up sentence."),
                           model, reference_audio_file)
    print("Model warm-up complete.")

    # Generate audio and compute RTF
    rtf_values = []
    for _, row in tqdm(corpus.iterrows(), total=len(corpus), desc="Generating audio"):
        sentence = row["transcription"].strip()
        speaker = row["speaker"].strip()

        reference_audio_file = os.path.join(REFERENCE_AUDIO_FOLDER, f"reference_{speaker}.wav")

        if not sentence:
            continue

        start_time     = time.time()
        wav            = generate_audio(verbalizer.verbalize(sentence), model, reference_audio_file)
        waveform       = wav if wav.dim() == 2 else wav.unsqueeze(0)
        inference_time = time.time() - start_time
        audio_duration = waveform.shape[1] / output_sampling_rate
        rtf            = inference_time / audio_duration if audio_duration > 0 else float("inf")
        rtf_values.append(rtf)

        torchaudio.save(os.path.join(OUTPUT_AUDIO_DIR, row["filename"]), waveform, output_sampling_rate)

    avg_rtf = sum(rtf_values) / len(rtf_values) if rtf_values else 0.0
    print(f"Audio files saved to: {OUTPUT_AUDIO_DIR}")
    print(f"Average RTF: {avg_rtf:.3f}")

    # Save team info + RTF to metadata.json
    metadata = {**METADATA, "rtf": round(avg_rtf, 4)}
    with open(OUTPUT_METADATA, "w") as f:
        json.dump(metadata, f, indent=4)