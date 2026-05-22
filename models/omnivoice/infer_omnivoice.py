"""
Requirements
------------
pip install omnivoice

Usage
-----
python infer_omnivoice.py

Text transcriptions of reference audio files ??
"""

import os
import json
import time
import torch
import torchaudio
import pandas as pd
from omnivoice import OmniVoice
from verbalizer import WolofTTSVerbalizer

TOKEN = os.getenv("HF_TOKEN")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CORPUS_FILE          = "/root/kallaama/data/corpus.csv"
REFERENCE_AUDIO_FOLDER = "/root/kallaama/data/references"
OUTPUT_DIR           = "/root/kallaama/submissions/omnivoice"
OUTPUT_AUDIO_DIR     = os.path.join(OUTPUT_DIR, "audio")
OUTPUT_METADATA      = os.path.join(OUTPUT_DIR, "metadata.json")

MODEL_REPO           = "k2-fsa/OmniVoice"
OUTPUT_SAMPLING_RATE = 24000

# Team information written to metadata.json
METADATA = {
    "team"            : "K2-FSA",
    "system"          : "Omnivoice",
    "submission_date" : "2026-05-19",
    "contact"         : "contact@example.com",
    "notes"           : "RTF measured on a single NVIDIA A100 40 GB",
}


if __name__ == "__main__":
    login(TOKEN)
    os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)

    # Load corpus
    corpus = pd.read_csv(CORPUS_FILE, sep="|", header=0, on_bad_lines="skip", dtype=str)
    corpus = corpus.dropna(subset=["filename", "transcription"]).reset_index(drop=True)
    print(f"Loaded {len(corpus)} sentences from corpus.")

    # Initialize model
    model = OmniVoice.from_pretrained(
        MODEL_REPO,
        device_map=DEVICE,
        dtype=torch.float16
    )

    # Initialize verbalizer
    verbalizer = WolofTTSVerbalizer()

    # Generate audio and compute RTF
    rtf_values = []
    for _, row in tqdm(corpus.iterrows(), total=len(corpus), desc="Generating audio"):
        sentence = row["transcription"].strip()
        speaker = row["speaker"].strip()

        if not sentence:
            continue

        start_time      = time.time()

        sentence        = verbalizer.verbalize(sentence)
        reference_audio_file = os.path.join(REFERENCE_AUDIO_FOLDER, f"reference_{speaker}.wav")
        audio = model.generate(
            text=sentence,
            ref_audio=reference_audio_file,
            ref_text="Transcription of the reference audio.",
        )
        inference_time  = time.time() - start_time
        audio_duration  = len(audio[0]) / OUTPUT_SAMPLING_RATE
        rtf             = inference_time / audio_duration if audio_duration > 0 else float("inf")
        rtf_values.append(rtf)

        torchaudio.save(os.path.join(OUTPUT_AUDIO_DIR, row["filename"]), audio[0], OUTPUT_SAMPLING_RATE)

    avg_rtf = sum(rtf_values) / len(rtf_values) if rtf_values else 0.0
    print(f"Audio files saved to: {OUTPUT_AUDIO_DIR}")
    print(f"Average RTF: {avg_rtf:.3f}")

    # Save team info + RTF to metadata.json
    metadata = {**METADATA, "rtf": round(avg_rtf, 4)}
    with open(OUTPUT_METADATA, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to: {OUTPUT_METADATA}")