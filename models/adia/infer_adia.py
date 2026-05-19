"""
Adia TTS Inference Script
=========================

Requirements
------------
pip install git+https://github.com/huggingface/parler-tts.git
pip install torch torchaudio transformers
apt-get update && apt-get install -y ffmpeg

Usage
-----
python infer_adia.py
"""

import os
import json
import time
import torch
import torchaudio
import pandas as pd
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
from tqdm import tqdm
from verbalizer import WolofTTSVerbalizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CORPUS_FILE      = "/root/kallaama/data/corpus.csv"
OUTPUT_DIR       = "/root/kallaama/submissions/adia"
OUTPUT_AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")
OUTPUT_METADATA  = os.path.join(OUTPUT_DIR, "metadata.json")

MODEL_REPO = "CONCREE/Adia_TTS"

# Team information written to metadata.json
METADATA = {
    "team"            : "CONCREE",
    "system"          : "Adia TTS (Parler-TTS fine-tune)",
    "submission_date" : "2026-05-19",
    "contact"         : "contact@example.com",
    "acoustic_model"  : "Parler-TTS",
    "notes"           : "RTF measured on a single NVIDIA A100 40 GB",
}

# Description used to guide the voice style
DEFAULT_VOICE_DESCRIPTION = "A clear and educational voice, with a flow adapted to learning"
MALE_VOICE_DESCRIPTION = "A clear and educational male voice, with a flow adapted to learning"
FEMALE_VOICE_DESCRIPTION = "A clear and educational female voice, with a flow adapted to learning"

GENERATION_CONFIG = dict(
    temperature=0.8,
    max_new_tokens=1000,
    do_sample=True,
    top_k=50,
    repetition_penalty=1.2,
)


def generate_audio(text, model, tokenizer, description):
    input_ids    = tokenizer(description, return_tensors="pt").input_ids.to(DEVICE)
    prompt_ids   = tokenizer(text, return_tensors="pt").input_ids.to(DEVICE)
    audio_values = model.generate(
        input_ids=input_ids,
        prompt_input_ids=prompt_ids,
        **GENERATION_CONFIG,
    )
    return audio_values.cpu().squeeze()


if __name__ == "__main__":
    os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)

    # Load corpus
    corpus = pd.read_csv(CORPUS_FILE, sep="|", header=0, on_bad_lines="skip", dtype=str)
    corpus = corpus.dropna(subset=["filename", "transcription"]).reset_index(drop=True)
    print(f"Loaded {len(corpus)} sentences from corpus.")

    # Load model and tokenizer
    model     = ParlerTTSForConditionalGeneration.from_pretrained(MODEL_REPO).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
    print("Model loaded successfully!")

    # Initialize verbalizer
    verbalizer = WolofTTSVerbalizer()

    # Warm up the model
    for _ in range(3):
        _ = generate_audio(verbalizer.verbalize("This is a warm-up sentence."),
                           model, tokenizer, DEFAULT_VOICE_DESCRIPTION)
    print("Model warm-up complete.")

    # Generate audio and compute RTF
    rtf_values = []
    for _, row in tqdm(corpus.iterrows(), total=len(corpus), desc="Generating audio"):
        sentence = row["transcription"].strip()
        speaker = row["speaker"].strip()
        gender = int(row["gender"])

        if gender == 1: voice_description = MALE_VOICE_DESCRIPTION
        elif gender == 0: voice_description = FEMALE_VOICE_DESCRIPTION
        else: voice_description = DEFAULT_VOICE_DESCRIPTION

        if not sentence:
            continue

        start_time     = time.time()
        audio          = generate_audio(verbalizer.verbalize(sentence), model, tokenizer, voice_description)
        waveform       = audio.unsqueeze(0)
        inference_time = time.time() - start_time
        audio_duration = waveform.shape[1] / model.config.sampling_rate
        rtf            = inference_time / audio_duration if audio_duration > 0 else float("inf")
        rtf_values.append(rtf)

        torchaudio.save(os.path.join(OUTPUT_AUDIO_DIR, row["filename"]), waveform, model.config.sampling_rate)

    avg_rtf = sum(rtf_values) / len(rtf_values) if rtf_values else 0.0
    print(f"Audio files saved to: {OUTPUT_AUDIO_DIR}")
    print(f"Average RTF: {avg_rtf:.3f}")

    # Save team info + RTF to metadata.json
    metadata = {**METADATA, "rtf": round(avg_rtf, 4)}
    with open(OUTPUT_METADATA, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to: {OUTPUT_METADATA}")