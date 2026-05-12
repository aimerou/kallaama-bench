# pip install librosa>=0.10.2 transformers==4.46.3 diffusers==0.29.0 conformer==0.3.2 numpy>=2.0.0 torchcodec s3tokenizer num2words omegaconf

import os
import json
import time
import torch
import torchaudio
from transformers import AutoModel
from huggingface_hub import snapshot_download
from tqdm import tqdm
from verbalizer import WolofTTSVerbalizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CORPUS_TYPE = "monolingual"
CORPUS_FILE = "/root/wolof-tts/evaluation/data/corpus_monolingual.txt"
REFERENCE_AUDIO_FILE = "/root/wolof-tts/evaluation/data/reference-female.wav"
OUTPUT_METRICS_FILE = "/root/wolof-tts/evaluation/metrics/metrics_rtf_all.json"

MODEL_NAME = "oolel-voices"
MODEL_REPO = "soynade-research/Oolel-Voices"
OUTPUT_AUDIO_DIR = f"/root/wolof-tts/evaluation/outputs/{CORPUS_TYPE}/{MODEL_NAME}"

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
    # Ensure the destination folder exists
    os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)

    # Read sentences from the corpus file
    with open(CORPUS_FILE, "r", encoding="utf-8") as file:
        sentences = file.readlines()
        print("Sentences loaded from corpus.")

    # Load model
    model_path = snapshot_download(repo_id=MODEL_REPO)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.to(DEVICE)
    output_sampling_rate = model.sr
    print("Model loaded successfully!")

    # Initialize verbalizer
    verbalizer = WolofTTSVerbalizer()

    # Warm up the model
    for i in range(3):
        warmup_sentence = verbalizer.verbalize("This is a warm-up sentence.")
        _ = generate_audio(warmup_sentence, model, REFERENCE_AUDIO_FILE)
    print("Model warm-up complete.")

    # Generate and save audio for each sentence
    rtf_values = []
    for i, sentence in enumerate(tqdm(sentences, desc="Generating audio")):
        sentence = sentence.strip()
        if not sentence:
            continue

        # Generate audio
        start_time = time.time()
        sentence  = verbalizer.verbalize(sentence)
        wav       = generate_audio(sentence, model, REFERENCE_AUDIO_FILE)
        waveform  = wav if wav.dim() == 2 else wav.unsqueeze(0)
        inference_time = time.time() - start_time
        audio_duration = waveform.shape[1] / output_sampling_rate
        rtf = inference_time / audio_duration if audio_duration > 0 else float("inf")
        rtf_values.append(rtf)
        print(f"Sentence {i + 1}: Duration = {audio_duration:.2f}s, Inference Time = {inference_time:.2f}s, RTF = {rtf:.2f}")

        # Save the audio to the destination folder
        output_path = os.path.join(OUTPUT_AUDIO_DIR, f"sentence_{i + 1:02}.wav")
        torchaudio.save(output_path, waveform, output_sampling_rate)

    print(f"Audio files generated and saved to: {OUTPUT_AUDIO_DIR}")

    avg_rtf = sum(rtf_values) / len(rtf_values) if rtf_values else 0
    print(f"Average realtime factor (RTF) across all sentences: {avg_rtf:.2f}")

    # Save RTF values to the JSON file
    if os.path.exists(OUTPUT_METRICS_FILE):
        with open(OUTPUT_METRICS_FILE, "r") as json_file:
            metrics_data = json.load(json_file)
    else:
        metrics_data = []
    metrics_data.append({"model_name": MODEL_NAME, "corpus": CORPUS_TYPE, "rtf": avg_rtf})
    with open(OUTPUT_METRICS_FILE, "w") as json_file:
        json.dump(metrics_data, json_file, indent=4)
    print(f"RTF values saved to: {OUTPUT_METRICS_FILE}")