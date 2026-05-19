import os
import json
import time
import librosa
import torch
import torchaudio
import numpy as np
import pandas as pd
from unsloth import FastModel
from huggingface_hub import login
from tqdm import tqdm
from ncodec.codec import TTSCodec
from ncodec.encoder.model import audio_volume_normalize
from lmdeploy import pipeline, GenerationConfig, PytorchEngineConfig
from verbalizer import WolofTTSVerbalizer

TOKEN = os.getenv("HF_TOKEN")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CORPUS_TYPE = "code-mixed"
CORPUS_FILE = "/root/wolof-tts/evaluation/data/corpus_mixed.txt"
REFERENCE_AUDIO_FILE = "/root/wolof-tts/evaluation/data/reference-female.wav"
OUTPUT_METRICS_FILE = "/root/wolof-tts/evaluation/metrics/metrics_rtf_all.json"

MODEL_NAME = "mira-tts"
MODEL_REPO = "YatharthS/MiraTTS"
OUTPUT_AUDIO_DIR = f"/root/wolof-tts/evaluation/outputs/{CORPUS_TYPE}/{MODEL_NAME}"
OUTPUT_SAMPLING_RATE = 48000

def generate_audio(prompt, context_tokens, pipeline, gen_config, tts_codec):
    response = pipeline([prompt], gen_config=gen_config, do_preprocess=False)
    audio = tts_codec.decode(response[0].text, context_tokens)
    return audio

if __name__ == "__main__":
    # Login to Hugging Face
    login(TOKEN)

    # Ensure the destination folder exists
    os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)

    # Read sentences from the corpus file
    with open(CORPUS_FILE, "r", encoding="utf-8") as file:
        sentences = file.readlines()
        print("Sentences loaded from corpus.")

    # Initialize inference pipeline
    backend_config = PytorchEngineConfig()
    generation_config = GenerationConfig(top_p=0.95, top_k=50, temperature=0.8, max_new_tokens=1024, repetition_penalty=1.2, min_p=0.05, do_sample=True)
    inference_pipeline = pipeline(MODEL_REPO, backend_config=backend_config)

    # Initialize verbalizer
    verbalizer = WolofTTSVerbalizer()

    # Encode the reference audio
    tts_codec = TTSCodec()
    context_tokens = tts_codec.encode(REFERENCE_AUDIO_FILE)
    print("Reference audio encoded.")

    # Generate a first time to warm up the model
    for i in range(3):
        warmup_sentence = "This is a warm-up sentence."
        warmup_sentence = verbalizer.verbalize(warmup_sentence)
        warmup_formatted_prompt = tts_codec.format_prompt(warmup_sentence, context_tokens, None)
        _ = generate_audio(warmup_formatted_prompt, context_tokens, inference_pipeline, generation_config, tts_codec)
    print("Model warm-up complete.")

    # Generate and save audio for each sentence
    rtf_values = []
    for i, sentence in enumerate(tqdm(sentences, desc="Generating audio")):
        sentence = sentence.strip()  # Remove leading/trailing whitespace
        if not sentence:  # Skip empty lines
            continue

        # Generate audio
        start_time = time.time()
        sentence = verbalizer.verbalize(sentence)
        formatted_prompt = tts_codec.format_prompt(sentence, context_tokens, None)
        audio = generate_audio(formatted_prompt, context_tokens, inference_pipeline, generation_config, tts_codec)
        waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        inference_time = time.time() - start_time
        audio_duration = waveform.shape[1] / OUTPUT_SAMPLING_RATE
        rtf = inference_time / audio_duration if audio_duration > 0 else float('inf')
        rtf_values.append(rtf)
        print(f"Sentence {i + 1}: Duration = {audio_duration:.2f}s, Inference Time = {inference_time:.2f}s, RTF = {rtf:.2f}")

        # Save the audio to the destination folder
        output_path = os.path.join(OUTPUT_AUDIO_DIR, f"sentence_{i + 1:02}.wav")
        torchaudio.save(output_path, waveform, OUTPUT_SAMPLING_RATE)

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