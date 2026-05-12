# pip install git+https://github.com/huggingface/parler-tts.git
# pip install torchcodec
# apt-get update && apt-get install -y ffmpeg

import os
import json
import time
import torch
import torchaudio
import soundfile as sf
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
from tqdm import tqdm
from verbalizer import WolofTTSVerbalizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CORPUS_TYPE = "monolingual"
CORPUS_FILE = "/root/wolof-tts/evaluation/data/corpus_monolingual.txt"
OUTPUT_METRICS_FILE = "/root/wolof-tts/evaluation/metrics/metrics_rtf_all.json"

MODEL_NAME = "adia-tts"
MODEL_REPO = "CONCREE/Adia_TTS"
OUTPUT_AUDIO_DIR = f"/root/wolof-tts/evaluation/outputs/{CORPUS_TYPE}/{MODEL_NAME}"

# Description used to guide the voice style
VOICE_DESCRIPTION = "A clear and educational voice, with a flow adapted to learning"

GENERATION_CONFIG = dict(
    temperature=0.8,
    max_new_tokens=1000,
    do_sample=True,
    top_k=50,
    repetition_penalty=1.2,
)


def generate_audio(text, model, tokenizer, description):
    input_ids     = tokenizer(description, return_tensors="pt").input_ids.to(DEVICE)
    prompt_ids    = tokenizer(text, return_tensors="pt").input_ids.to(DEVICE)
    audio_values  = model.generate(
        input_ids=input_ids,
        prompt_input_ids=prompt_ids,
        **GENERATION_CONFIG,
    )
    return audio_values.cpu().squeeze()


if __name__ == "__main__":
    # Ensure the destination folder exists
    os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)

    # Read sentences from the corpus file
    with open(CORPUS_FILE, "r", encoding="utf-8") as file:
        sentences = file.readlines()
        print("Sentences loaded from corpus.")

    # Load model and tokenizer
    model     = ParlerTTSForConditionalGeneration.from_pretrained(MODEL_REPO).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
    print("Model loaded successfully!")

    # Initialize verbalizer
    verbalizer = WolofTTSVerbalizer()

    # Warm up the model
    for i in range(3):
        warmup_sentence = verbalizer.verbalize("This is a warm-up sentence.")
        _ = generate_audio(warmup_sentence, model, tokenizer, VOICE_DESCRIPTION)
    print("Model warm-up complete.")

    # Generate and save audio for each sentence
    rtf_values = []
    for i, sentence in enumerate(tqdm(sentences, desc="Generating audio")):
        sentence = sentence.strip()
        if not sentence:
            continue

        # Generate audio
        start_time = time.time()
        sentence   = verbalizer.verbalize(sentence)
        audio      = generate_audio(sentence, model, tokenizer, VOICE_DESCRIPTION)
        waveform   = audio.unsqueeze(0)
        inference_time = time.time() - start_time
        audio_duration = waveform.shape[1] / model.config.sampling_rate
        rtf = inference_time / audio_duration if audio_duration > 0 else float("inf")
        rtf_values.append(rtf)
        print(f"Sentence {i + 1}: Duration = {audio_duration:.2f}s, Inference Time = {inference_time:.2f}s, RTF = {rtf:.2f}")

        # Save the audio to the destination folder
        output_path = os.path.join(OUTPUT_AUDIO_DIR, f"sentence_{i + 1:02}.wav")
        torchaudio.save(output_path, waveform, model.config.sampling_rate)

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