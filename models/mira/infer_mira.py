import os
import json
import time
import torch
import torchaudio
import pandas as pd
from unsloth import FastModel
from huggingface_hub import login
from tqdm import tqdm
from ncodec.codec import TTSCodec
from verbalizer import WolofTTSVerbalizer

TOKEN = os.getenv("HF_TOKEN")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CORPUS_FILE          = "/root/kallaama/data/corpus.csv"
REFERENCE_AUDIO_FOLDER = "/root/kallaama/data/references"
OUTPUT_DIR           = "/root/kallaama/outputs/mira-tts"
OUTPUT_AUDIO_DIR     = os.path.join(OUTPUT_DIR, "audio")
OUTPUT_METADATA      = os.path.join(OUTPUT_DIR, "metadata.json")

MODEL_REPO           = "YatharthS/MiraTTS"
OUTPUT_SAMPLING_RATE = 48000

# Team information written to metadata.json
METADATA = {
    "team"            : "YatharthS",
    "system"          : "MiraTTS",
    "submission_date" : "2026-05-19",
    "contact"         : "contact@example.com",
    "notes"           : "RTF measured on a single NVIDIA A100 40 GB",
}


def generate_audio(prompt, context_tokens, pipeline, gen_config, tts_codec):
    response = pipeline([prompt], gen_config=gen_config, do_preprocess=False)
    return tts_codec.decode(response[0].text, context_tokens)


if __name__ == "__main__":
    login(TOKEN)
    os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)

    # Load corpus
    corpus = pd.read_csv(CORPUS_FILE, sep="|", header=0, on_bad_lines="skip", dtype=str)
    corpus = corpus.dropna(subset=["filename", "transcription"]).reset_index(drop=True)
    print(f"Loaded {len(corpus)} sentences from corpus.")

    # Initialize inference pipeline
    from lmdeploy import pipeline, GenerationConfig, PytorchEngineConfig
    backend_config    = PytorchEngineConfig()
    generation_config = GenerationConfig(top_p=0.95, top_k=50, temperature=0.8,
                                          max_new_tokens=1024, repetition_penalty=1.2,
                                          min_p=0.05, do_sample=True)
    inference_pipeline = pipeline(MODEL_REPO, backend_config=backend_config)

    # Initialize verbalizer
    verbalizer = WolofTTSVerbalizer()

    # Intialize TTSCodec
    tts_codec      = TTSCodec()

    # Generate audio and compute RTF
    rtf_values = []
    for _, row in tqdm(corpus.iterrows(), total=len(corpus), desc="Generating audio"):
        sentence = row["transcription"].strip()
        speaker = row["speaker"].strip()

        if not sentence:
            continue

        start_time      = time.time()

        reference_audio_file = os.path.join(REFERENCE_AUDIO_FOLDER, f"reference_{speaker}.wav")
        context_tokens = tts_codec.encode(reference_audio_file)

        sentence        = verbalizer.verbalize(sentence)
        formatted_prompt = tts_codec.format_prompt(sentence, context_tokens, None)
        audio           = generate_audio(formatted_prompt, context_tokens,
                                         inference_pipeline, generation_config, tts_codec)
        waveform        = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        inference_time  = time.time() - start_time
        audio_duration  = waveform.shape[1] / OUTPUT_SAMPLING_RATE
        rtf             = inference_time / audio_duration if audio_duration > 0 else float("inf")
        rtf_values.append(rtf)

        torchaudio.save(os.path.join(OUTPUT_AUDIO_DIR, row["filename"]), waveform, OUTPUT_SAMPLING_RATE)

    avg_rtf = sum(rtf_values) / len(rtf_values) if rtf_values else 0.0
    print(f"Audio files saved to: {OUTPUT_AUDIO_DIR}")
    print(f"Average RTF: {avg_rtf:.3f}")

    # Save team info + RTF to metadata.json
    metadata = {**METADATA, "rtf": round(avg_rtf, 4)}
    with open(OUTPUT_METADATA, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to: {OUTPUT_METADATA}")