# Download model: gdown 1jsGAMBo354uRhwVKNnuJpLtq2uJYcVHN
# Unzip model: unzip galsenai-xtts-wo-checkpoints
# Clone repository: git clone https://github.com/nguyenhoanganh2002/XTTSv2-Finetuning-for-New-Languages.git
# Install dependencies: cd XTTSv2-Finetuning-for-New-Languages
#                       pip install -r requirements.txt
#                       pip install TTS

import os
import json
import time
import torch
import torchaudio
import soundfile as sf
import numpy as np
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from tqdm import tqdm
from underthesea import sent_tokenize
from verbalizer import WolofTTSVerbalizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CORPUS_TYPE = "code-mixed"
CORPUS_FILE = "/root/wolof-tts/evaluation/data/corpus_mixed.txt"
REFERENCE_AUDIO_FILE = "/root/wolof-tts/evaluation/data/reference-female.wav"
OUTPUT_METRICS_FILE = "/root/wolof-tts/evaluation/metrics/metrics_rtf_all.json"

MODEL_NAME = "galsenai"
CHECKPOINTS_ROOT = "/root/wolof-tts/models/galsenai-xtts-wo-checkpoints/"
CHECKPOINT_DIR   = os.path.join(CHECKPOINTS_ROOT, "Anta_GPT_XTTS_Wo")
MODEL_FILE       = "best_model_89250.pth"
VOCAB_FILE       = os.path.join(CHECKPOINTS_ROOT, "XTTS_v2.0_original_model_files/vocab.json")
OUTPUT_AUDIO_DIR = f"/root/wolof-tts/evaluation/outputs/{CORPUS_TYPE}/{MODEL_NAME}"
OUTPUT_SAMPLING_RATE = 24000


def load_model():
    xtts_checkpoint = os.path.join(CHECKPOINT_DIR, MODEL_FILE)
    xtts_config = os.path.join(CHECKPOINT_DIR, "config.json")

    config = XttsConfig()
    config.load_json(xtts_config)
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_path=xtts_checkpoint,
        checkpoint_dir=CHECKPOINT_DIR,
        vocab_path=VOCAB_FILE,
        use_deepspeed=False,
    )
    model.to(DEVICE)
    return model


def generate_audio(texts, model, gpt_cond_latent, speaker_embedding, lang="wo"):
    wav_chunks = []
    for text in tqdm(texts):
        wav_chunk = XTTS_MODEL.inference(
            text=text,
            language=lang,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=0.1,
            length_penalty=1.0,
            repetition_penalty=10.0,
            top_k=10,
            top_p=0.3,
        )
        wav_chunks.append(torch.tensor(wav_chunk["wav"]))
    out_wav = torch.cat(wav_chunks, dim=0).unsqueeze(0).cpu()
    return out_wav


if __name__ == "__main__":
    # Ensure the destination folder exists
    os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)

    # Read sentences from the corpus file
    with open(CORPUS_FILE, "r", encoding="utf-8") as file:
        sentences = file.readlines()
        print("Sentences loaded from corpus.")

    # Load model
    XTTS_MODEL = load_model()
    print("Model loaded successfully!")

    # Initialize verbalizer
    verbalizer = WolofTTSVerbalizer()

    # Extract speaker conditioning from the reference audio
    gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(
        audio_path=REFERENCE_AUDIO_FILE,
        gpt_cond_len=XTTS_MODEL.config.gpt_cond_len,
        max_ref_length=XTTS_MODEL.config.max_ref_len,
        sound_norm_refs=XTTS_MODEL.config.sound_norm_refs,
    )
    print("Reference audio conditioning extracted.")

    # Generate and save audio for each sentence
    rtf_values = []
    for i, sentence in enumerate(tqdm(sentences, desc="Generating audio")):
        sentence = sentence.strip()
        if not sentence:
            continue

        # Generate audio
        start_time = time.time()
        sentence = verbalizer.verbalize(sentence)
        tts_texts = sent_tokenize(sentence)
        audio = generate_audio(tts_texts, XTTS_MODEL, gpt_cond_latent, speaker_embedding)
        waveform = torch.tensor(audio, dtype=torch.float32)
        inference_time = time.time() - start_time
        audio_duration = waveform.shape[1] / OUTPUT_SAMPLING_RATE
        rtf = inference_time / audio_duration if audio_duration > 0 else float("inf")
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