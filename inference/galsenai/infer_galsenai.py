# /// script
# requires-python = ">=3.10,<3.12"
# dependencies = [
#     "TTS==0.22.0",
#     "torch==2.5.1",
#     "torchaudio==2.5.1",
#     "transformers>=4.45.2,<4.47",
#     "tokenizers==0.20.1",
#     "underthesea==6.8.4",
#     "pandas<2.0",
#     "setuptools<81",  # librosa 0.10.0 (pinned by TTS) imports pkg_resources
#     "tqdm",
# ]
# ///
"""
Requirements
------------
Download model: uvx gdown 1jsGAMBo354uRhwVKNnuJpLtq2uJYcVHN
Unzip model: unzip galsenai-xtts-wo-checkpoints
Python dependencies are declared in the inline script metadata above; `uv run` installs them
in an isolated environment (Coqui TTS needs Python < 3.12, uv fetches one).

Usage
-----
uv run infer_galsenai.py
"""

import os
import json
import time
import torch
import torchaudio
import pandas as pd
import TTS.tts.models.xtts as xtts_module
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from tqdm import tqdm
from underthesea import sent_tokenize
import wolof_xtts_tokenizer
from verbalizer import WolofTTSVerbalizer

# Coqui TTS 0.22 has no Wolof ("wo") text rules: swap in wolof_xtts_tokenizer.py, a copy of
# TTS/tts/layers/xtts/tokenizer.py with Wolof support, instead of editing the installed package.
xtts_module.VoiceBpeTokenizer = wolof_xtts_tokenizer.VoiceBpeTokenizer
xtts_module.split_sentence = wolof_xtts_tokenizer.split_sentence

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CORPUS_FILE          = "/root/kallaama-bench/data/corpus.csv"
REFERENCE_AUDIO_FOLDER = "/root/kallaama-bench/data/references/audio/"
OUTPUT_DIR           = "/root/kallaama-bench/submissions/galsenai/"
OUTPUT_AUDIO_DIR     = os.path.join(OUTPUT_DIR, "audio")
OUTPUT_METADATA      = os.path.join(OUTPUT_DIR, "metadata.json")

CHECKPOINTS_ROOT = "/root/kallaama-bench/inference/galsenai/galsenai-xtts-wo-checkpoints/"
CHECKPOINT_DIR   = os.path.join(CHECKPOINTS_ROOT, "Anta_GPT_XTTS_Wo")
MODEL_FILE       = "best_model_89250.pth"
VOCAB_FILE       = os.path.join(CHECKPOINTS_ROOT, "XTTS_v2.0_original_model_files/vocab.json")
OUTPUT_SAMPLING_RATE = 24000

# Team information written to metadata.json
METADATA = {
    "team"            : "Kallaama-Bench Team",
    "system"          : "XTTSv2 fine-tuned on Wolof",
    "submission_date" : "2026-05-19",
    "contact"         : "contact@example.com",
    "notes"           : "RTF measured on a single NVIDIA A100 40 GB",
}


def load_model():
    xtts_checkpoint = os.path.join(CHECKPOINT_DIR, MODEL_FILE)
    xtts_config     = os.path.join(CHECKPOINT_DIR, "config.json")
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
    for text in texts:
        wav_chunk = model.inference(
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
    return torch.cat(wav_chunks, dim=0).unsqueeze(0).cpu()


if __name__ == "__main__":
    os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)

    # Load corpus
    corpus = pd.read_csv(CORPUS_FILE, sep="|", header=0, on_bad_lines="skip", dtype=str)
    corpus = corpus.dropna(subset=["filename", "transcription"]).reset_index(drop=True)
    print(f"Loaded {len(corpus)} sentences from corpus.")

    # Load model
    XTTS_MODEL = load_model()
    print("Model loaded successfully!")

    # Initialize verbalizer
    verbalizer = WolofTTSVerbalizer()

    # Generate audio and compute RTF
    rtf_values = []
    for _, row in tqdm(corpus.iterrows(), total=len(corpus), desc="Generating audio"):
        sentence = row["transcription"].strip()
        speaker = row["speaker"].strip()
        if not sentence:
            continue

        start_time     = time.time()

        reference_audio_file = os.path.join(REFERENCE_AUDIO_FOLDER, f"reference_{speaker}.wav")
        gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(
            audio_path=reference_audio_file,
            gpt_cond_len=XTTS_MODEL.config.gpt_cond_len,
            max_ref_length=XTTS_MODEL.config.max_ref_len,
            sound_norm_refs=XTTS_MODEL.config.sound_norm_refs,
        )

        sentence       = verbalizer.verbalize(sentence)
        waveform       = generate_audio(sent_tokenize(sentence), XTTS_MODEL, gpt_cond_latent, speaker_embedding)
        
        inference_time = time.time() - start_time
        audio_duration = waveform.shape[1] / OUTPUT_SAMPLING_RATE
        rtf            = inference_time / audio_duration if audio_duration > 0 else float("inf")
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