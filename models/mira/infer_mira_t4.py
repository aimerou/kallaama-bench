"""
Requirements
------------
1. Install training requirements
2. Install inference requirements
    pip install git+https://github.com/ysharma3501/MiraTTS.git
3. In case of tvm-ffi issue (unsloth dependency), install the right version
    pip install "apache-tvm-ffi==0.1.11"
4. In case of torchcodec issue, make sure ffmpeg is available, then install the right version
    pip uninstall -y torchcodec
    pip install "torchcodec==0.10.*" --index-url https://download.pytorch.org/whl/cu128

Usage
-----
python3 infer_mira_t4.py --config-path configs/infer_mira_ft.yaml

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

import librosa
import pandas as pd
import torch
import torchaudio
import numpy as np
import yaml
from unsloth import FastModel
from huggingface_hub import login
from tqdm import tqdm
from ncodec.codec import TTSCodec
from ncodec.encoder.model import audio_volume_normalize
from verbalizer import WolofTTSVerbalizer
import soundfile as sf

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


@torch.inference_mode()
def encode(audio, encode_semantic=True, duration=None):
    """Encodes audio into speech tokens and context tokens.

    Args:
        audio: numpy array of audio samples at 16 kHz (already loaded by caller).
        encode_semantic: if True, also return speech tokens; otherwise context tokens only.
        duration: unused, kept for API compatibility.
    """
    self = tts_codec.audio_encoder
    audio = audio_volume_normalize(audio)

    ref_clip = self.get_ref_clip(audio)
    wav_ref = torch.from_numpy(ref_clip).unsqueeze(0).float()

    mel = self.mel_transformer(wav_ref).squeeze(1)
    new_arr = np.array(mel.transpose(1, 2).cpu())

    global_tokens = self.s_encoder.run(["global_tokens"], {"mel_spectrogram": new_arr})
    context_tokens = "".join([f"<|context_token_{i}|>" for i in global_tokens[0].squeeze()])

    if encode_semantic:
        feat = self.extract_wav2vec2_features(audio)
        speech_tokens = self.q_encoder.run(["semantic_tokens"], {"features": feat.cpu().detach().numpy()})
        speech_tokens = "".join([f"<|speech_token_{i}|>" for i in speech_tokens[0][0]])
        return speech_tokens, context_tokens
    else:
        return context_tokens


def infer(
    formatted_prompt,
    context_tokens,
    model,
    tokenizer,
    top_k,
    top_p,
    temperature,
    repetition_penalty,
    max_new_audio_tokens,
):
    model_inputs = tokenizer([formatted_prompt], return_tensors="pt").to(DEVICE)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_audio_tokens,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    generated_ids_trimmed = generated_ids[:, model_inputs.input_ids.shape[1]:]
    predicts_text = tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=False)[0]
    audio = tts_codec.decode(predicts_text, context_tokens)
    return audio


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
    parser = argparse.ArgumentParser(description="miraTTS inference script")
    parser.add_argument("--config-path", required=True, help="Path to config.yaml")
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────────────────
    cfg = load_config(args.config_path)

    token = os.getenv("HF_TOKEN")

    # Corpus / reference audio
    corpus_file          = cfg["inference"]["corpus_file"]
    corpus_separator     = cfg["inference"].get("corpus_separator", "|")
    reference_audio_dir  = cfg["inference"]["reference_audio_dir"]

    # Submission output
    output_dir            = cfg["inference"]["output_dir"]
    output_audio_dir       = os.path.join(output_dir, "audio")
    synthesis_sampling_rate = cfg["inference"]["synthesis_sampling_rate"]
    output_sampling_rate    = cfg["inference"]["output_sampling_rate"]
    num_sentences          = cfg["inference"].get("num_sentences")

    # Generation hyper-parameters
    gen                = cfg["inference"]["generation"]
    top_k              = gen["top_k"]
    top_p              = gen["top_p"]
    temperature        = gen["temperature"]
    repetition_penalty = gen["repetition_penalty"]
    max_new_tokens     = gen["max_new_audio_tokens"]

    # Audio processing (used when loading reference audio for encoding)
    ref_load_sample_rate = cfg["inference"]["audio_processing"]["sample_rate"]

    # Model
    model_cfg     = cfg["model"]
    model_repo    = model_cfg["repo"]
    model_dtype   = getattr(torch, model_cfg["dtype"])
    model_4bit    = model_cfg["load_in_4bit"]
    model_full_ft = model_cfg["full_finetuning"]

    # Submission metadata
    sub_cfg = cfg.get("submission", {})
    # ─────────────────────────────────────────────────────────────────────────────

    # Login to Hugging Face
    login(token)

    # Ensure the destination folders exist
    os.makedirs(output_audio_dir, exist_ok=True)

    # Load the KallaamaBench test corpus
    corpus = load_corpus(corpus_file, corpus_separator)
    if num_sentences is not None:
        corpus = corpus.iloc[:num_sentences]
    print(f"Loaded {len(corpus)} rows from corpus.")

    # Download model and tokenizer
    print("Downloading model...")
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_repo,
        dtype=model_dtype,
        full_finetuning=model_full_ft,
        load_in_4bit=model_4bit,
    )
    print("Model downloaded.")

    # Initialize verbalizer
    verbalizer = WolofTTSVerbalizer()

    # Set up the codec
    tts_codec = TTSCodec()
    tts_codec.audio_encoder.encode = encode
    tts_codec.audio_encoder.feature_extractor.config.output_hidden_states = True

    # Encode reference audio per-speaker, caching context tokens so each unique
    # reference is only encoded once (encode() with encode_semantic=False mirrors
    # the exact pipeline used during training in process_wavs()).
    context_tokens_cache = {}

    def get_context_tokens(speaker):
        if speaker not in context_tokens_cache:
            ref_path = resolve_reference_audio_path(speaker, reference_audio_dir)
            ref_audio, _ = librosa.load(ref_path, sr=ref_load_sample_rate)
            context_tokens_cache[speaker] = tts_codec.audio_encoder.encode(ref_audio, encode_semantic=False)
        return context_tokens_cache[speaker]

    # Generate and save audio for each corpus row
    rtf_values = []
    for _, row in tqdm(corpus.iterrows(), total=len(corpus), desc="Generating audio"):
        filename = row["filename"]
        transcription = row["transcription"]
        speaker = row["speaker"].strip()

        context_tokens = get_context_tokens(speaker)

        start_time = time.time()
        sentence = verbalizer.verbalize(transcription)
        formatted_prompt = (
            f"<|task_tts|><|start_text|>{sentence}<|end_text|>"
            f"<|context_audio_start|>{context_tokens}<|context_audio_end|>"
            f"<|prompt_speech_start|>"
        )
        audio = infer(
            formatted_prompt, context_tokens, model, tokenizer,
            top_k, top_p, temperature, repetition_penalty, max_new_tokens,
        )
        inference_time = time.time() - start_time

        output_path = os.path.join(output_audio_dir, filename)
        audio_duration = save_submission_audio(
            audio, synthesis_sampling_rate, output_sampling_rate, output_path
        )
        rtf = inference_time / audio_duration if audio_duration > 0 else float("inf")
        rtf_values.append(rtf)

    print(f"Audio files generated and saved to: {output_audio_dir}")

    avg_rtf = sum(rtf_values) / len(rtf_values) if rtf_values else 0
    print(f"Average realtime factor (RTF) across all sentences: {avg_rtf:.3f}")

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