"""
Requirements
------------
pip install -r requirements.txt
pip install git+https://github.com/ysharma3501/FastBiCodec.git git+https://github.com/ysharma3501/FlashSR.git
pip install "torch==2.8.0" "torchaudio==2.8.0" "torchvision==0.23.0" --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir
pip install git+https://github.com/ysharma3501/MiraTTS.git

Usage
-----
python infer_mira_finetuned.py
"""

import os
import time
import json
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
from verbalizer import WolofTTSVerbalizer

TOKEN = os.getenv("HF_TOKEN")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CORPUS_FILE          = "/root/kallaama/data/corpus.csv"
REFERENCE_AUDIO_FOLDER = "/root/kallaama/data/references"
OUTPUT_DIR           = "/root/kallaama/submissions/mira-finetuned"
OUTPUT_AUDIO_DIR     = os.path.join(OUTPUT_DIR, "audio")
OUTPUT_METADATA      = os.path.join(OUTPUT_DIR, "metadata.json")

MODEL_REPO = "intronhealth/miraTTS-kinyarwanda"
OUTPUT_SAMPLING_RATE = 48000

# Team information written to metadata.json
METADATA = {
    "team"            : "Baamtu Datamation",
    "system"          : "MiraTTS Finetuned",
    "submission_date" : "2026-05-19",
    "contact"         : "contact@example.com",
    "notes"           : "RTF measured on a single NVIDIA A100 40 GB",
}


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
    formatted_prompt, context_tokens, model, tokenizer, top_k=50, top_p=1.0,
    temperature=0.8, repetition_penalty=1.2, max_new_audio_tokens=1024
    ):
    model_inputs = tokenizer([formatted_prompt], return_tensors="pt").to(DEVICE)

    print("Generating token sequence...")
    print(f"Eos token id: {tokenizer.eos_token_id}, Pad token id: {tokenizer.pad_token_id}")
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_audio_tokens, # Limit generation length
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_token_id=tokenizer.eos_token_id, # Stop token
        pad_token_id=tokenizer.pad_token_id, # Use models pad token id
    )
    print("Token sequence generated.")

    generated_ids_trimmed = generated_ids[:, model_inputs.input_ids.shape[1]:]
    predicts_text = tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=False)[0]
    audio = tts_codec.decode(predicts_text, context_tokens)
    return audio

if __name__ == "__main__":
    # Login to Hugging Face
    login(TOKEN)

    # Ensure the destination folder exists
    os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)

    # Load corpus
    corpus = pd.read_csv(CORPUS_FILE, sep="|", header=0, on_bad_lines="skip", dtype=str)
    corpus = corpus.dropna(subset=["filename", "transcription"]).reset_index(drop=True)
    print(f"Loaded {len(corpus)} sentences from corpus.")

    # Download model and tokenizer
    print("Downloading model...")
    model, tokenizer = FastModel.from_pretrained(
        model_name = MODEL_REPO,
        dtype = torch.float32, # Only works in float32 or bfloat16, NaNs in fp16
        full_finetuning = True,
        load_in_4bit = False
    )
    print("Model downloaded.")

    # Initialize verbalizer
    verbalizer = WolofTTSVerbalizer()

    # Intialize TTSCodec
    tts_codec = TTSCodec()
    tts_codec.audio_encoder.encode = encode
    tts_codec.audio_encoder.feature_extractor.config.output_hidden_states = True

    # Generate and save audio for each sentence
    rtf_values = []
    for index, row in tqdm(corpus.iterrows(), total=len(corpus), desc="Generating audio"):
        if index < 10:  # Limit to first 10 sentences for faster testing; remove this line for full corpus
            sentence = row["transcription"].strip()
            speaker = row["speaker"].strip()

            if not sentence:  # Skip empty lines
                continue

            # Build the prompt using the same template as process_wavs() in training.
            # The prompt ends with <|prompt_speech_start|> — the model generates speech
            # tokens from that point, terminated by its eos token (<|im_end|>).
            start_time = time.time()

            reference_audio_file = os.path.join(REFERENCE_AUDIO_FOLDER, f"reference_{speaker}.wav")
            ref_audio, _ = librosa.load(reference_audio_file, sr=16000)
            context_tokens = tts_codec.audio_encoder.encode(ref_audio, encode_semantic=False)

            sentence        = verbalizer.verbalize(sentence)
            formatted_prompt = (
                f"<|task_tts|><|start_text|>{sentence}<|end_text|>"
                f"<|context_audio_start|>{context_tokens}<|context_audio_end|>"
                f"<|prompt_speech_start|>"
            )

            audio = infer(formatted_prompt, context_tokens, model, tokenizer)
            waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

            inference_time = time.time() - start_time
            audio_duration = waveform.shape[1] / OUTPUT_SAMPLING_RATE
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