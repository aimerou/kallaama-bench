import os
import json
import time
import torch
import torchaudio
import pandas as pd
from snac import SNAC
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from verbalizer import WolofTTSVerbalizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOKEN  = os.getenv("HF_TOKEN")

CORPUS_FILE      = "/root/kallaama/data/corpus.csv"
OUTPUT_DIR       = "/root/kallaama/outputs/orpheus-tts"
OUTPUT_AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")
OUTPUT_METADATA  = os.path.join(OUTPUT_DIR, "metadata.json")

BASE_MODEL_REPO      = "canopylabs/3b-fr-ft-research_release"
OUTPUT_SAMPLING_RATE = 24000

# Team information written to metadata.json
METADATA = {
    "team"            : "Canopy Labs",
    "system"          : "Orpheus TTS",
    "submission_date" : "2026-05-19",
    "contact"         : "contact@example.com",
    "notes"           : "RTF measured on a single NVIDIA A100 40 GB",
}


def redistribute_codes(code_list):
    layer_1, layer_2, layer_3 = [], [], []
    for i in range((len(code_list) + 1) // 7):
        layer_1.append(code_list[7*i])
        layer_2.append(code_list[7*i+1] - 4096)
        layer_3.append(code_list[7*i+2] - (2*4096))
        layer_3.append(code_list[7*i+3] - (3*4096))
        layer_2.append(code_list[7*i+4] - (4*4096))
        layer_3.append(code_list[7*i+5] - (5*4096))
        layer_3.append(code_list[7*i+6] - (6*4096))
    codes = [
        torch.tensor(layer_1).unsqueeze(0).to(DEVICE),
        torch.tensor(layer_2).unsqueeze(0).to(DEVICE),
        torch.tensor(layer_3).unsqueeze(0).to(DEVICE),
    ]
    with torch.no_grad():
        return snac_model.decode(codes)


def generate_audio(prompts, chosen_voice, model, tokenizer):
    start_token = torch.tensor([[128259]], dtype=torch.int64)
    end_tokens  = torch.tensor([[128009, 128260]], dtype=torch.int64)

    prompts_ = [(f"{chosen_voice}: " + p) if chosen_voice else p for p in prompts]

    all_modified = []
    for prompt in prompts_:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        all_modified.append(torch.cat([start_token, input_ids, end_tokens], dim=1))

    max_length = max(t.shape[1] for t in all_modified)
    padded, masks = [], []
    for modified in all_modified:
        pad = max_length - modified.shape[1]
        padded.append(torch.cat([torch.full((1, pad), tokenizer.pad_token_id, dtype=torch.int64), modified], dim=1))
        masks.append(torch.cat([torch.zeros((1, pad), dtype=torch.int64), torch.ones((1, modified.shape[1]), dtype=torch.int64)], dim=1))

    input_ids      = torch.cat(padded, dim=0).to(DEVICE)
    attention_mask = torch.cat(masks,  dim=0).to(DEVICE)

    generated_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=1200,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        repetition_penalty=1.1,
        num_return_sequences=1,
        eos_token_id=128258,
        use_cache=True,
    )

    token_indices = (generated_ids == 128257).nonzero(as_tuple=True)
    if len(token_indices[1]) > 0:
        cropped = generated_ids[:, token_indices[1][-1].item() + 1:]
    else:
        cropped = generated_ids

    code_lists = []
    for row in cropped:
        masked    = row[row != 128258]
        trimmed   = masked[: (masked.size(0) // 7) * 7]
        code_lists.append([t - 128266 for t in trimmed])

    return [redistribute_codes(cl) for cl in code_lists]


if __name__ == "__main__":
    login(TOKEN)
    os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)

    # Load corpus
    corpus = pd.read_csv(CORPUS_FILE, sep="|", header=0, on_bad_lines="skip", dtype=str)
    corpus = corpus.dropna(subset=["filename", "transcription"]).reset_index(drop=True)
    print(f"Loaded {len(corpus)} sentences from corpus.")

    # Load model and tokenizer
    model     = AutoModelForCausalLM.from_pretrained(BASE_MODEL_REPO).to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_REPO)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print("Model loaded successfully!")

    # Initialize verbalizer and SNAC vocoder
    verbalizer = WolofTTSVerbalizer()
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(DEVICE)

    # Generate audio and compute RTF
    rtf_values = []
    for _, row in corpus.iterrows():
        sentence = row["transcription"].strip()
        speaker = row["speaker"].strip()
        gender = int(row["gender"])

        if not sentence:
            continue

        start_time = time.time()
        sentence   = verbalizer.verbalize(sentence)
        try:
            if gender == 0:
                chosen_voice = "tara" #female
            else:
                chosen_voice = "zac" #male
            with torch.no_grad():
                output_audio = generate_audio([sentence], chosen_voice, model, tokenizer)[0]
        except Exception as e:
            print(f"Error on {row['filename']}: {e}")
            continue

        inference_time = time.time() - start_time
        audio_duration = output_audio.shape[2] / OUTPUT_SAMPLING_RATE
        rtf            = inference_time / audio_duration if audio_duration > 0 else float("inf")
        rtf_values.append(rtf)

        torchaudio.save(os.path.join(OUTPUT_AUDIO_DIR, row["filename"]),
                        output_audio.detach().cpu().squeeze(0), OUTPUT_SAMPLING_RATE)

    avg_rtf = sum(rtf_values) / len(rtf_values) if rtf_values else 0.0
    print(f"Audio files saved to: {OUTPUT_AUDIO_DIR}")
    print(f"Average RTF: {avg_rtf:.3f}")

    # Save team info + RTF to metadata.json
    metadata = {**METADATA, "rtf": round(avg_rtf, 4)}
    with open(OUTPUT_METADATA, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to: {OUTPUT_METADATA}")