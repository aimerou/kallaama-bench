import os
import json
import time
import torch
import torchaudio
from snac import SNAC
from huggingface_hub import login
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from verbalizer import WolofTTSVerbalizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOKEN = os.getenv("HF_TOKEN")

CORPUS_TYPE = "code-mixed"
CORPUS_FILE = "/root/wolof-tts/evaluation/data/corpus_mixed.txt"
OUTPUT_METRICS_FILE = "/root/wolof-tts/evaluation/metrics/metrics_rtf_all.json"

MODEL_NAME = "orpheus-tts"
BASE_MODEL_REPO = "canopylabs/3b-fr-ft-research_release"
DESTINATION_FOLDER = f"/root/wolof-tts/evaluation/outputs/{CORPUS_TYPE}/{MODEL_NAME}"
OUTPUT_SAMPLING_RATE = 24000

def redistribute_codes(code_list):
  layer_1 = []
  layer_2 = []
  layer_3 = []
  for i in range((len(code_list)+1)//7):
    layer_1.append(code_list[7*i])
    layer_2.append(code_list[7*i+1]-4096)
    layer_3.append(code_list[7*i+2]-(2*4096))
    layer_3.append(code_list[7*i+3]-(3*4096))
    layer_2.append(code_list[7*i+4]-(4*4096))
    layer_3.append(code_list[7*i+5]-(5*4096))
    layer_3.append(code_list[7*i+6]-(6*4096))
  codes = [torch.tensor(layer_1).unsqueeze(0).to(DEVICE),
         torch.tensor(layer_2).unsqueeze(0).to(DEVICE),
         torch.tensor(layer_3).unsqueeze(0).to(DEVICE)]

  with torch.no_grad():
    audio_hat = snac_model.decode(codes)
  return audio_hat

def generate_audio(prompts, chosen_voice, model, tokenizer):
  start_token = torch.tensor([[ 128259]], dtype=torch.int64) # Start of human
  end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64) # End of text, End of human

  prompts_ = [(f"{chosen_voice}: " + p) if chosen_voice else p for p in prompts]

  all_input_ids = []

  for prompt in prompts_:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    all_input_ids.append(input_ids)

  all_modified_input_ids = []
  for input_ids in all_input_ids:
    modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1) # SOH SOT Text EOT EOH
    all_modified_input_ids.append(modified_input_ids)

  all_padded_tensors = []
  all_attention_masks = []
  max_length = max([modified_input_ids.shape[1] for modified_input_ids in all_modified_input_ids])
  for modified_input_ids in all_modified_input_ids:
    padding = max_length - modified_input_ids.shape[1]
    padded_tensor = torch.cat([torch.full((1, padding), tokenizer.pad_token_id, dtype=torch.int64), modified_input_ids], dim=1)
    attention_mask = torch.cat([torch.zeros((1, padding), dtype=torch.int64), torch.ones((1, modified_input_ids.shape[1]), dtype=torch.int64)], dim=1)
    all_padded_tensors.append(padded_tensor)
    all_attention_masks.append(attention_mask)

  all_padded_tensors = torch.cat(all_padded_tensors, dim=0)
  all_attention_masks = torch.cat(all_attention_masks, dim=0)

  input_ids = all_padded_tensors.to(DEVICE)
  attention_mask = all_attention_masks.to(DEVICE)

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
      use_cache = True
    )

  token_to_find = 128257
  token_to_remove = 128258

  token_indices = (generated_ids == token_to_find).nonzero(as_tuple=True)

  if len(token_indices[1]) > 0:
      last_occurrence_idx = token_indices[1][-1].item()
      cropped_tensor = generated_ids[:, last_occurrence_idx+1:]
  else:
      cropped_tensor = generated_ids

  mask = cropped_tensor != token_to_remove

  processed_rows = []

  for row in cropped_tensor:
      masked_row = row[row != token_to_remove]
      processed_rows.append(masked_row)

  code_lists = []

  for row in processed_rows:
      row_length = row.size(0)
      new_length = (row_length // 7) * 7
      trimmed_row = row[:new_length]
      trimmed_row = [t - 128266 for t in trimmed_row]
      code_lists.append(trimmed_row)

  my_samples = []
  for code_list in code_lists:
    samples = redistribute_codes(code_list)
    my_samples.append(samples)

  return my_samples

if __name__ == "__main__":
    # Login to Hugging Face
    login(TOKEN)

    # Ensure the destination folder exists
    os.makedirs(DESTINATION_FOLDER, exist_ok=True)

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_REPO)
    model.to(DEVICE)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_REPO)

    # Define pad token
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Initialize verbalizer
    verbalizer = WolofTTSVerbalizer()

    # Load SNAC model
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
    snac_model.to(DEVICE)

    # Read sentences from the corpus file
    with open(CORPUS_FILE, "r", encoding="utf-8") as file:
        sentences = file.readlines()
        print("Sentences loaded from corpus.")

    # Generate and save audio for each sentence
    rtf_values = []
    for idx, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not sentence:
            continue

        # Verbalize the sentence
        start_time = time.time()
        sentence = verbalizer.verbalize(sentence)

        # Text input
        prompt = [sentence]
        chosen_voice = None #for single speaker

        #Run Inference
        try:
            with torch.no_grad():
                output_audio = generate_audio(prompt, chosen_voice, model, tokenizer)[0]
        except Exception as e:
            print(f"Error generating audio for sentence {idx + 1}: {e}")
            continue
        inference_time = time.time() - start_time
        audio_duration = output_audio.shape[2] / OUTPUT_SAMPLING_RATE
        rtf = inference_time / audio_duration if audio_duration > 0 else float('inf')
        rtf_values.append(rtf)
        print(f"Sentence {idx + 1}: Duration = {audio_duration:.2f}s, Inference Time = {inference_time:.2f}s, RTF = {rtf:.2f}")

        # Save the audio to the destination folder
        output_path = os.path.join(DESTINATION_FOLDER, f"sentence_{idx + 1:02}.wav")
        torchaudio.save(output_path, output_audio.detach().cpu().squeeze(0), OUTPUT_SAMPLING_RATE)

    print(f"Audio files generated and saved to: {DESTINATION_FOLDER}")

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