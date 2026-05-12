from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "soynade-research/Oolel-Corrector"
device = "cuda"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

SYSTEM_PROMPT = (
        "Fix the orthography of this Wolof social media text. "
        "Apply standard rules: correct vowels with diacritics, restore geminates, fix French-influenced spellings. "
        "Keep mixed French/English as-is. Reply only with <CORRECTION>corrected text</CORRECTION>."
    )

def correct(text, system_prompt=SYSTEM_PROMPT, max_new_tokens=512, temperature=0.1):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": text}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    generated_ids = model.generate(
        inputs.input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature
    )
    generated_ids = [
        output[len(inp):]
        for inp, output in zip(inputs.input_ids, generated_ids)
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

text = "man mom khawma sakh li loumou done. Niom gneup dem naniou dakar ngir vacances scolaires yi"
print(correct(text))
# <CORRECTION>Man moom xawma sax li lu mu doon. Ñoom ñépp dem nañu Dakar ngir vacances scolaires yi.</CORRECTION>
