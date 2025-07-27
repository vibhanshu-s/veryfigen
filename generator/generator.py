import json
import random
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os
# Load prompt templates
labels_path = os.path.join(os.path.dirname(__file__), "labels.json")
with open(labels_path, "r") as f:
    prompt_templates = json.load(f)

# Load model & tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

def generate_fake_news(label, max_length=50):
    if label not in prompt_templates:
        raise ValueError(f"Invalid label: '{label}' not found in prompt templates.")

    prompt = random.choice(prompt_templates[label])
    inputs = tokenizer(prompt, return_tensors='pt')
    attention_mask = inputs['attention_mask']

    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=attention_mask,
            max_length=max_length,
            do_sample=True,
            top_k=50,
            top_p=0.92,
            temperature=0.8,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Keep only the first full sentence for headline format
    if "." in generated:
        generated = generated[:generated.find(".")+1]

    # print(f"[{label}] Generated Fake Headline: {generated}")
    return generated
