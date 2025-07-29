# detector/predict.py
import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification

from pathlib import Path
from model_dl import main as download_models

MODEL_PATH = "detector/fine_tuned_bert"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Check if model folder exists ====
if not Path(MODEL_PATH).exists():
    print(f"⚠️ Model directory '{MODEL_PATH}' not found. Downloading...")
    download_models()

# ==== Load tokenizer and model ====
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()

# ==== Predict function ====
def predict_headline(text: str) -> str:
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
        return "Fake" if prediction == 0 else "Real"
