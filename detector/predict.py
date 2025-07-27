# detector/predict.py
import torch
from transformers import BertTokenizer, BertForSequenceClassification

MODEL_PATH = "detector/fine_tuned_bert"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load once
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()

def predict_headline(text: str) -> str:
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
        return "Fake" if prediction == 0 else "Real"
