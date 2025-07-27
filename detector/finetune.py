import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_scheduler
from tqdm import tqdm

# === CONFIG ===
MODEL_PATH = "bert_model"
CSV_PATH = "../data/finetune.csv"
SAVE_PATH = "fine_tuned_bert"
EPOCHS = 3
BATCH_SIZE = 16
LR = 2e-5
MAX_LEN = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Dataset ===
class NewsDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.texts = dataframe["text"].tolist()
        self.labels = dataframe["label"].astype(int).tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoded.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# === Load tokenizer & model ===
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)

# === Load and preprocess dataset ===
df = pd.read_csv(CSV_PATH).dropna()
assert "text" in df.columns and "label" in df.columns, "CSV must contain 'text' and 'label' columns."

dataset = NewsDataset(df, tokenizer)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# === Optimizer & Scheduler ===
optimizer = AdamW(model.parameters(), lr=LR)
num_training_steps = EPOCHS * len(dataloader)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# === Fine-tuning ===
model.train()
for epoch in range(EPOCHS):
    loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
    for batch in loop:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        loop.set_postfix(loss=loss.item())

# === Save fine-tuned model ===
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print("✅ Fine-tuning complete. Model saved to:", SAVE_PATH)
