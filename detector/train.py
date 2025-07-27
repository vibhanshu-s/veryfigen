import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# ==== CONFIG ====
EPOCHS = 5
BATCH_SIZE = 32
MAX_LEN = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "bert_model"

# ==== Dataset ====
class NewsDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=MAX_LEN,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx])
        }

# ==== Load Data ====
def load_data(tokenizer):
    label_map = {"fake": 0, "real": 1}
    train_df = pd.read_csv("../data/train.csv")
    val_df = pd.read_csv("../data/val.csv")
    test_df = pd.read_csv("../data/test.csv")

    for df in [train_df, val_df, test_df]:
        df["label"] = df["label"].map(label_map)

    train_dataset = NewsDataset(train_df, tokenizer)
    val_dataset = NewsDataset(val_df, tokenizer)
    test_dataset = NewsDataset(test_df, tokenizer)
    return train_dataset, val_dataset, test_dataset

# ==== Evaluation ====
def evaluate(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            all_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return total_loss / len(dataloader), acc, f1

# ==== Training ====
def train():
    print("📥 Loading tokenizer and datasets...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_dataset, val_dataset, test_dataset = load_data(tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    print("🧠 Initializing model...")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        correct, total = 0, 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in loop:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            loop.set_postfix(loss=loss.item())

        train_acc = correct / total
        train_loss = total_train_loss / len(train_loader)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader)

        print(f"\n📊 Epoch {epoch+1} Results:")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}\n")

    # ==== Final Test Evaluation ====
    test_loss, test_acc, test_f1 = evaluate(model, test_loader)
    print("✅ Final Test Evaluation:")
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}")

    print("💾 Saving model and tokenizer...")
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"✅ Model saved to ./{MODEL_DIR}")

if __name__ == "__main__":
    train()
