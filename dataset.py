import pandas as pd
import re
from sklearn.model_selection import train_test_split

# Load datasets
true_df = pd.read_csv("dataset/True.csv")
fake_df = pd.read_csv("dataset/Fake.csv")

# Assume headlines are stored in 'title' or 'text' column
HEADLINE_COLUMN = "title" if "title" in true_df.columns else "text"

# Assign labels
true_df = true_df[[HEADLINE_COLUMN]].copy()
true_df['label'] = 'real'

fake_df = fake_df[[HEADLINE_COLUMN]].copy()
fake_df['label'] = 'fake'

# Merge
df = pd.concat([true_df, fake_df], ignore_index=True)
df.rename(columns={HEADLINE_COLUMN: "text"}, inplace=True)

# Clean text: remove anything in brackets and links
def clean_text(text):
    text = re.sub(r'\[[^\]]*\]', '', text)  # Remove [brackets]
    text = re.sub(r'\([^\)]*\)', '', text)  # Remove (brackets)
    text = re.sub(r'http\S+|www.\S+', '', text)  # Remove URLs
    return text.strip()

df['text'] = df['text'].astype(str).apply(clean_text)
df = df[df['text'].str.strip().astype(bool)]  # Remove empty texts

# Shuffle and split
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

# Save CSVs
train_df.to_csv("data/train.csv", index=False)
val_df.to_csv("data/val.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

print("✅ Dataset with headlines only cleaned and split:")
print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
