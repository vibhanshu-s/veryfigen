import random
import csv
import json
from tqdm import tqdm
from generator import generate_fake_news  # assumes your function is named generate_fake_news

# Load prompt labels
with open("labels.json", "r") as f:
    prompt_templates = json.load(f)
labels = list(prompt_templates.keys())

# Generate 1000 fake news headlines
generated_data = []

for _ in tqdm(range(100), desc="Generating 100 headlines"):
    label = random.choice(labels)
    headline = generate_fake_news(label)
    generated_data.append({"headline": headline})

# Save to generated.csv (no label)
with open("test.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["headline"])
    writer.writeheader()
    writer.writerows(generated_data)

print("✅ 100 fake headlines saved to test.csv")
