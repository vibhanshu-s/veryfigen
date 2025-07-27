# 📰 VeryFiGen: AI-Powered Fake News Generation & Detection

**VeryFiGen** is a dual-purpose AI system that can generate realistic fake news headlines using GPT-2 and detect fake headlines using a fine-tuned BERT classifier. Built with an intuitive interface using **Streamlit**, it serves as a demo tool for understanding generative and discriminative NLP models in action.

---

## 🚀 Features

- ✅ **Generate** realistic fake news headlines by topic
- 🔍 **Detect** whether a headline is real or fake using BERT
- 💾 Saves generated fake news to CSV for analysis
- 📂 Google Drive model downloads to bypass GitHub size limits
- 🖥️ Clean Streamlit interface with tabbed UX

---

## ⚙️ Tech Stack

| Tool/Library         | Purpose                                      |
|----------------------|----------------------------------------------|
| `Streamlit`          | UI frontend                                  |
| `transformers`       | GPT-2 and BERT model loading (Hugging Face) |
| `PyTorch`            | Deep learning backend                        |
| `GPT-2`              | Fake news headline generation                |
| `BERT`               | Fine-tuned for headline classification       |
| `Google Drive`       | Stores large model files                     |
| `Python`             | Core language                                |

---

## 📁 Project Structure

```
veryfigen/
├── app.py                     # Main Streamlit app
├── model_dl.py                # Script to download models from Google Drive
├── requirements.txt
├── data/
│   └── fake_news_generated.csv  # Output of generated headlines
├── generator/
│   ├── generator.py           # GPT-2 based text generator
│   └── labels.json            # Prompt templates
├── detector/
│   ├── predict.py             # Fake news detector using BERT
│   └── model/                 # BERT fine-tuned model (downloaded)
```

---

## 💾 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/vibhanshu-s/veryfigen.git
cd veryfigen
```

### 2. Create Virtual Environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Download Pretrained Models

Use the included script to download your fine-tuned GPT-2 and BERT models from Google Drive:

```bash
python model_dl.py
```

This will download the files into the `detector/model/` folder.

---

## 🧠 How It Works

### Fake News Generator

- Utilizes GPT-2 with category-specific prompts defined in `labels.json`
- Generates short, tweet-style headlines based on topic

### Fake News Detector

- A fine-tuned BERT model classifies input headlines into `real` or `fake`
- The prediction logic is handled in `predict.py`

---

## 🧪 Run the App

Launch the Streamlit interface:

```bash
streamlit run app.py
```

You'll see two tabs:

1. **Generate + Detect** – Generate fake news and optionally detect
2. **Detect Headline** – Manually test any headline for authenticity

---

## 🗂️ Categories for Headline Generation

- 🌐 World Affairs  
- 🏛️ Politics and Government  
- 📈 Business and Economy  
- 🧪 Science and Health  
- 🧑‍💻 Education and Tech  
- 🕵️‍♂️ Crime  

---

## 📬 Contact

Created by [Vibhanshu Singh](https://github.com/vibhanshu-s)

- 📧 Email: payash2404@gmail.com  
- 🌐 GitHub: [@vibhanshu-s](https://github.com/vibhanshu-s)

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).
