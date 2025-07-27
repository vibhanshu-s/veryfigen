# VeryFiGen: AI-Based Fake News Headline Generator and Detector

## Abstract

Fake news has become a critical issue in the digital era, with profound effects on public opinion, democracy, and trust. This paper presents **VeryFiGen**, an integrated system that both **generates** and **detects** fake news headlines. It combines the natural language generation capabilities of **GPT-2** with the classification strength of a **fine-tuned BERT** model. The system aims to provide a tool for research and education about misinformation while offering practical insights into NLP-based detection.

---

## 1. Introduction

With the rise of social media and decentralized news distribution, distinguishing between real and fake news has become increasingly difficult. NLP techniques provide powerful tools for both detecting fake news and studying its propagation. VeryFiGen aims to contribute to this domain by:

- Generating realistic fake news headlines using GPT-2.
- Detecting fake headlines using a fine-tuned BERT classifier.

---

## 2. System Overview

VeryFiGen consists of two major components:

### 2.1 Headline Generator (GPT-2)

- Utilizes Hugging Face’s `GPT-2` model.
- Conditioned on domain-specific prompts (e.g., "BREAKING: Scientists have found").
- Generates realistic-looking headlines in six categories:
  - Science and Health
  - Politics and Government
  - World Affairs
  - Business and Economy
  - Education and Tech
  - Crime

### 2.2 Headline Detector (BERT)

- Fine-tuned version of `bert-base-uncased`.
- Trained on a labeled dataset (`text`, `label`), where `label = 1` for real and `0` for fake.
- Accepts user input and returns prediction (Real/Fake).

---

## 3. Technologies Used

| Technology        | Role                                      |
|------------------|-------------------------------------------|
| Python           | Core programming language                 |
| PyTorch          | Model training and inference              |
| HuggingFace      | Transformers library for GPT-2 & BERT     |
| Streamlit        | Interactive web UI                        |
| Google Drive     | Hosting models over 100MB                 |
| GitHub Actions   | Version control and collaboration         |

---

## 4. Model Training

### BERT Fine-Tuning

- Pretrained `bert-base-uncased` used.
- Fine-tuned on custom dataset of real and fake headlines.
- Achieved high classification accuracy (reported in model logs).

### GPT-2 Conditioning

- `GPT2LMHeadModel` used to generate text.
- Prompts structured by category and randomly selected at runtime.

---

## 5. Streamlit App Functionality

The web app consists of two tabs:

### Tab 1: Generate + Detect
- User selects a category.
- A fake news headline is generated.
- Optionally, the detector can classify it as real/fake.

### Tab 2: Detect Headline
- User inputs a custom headline (up to 30 words).
- Detector returns prediction (`Real` / `Fake`).

---

## 6. Deployment Strategy

- Large model files (>100MB) hosted on **Google Drive**.
- Python script `model_dl.py` downloads models into `detector/`.
- Models used:
  - `bert-fine-tuned` → for classification
  - GPT-2 (downloaded at runtime via Hugging Face)

---

## 7. Limitations and Future Work

- Only headlines are used (not full articles).
- Detection is binary; confidence scores could improve usability.
- Expansion to multilingual support is possible using `mBERT` or `XLM-R`.

---

## 8. References

1. Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. [**BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**](https://arxiv.org/abs/1810.04805). arXiv:1810.04805 (2018).
2. Alec Radford et al. [**Language Models are Unsupervised Multitask Learners (GPT-2 Paper)**](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf). OpenAI, 2019.
3. HuggingFace Transformers Documentation: https://huggingface.co/docs/transformers/
4. Streamlit Documentation: https://docs.streamlit.io/

---

## 9. How to Run

```bash
git clone https://github.com/vibhanshu-s/veryfigen
cd veryfigen
pip install -r requirements.txt
python model_dl.py   # downloads the models
streamlit run app.py
