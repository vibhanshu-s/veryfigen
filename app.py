import streamlit as st
import json
import csv
import os
from generator.generator import generate_fake_news
from detector.predict import predict_headline

# ========== CONFIG ==========
st.set_page_config(page_title="Fake News Generator & Detector", layout="centered")

# ========== HEADER ==========
st.markdown(
    "<h1 style='text-align: center;'>📰 AI Fake News System</h1>", 
    unsafe_allow_html=True
)
st.markdown("---")

# ========== TABS ==========
tab1, tab2 = st.tabs(["🛠️ Generate + Detect", "🔍 Detect Headline"])

# ========== TAB 1 ==========
with tab1:
    st.subheader("Generate a Fake News Headline")

    # Load label prompts
    with open("generator/labels.json", "r") as f:
        prompt_templates = json.load(f)
    labels = list(prompt_templates.keys())
    
    label = st.selectbox("Select a news category", labels)

    if st.button("🚀 Generate Fake Headline"):
        headline = generate_fake_news(label)
        
        st.markdown(f"#### 🎯 Generated Headline\n> **[{label}]** {headline}")

        # Automatically detect using BERT
        pred = predict_headline(headline)
        st.markdown(f"#### 🧠 BERT Prediction: `{pred.upper()}`")

        # Save to CSV
        os.makedirs("data", exist_ok=True)
        with open("data/fake_news_generated.csv", "a", encoding="utf-8", newline='') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(["text", "label"])
            writer.writerow([headline, "fake"])

# ========== TAB 2 ==========
with tab2:
    st.subheader("Paste a Headline for Fake News Detection")
    user_input = st.text_area("📝 Enter news headline (max 30 words):", max_chars=300, height=120)

    if st.button("🔍 Detect"):
        if len(user_input.strip()) == 0:
            st.warning("⚠️ Please enter a headline.")
        elif len(user_input.strip().split()) > 30:
            st.warning("⚠️ Headline should not exceed 30 words.")
        else:
            pred = predict_headline(user_input)
            st.markdown(f"#### 🔎 Prediction: `{pred.upper()}`")

# ========== FOOTER ==========
st.markdown("---")
st.markdown("<center>🔬 Built by Vibhanshu Singh using GPT-2 + BERT</center>", unsafe_allow_html=True)
