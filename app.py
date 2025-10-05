import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "xlmr-burmese")
st.set_page_config(page_title="Burmese/Zomi Classifier", page_icon="🧠")

MODEL_PATH = "/Users/tksiankop/Desktop/ai-mentalhealth-research/models/xlmr-burmese"


st.title("🧠 Burmese/Zomi Mental Health Classifier")
st.write("Analyze whether a short text expresses emotional distress or is neutral.")

@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    return tokenizer, model

with st.spinner("Loading model, please wait..."):
    try:
        tokenizer, model = load_model()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to load model from `{MODEL_PATH}`.\n\nError: {e}")
        st.stop()

text = st.text_area("Enter Burmese or Zomi text:", height=120)

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=1)[0].tolist()
        st.markdown(f"**Neutral:** {probs[0]:.2%} | **Distress:** {probs[1]:.2%}")
        label = "Distress 😔" if probs[1] > probs[0] else "Neutral 🙂"
        st.success(f"Prediction: **{label}**")

