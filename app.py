import os
import time
import csv
import torch
import streamlit as st
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "xlmr-burmese")  # portable path

# ---------- Streamlit Config ----------
st.set_page_config(page_title="Burmese/Zomi Classifier", page_icon="🧠")
st.title("🧠 Burmese/Zomi Mental Health Classifier")
st.write("Analyze whether a short text expresses emotional distress or is neutral.")
st.caption("အသံထွက်ကောင်းဖို့ စာတန်းတို (၁–၃) လောက်ရိုက်ပါ | Use short 1–3 sentences.")

# ---------- Load Model ----------
@st.cache_resource(show_spinner=False)
def load_model_and_labels(model_dir: str):
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)

    # detect label order
    try:
        id2label = {int(k): v for k, v in mdl.config.id2label.items()}
    except Exception:
        id2label = {0: "neutral", 1: "distress"}

    distress_idx = next((i for i, name in id2label.items() if str(name).lower().startswith("distress")), 1)
    neutral_idx = 0 if distress_idx == 1 else 1
    return tok, mdl, id2label, distress_idx, neutral_idx


with st.spinner("Loading model, please wait..."):
    try:
        tokenizer, model, id2label, DISTRESS_IDX, NEUTRAL_IDX = load_model_and_labels(MODEL_PATH)
        st.success(f"✅ Model loaded. Labels: {id2label} (distress_index={DISTRESS_IDX})")
    except Exception as e:
        st.error(f"❌ Failed to load model from `{MODEL_PATH}`.\n\nError: {e}")
        st.stop()

# ---------- Demo Button ----------
if "demo" not in st.session_state:
    st.session_state["demo"] = ""
if st.button("Try sample text"):
    st.session_state["demo"] = "ဒီနေ့ အလုပ်တွေကို လုံးဝ handle မရတော့ဘူး။ စိတ်အလွန်ပင်ပန်းနေတယ်။"

# ---------- Text Area (Single) ----------
text = st.text_area(
    "Enter Burmese or Zomi text:",
    value=st.session_state.get("demo", ""),
    height=120,
    key="main_textarea",  # ensures unique ID
)

# ---------- Analyze ----------
if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=1)[0].tolist()

        # --- Threshold-based classification ---
        THR = 0.45  # tweak for sensitivity
        p_distress = probs[DISTRESS_IDX]
        p_neutral = probs[NEUTRAL_IDX]
        pred_label = 1 if p_distress >= THR else 0
        label_text = "Distress 😔" if pred_label == 1 else "Neutral 🙂"

        # --- Display results ---
        st.caption(f"Decision threshold = {THR:.2f}")
        st.markdown(f"**Neutral:** {p_neutral:.2%} | **Distress:** {p_distress:.2%}")
        st.success(f"Prediction: **{label_text}**")

        # --- Logging (only after analyze) ---
        Path("reports/app_logs").mkdir(parents=True, exist_ok=True)
        with open("reports/app_logs/predictions.csv", "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([time.time(), text, p_neutral, p_distress, THR, label_text])
