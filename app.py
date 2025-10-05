import os
import time
import csv
from pathlib import Path

import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ============== Paths ==============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "xlmr-burmese")
LOG_DIR = Path(BASE_DIR) / "reports" / "app_logs"
LOG_FILE = LOG_DIR / "predictions.csv"

# ============== Streamlit config ==============
st.set_page_config(page_title="Burmese/Zomi Classifier", page_icon="🧠")
st.title("🧠 Burmese/Zomi Mental Health Classifier")
st.write("Analyze whether a short text expresses emotional distress or is neutral.")
st.caption("အသံထွက်ကောင်းဖို့ စာတန်းတို (၁–၃) လောက်ရိုက်ပါ | Use short 1–3 sentences.")

# ============== Load model (cached) ==============
@st.cache_resource(show_spinner=False)
def load_model_and_labels(model_dir: str):
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)

    # Detect label order robustly
    try:
        id2label = {int(k): v for k, v in mdl.config.id2label.items()}
    except Exception:
        id2label = {0: "neutral", 1: "distress"}

    # Find distress index; default to 1 if not found
    distress_idx = next((i for i, name in id2label.items()
                         if str(name).lower().startswith("distress")), 1)
    neutral_idx = 0 if distress_idx == 1 else 1
    return tok, mdl, id2label, distress_idx, neutral_idx

with st.spinner("Loading model, please wait..."):
    try:
        tokenizer, model, id2label, DISTRESS_IDX, NEUTRAL_IDX = load_model_and_labels(MODEL_PATH)
        st.success(f"✅ Model loaded. Labels: {id2label} (distress_index={DISTRESS_IDX})")
    except Exception as e:
        st.error(f"❌ Failed to load model from `{MODEL_PATH}`.\n\nError: {e}")
        st.stop()

# ============== Session state & demo helpers ==============
# Initialize the single textarea's state key
if "main_textarea" not in st.session_state:
    st.session_state["main_textarea"] = ""

# Buttons row
c1, c2, _ = st.columns([1, 1, 2])
with c1:
    if st.button("Try sample"):
        st.session_state["main_textarea"] = "ဒီနေ့ အလုပ်တွေကို လုံးဝ handle မရတော့ဘူး။ စိတ်အလွန်ပင်ပန်းနေတယ်။"
        st.rerun()
with c2:
    if st.button("Clear"):
        st.session_state["main_textarea"] = ""
        st.rerun()

# ---- Single text area bound to the key (only ONE) ----
text = st.text_area(
    "Enter Burmese or Zomi text:",
    key="main_textarea",      # binds widget value to session state
    height=140,
)

# Threshold slider
THR = st.slider("Adjust decision threshold (higher = stricter distress)",
                0.10, 0.90, 0.45, 0.05)

# ============== Analyze ==============
if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=1)[0].tolist()

        p_distress = probs[DISTRESS_IDX]
        p_neutral  = probs[NEUTRAL_IDX]
        pred_label = 1 if p_distress >= THR else 0
        label_text = "Distress 😔" if pred_label == 1 else "Neutral 🙂"

        st.caption(f"Decision threshold = {THR:.2f}")
        st.markdown(f"**Neutral:** {p_neutral:.2%} | **Distress:** {p_distress:.2%}")
        st.success(f"Prediction: **{label_text}**")

        # ============== Logging (append) ==============
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        new_file = not LOG_FILE.exists()
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["ts", "text", "p_neutral", "p_distress", "threshold", "label"])
            w.writerow([time.time(), text, p_neutral, p_distress, THR, label_text])

# ============== Download logs ==============
if LOG_FILE.exists():
    import pandas as pd
    df_logs = pd.read_csv(LOG_FILE)
    st.download_button(
        "Download prediction logs (CSV)",
        data=df_logs.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv",
        help="All predictions recorded from this app session."
    )
