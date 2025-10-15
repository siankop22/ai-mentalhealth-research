# AI for Mental Health Research
**Multilingual NLP for emotion & distress signals across English, Burmese, and Zomi**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](#)
[![HuggingFace Models](https://img.shields.io/badge/🤗-Transformers-orange.svg)](#)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/siankop22/ai-mentalhealth-research/blob/main/notebooks/demo_inference.ipynb)

> **Goal:** Build transparent, culturally-aware NLP models that identify affective signals (stress, depression, burnout) in multilingual text, with a focus on under-resourced languages.

---

## 🔍 Highlights
- **Multilingual pipelines** for English, Burmese (မြန်မာ), and Zomi.
- **Transfer learning** with BERT/RoBERTa; configurable training via YAML.
- **Reproducible eval** (precision/recall/F1, macro/micro, per-group slices).
- **Ethics & safety** guardrails, model cards, and bias diagnostics.
- **One-click demo** notebook for quick inference.

---

## 📁 Project Structure
ai-mentalhealth-research/
├─ data/ # small samples or scripts to fetch data
├─ notebooks/
│ ├─ demo_inference.ipynb # quick portfolio demo
│ └─ training_walkthrough.ipynb
├─ src/
│ ├─ configs/ # .yaml training/eval configs
│ ├─ data/ # loaders, preprocessing, tokenization
│ ├─ models/ # model wrappers, heads, losses
│ ├─ training/ # loops, schedulers, logging
│ └─ evaluation/ # metrics, slice analysis, confusion matrices
├─ assets/ # screenshots, plots, social preview
├─ tests/ # unit tests (pytest)
├─ docs/ # site (GitHub Pages)
├─ LICENSE
├─ CITATION.cff
├─ CODE_OF_CONDUCT.md
├─ CONTRIBUTING.md
├─ SECURITY.md
└─ README.md


---

## 🚀 Quickstart
```bash
# 1) Create env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Run a quick inference demo
python -m src.models.infer \
  --model_name "bert-base-multilingual-cased" \
  --text "I feel overwhelmed and exhausted lately."

# 3) Train (example config)
python -m src.training.run --config src/configs/eng_bert.yaml

