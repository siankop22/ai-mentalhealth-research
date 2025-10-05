#!/usr/bin/env python3
"""
qa_burmese_data.py
Quality checks for data/burmese/burmese_sample.csv and optional cleaning.

Usage:
  python scripts/qa_burmese_data.py       --in_path data/burmese/burmese_sample.csv       --out_clean data/burmese/burmese_sample_clean.csv       --report_path reports/qa_burmese_data_report.md

The script:
 - Verifies required columns
 - Checks empty fields
 - Validates label ∈ {distress, neutral}
 - Validates language ∈ {my, zom, en-my}
 - Detects duplicate ids and duplicate texts
 - Checks reasonable text length (5..500 chars by default)
 - Validates collection_date parseability (YYYY-MM-DD)
 - Writes a Markdown report and (optionally) a cleaned CSV
"""

import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd

REQUIRED_COLS = [
    "id", "text", "language", "label", "source",
    "license", "collection_date", "split", "translation_of",
    "collector", "notes"
]

VALID_LABELS = {"distress", "neutral"}
VALID_LANGS = {"my", "zom", "en-my"}

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", default="data/burmese/burmese_sample.csv")
    ap.add_argument("--out_clean", default="data/burmese/burmese_sample_clean.csv")
    ap.add_argument("--report_path", default="reports/qa_burmese_data_report.md")
    ap.add_argument("--min_len", type=int, default=5)
    ap.add_argument("--max_len", type=int, default=500)
    return ap.parse_args()

def validate_date(s) -> bool:
    try:
        datetime.strptime(str(s), "%Y-%m-%d")
        return True
    except Exception:
        return False

def main():
    args = parse_args()
    df = pd.read_csv(args.in_path)
    report_lines = []
    ok = True

    # Column presence
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        ok = False
        report_lines.append(f"**Missing required columns:** {missing_cols}")
    else:
        report_lines.append("All required columns present.")

    if "text" in df.columns:
        # Empty text
        empty_text_idx = df.index[df["text"].astype(str).str.strip().eq("")].tolist()
        if empty_text_idx:
            ok = False
            report_lines.append(f"**Empty text rows:** {len(empty_text_idx)} (indices: {empty_text_idx[:15]} ...)")
        # Length checks
        lens = df["text"].astype(str).str.len()
        too_short = df.index[lens < args.min_len].tolist()
        too_long = df.index[lens > args.max_len].tolist()
        if too_short:
            ok = False
            report_lines.append(f"**Too short texts (<{args.min_len}):** {len(too_short)} (samples: {too_short[:15]} ...)")
        if too_long:
            ok = False
            report_lines.append(f"**Too long texts (>{args.max_len}):** {len(too_long)} (samples: {too_long[:15]} ...)")

    # Label validity
    if "label" in df.columns:
        bad_labels = df.index[~df["label"].isin(VALID_LABELS)].tolist()
        if bad_labels:
            ok = False
            uniq = sorted(set(df.loc[bad_labels, "label"].astype(str)))
            report_lines.append(f"**Invalid labels:** {uniq} at rows {bad_labels[:15]} ...")
        # Label counts
        report_lines.append("**Label distribution:**\n" + df["label"].value_counts().to_string())

    # Language validity
    if "language" in df.columns:
        bad_langs = df.index[~df["language"].isin(VALID_LANGS)].tolist()
        if bad_langs:
            ok = False
            uniq = sorted(set(df.loc[bad_langs, "language"].astype(str)))
            report_lines.append(f"**Invalid language tags:** {uniq} at rows {bad_langs[:15]} ...")
        # Language counts
        report_lines.append("**Language distribution:**\n" + df["language"].value_counts().to_string())

    # Duplicate IDs
    if "id" in df.columns:
        dup_id_mask = df["id"].duplicated(keep=False)
        dup_ids = df.loc[dup_id_mask, "id"].tolist()
        if dup_ids:
            ok = False
            report_lines.append(f"**Duplicate IDs:** {len(set(dup_ids))} duplicates.")

    # Duplicate texts
    if "text" in df.columns:
        dup_text_mask = df["text"].duplicated(keep=False)
        dup_texts = df.loc[dup_text_mask, "text"].tolist()
        if dup_texts:
            ok = False
            report_lines.append(f"**Duplicate texts:** {len(set(dup_texts))} duplicates.")

    # Date validity
    if "collection_date" in df.columns:
        bad_dates_idx = df.index[~df["collection_date"].apply(validate_date)].tolist()
        if bad_dates_idx:
            ok = False
            report_lines.append(f"**Invalid collection_date values:** rows {bad_dates_idx[:15]} ...")

    # Write report
    report_lines.insert(0, f"# QA Report for {args.in_path}\n")
    report_lines.append(f"\n**Overall status:** {'PASS ✅' if ok else 'REVIEW ❗'}\n")
    Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(report_lines))

    # Optional cleaning: drop duplicates and overly long/short, trim whitespace
    if not df.empty:
        if "text" in df.columns:
            df["text"] = df["text"].astype(str).str.replace("\u200b", "", regex=False).str.strip()
            df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
        if "id" in df.columns:
            df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)
        if "label" in df.columns:
            df = df[df["label"].isin(VALID_LABELS)]
        if "language" in df.columns:
            df = df[df["language"].isin(VALID_LANGS)]
        if "text" in df.columns:
            lens = df["text"].astype(str).str.len()
            df = df[(lens >= args.min_len) & (lens <= args.max_len)]
        df.to_csv(args.out_clean, index=False, encoding="utf-8")

    print(f"Wrote QA report to {args.report_path}")
    print(f"Wrote cleaned CSV to {args.out_clean}")

if __name__ == "__main__":
    main()