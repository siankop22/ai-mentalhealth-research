#!/usr/bin/env python3
"""
split_burmese_data.py
Create stratified train/dev/test splits for Burmese/Zomi data.

Usage:
  python scripts/split_burmese_data.py \
      --in_path data/burmese/burmese_sample_clean.csv \
      --out_dir data/burmese/splits \
      --test_size 0.1 --dev_size 0.1

Notes:
 - Splits are stratified by label.
 - Requires scikit-learn (`pip install scikit-learn`).
"""

import argparse
import os
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", default="data/burmese_sample_clean.csv")
    ap.add_argument("--out_dir", default="data/splits")
    ap.add_argument("--test_size", type=float, default=0.1)
    ap.add_argument("--dev_size", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

def stratified_splits(df, label_col, test_size, dev_size, seed=42):
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx_train_dev, idx_test = next(sss1.split(df, df[label_col]))
    train_dev = df.iloc[idx_train_dev].reset_index(drop=True)
    test = df.iloc[idx_test].reset_index(drop=True)

    dev_ratio = dev_size / (1 - test_size)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=dev_ratio, random_state=seed)
    idx_train, idx_dev = next(sss2.split(train_dev, train_dev[label_col]))
    train = train_dev.iloc[idx_train].reset_index(drop=True)
    dev = train_dev.iloc[idx_dev].reset_index(drop=True)
    return train, dev, test

def main():
    args = parse_args()
    df = pd.read_csv(args.in_path)
    if "label" not in df.columns:
        raise ValueError("Expected 'label' column in input CSV.")
    os.makedirs(args.out_dir, exist_ok=True)
    train, dev, test = stratified_splits(df, "label", args.test_size, args.dev_size, args.seed)

    train_path = os.path.join(args.out_dir, "train.csv")
    dev_path = os.path.join(args.out_dir, "dev.csv")
    test_path = os.path.join(args.out_dir, "test.csv")

    train.to_csv(train_path, index=False, encoding="utf-8")
    dev.to_csv(dev_path, index=False, encoding="utf-8")
    test.to_csv(test_path, index=False, encoding="utf-8")

    def stats(name, d):
        counts = d["label"].value_counts(normalize=True).round(3)
        print(f"{name}: n={len(d)} | label ratio=\n{counts}\n")

    stats("TRAIN", train)
    stats("DEV", dev)
    stats("TEST", test)

if __name__ == "__main__":
    main()
