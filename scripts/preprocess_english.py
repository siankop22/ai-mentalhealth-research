import re, os, json, random
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

random.seed(42)

URL_RE = re.compile(r"https?://\S+|www\.\S+")
USR_RE = re.compile(r"(@|u/)[\w_]+")
WS_RE  = re.compile(r"\s+")

def clean_text(t: str) -> str:
    if not isinstance(t, str):
        return ""
    t = URL_RE.sub(" ", t)
    t = USR_RE.sub(" ", t)
    t = t.replace("\u200b", " ")
    t = WS_RE.sub(" ", t).strip()
    return t

def load_raw(path):
    # Expect a CSV with columns: id,text,label (you can adapt here)
    df = pd.read_csv(path)
    if "text" not in df or "label" not in df:
        raise ValueError("Expected columns: text,label (and optional id)")
    return df

def dedup(df):
    before = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"Deduped: {before} -> {len(df)}")
    return df

def preprocess(df):
    df["text"] = df["text"].astype(str).apply(clean_text)
    df = df[df["text"].str.len() >= 5]  # drop ultra-short
    df = df.dropna(subset=["label"])
    return df

def stratified_split(df, test_size=0.1, dev_size=0.1, label_col="label"):
    # First split off test, then split remaining into train/dev
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    idx_train_dev, idx_test = next(sss1.split(df, df[label_col]))
    train_dev = df.iloc[idx_train_dev].reset_index(drop=True)
    test = df.iloc[idx_test].reset_index(drop=True)

    dev_ratio = dev_size / (1 - test_size)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=dev_ratio, random_state=42)
    idx_train, idx_dev = next(sss2.split(train_dev, train_dev[label_col]))
    train = train_dev.iloc[idx_train].reset_index(drop=True)
    dev = train_dev.iloc[idx_dev].reset_index(drop=True)
    return train, dev, test

def main():
    in_path = "data/english/raw/english_raw.csv"  # adjust if needed
    out_dir = "data/english"
    os.makedirs(f"{out_dir}/clean", exist_ok=True)
    os.makedirs(f"{out_dir}/splits", exist_ok=True)

    df = load_raw(in_path)
    df = dedup(df)
    df = preprocess(df)

    df.to_csv(f"{out_dir}/clean/english_clean.csv", index=False)

    train, dev, test = stratified_split(df, test_size=0.1, dev_size=0.1, label_col="label")
    train.to_csv(f"{out_dir}/splits/train.csv", index=False)
    dev.to_csv(f"{out_dir}/splits/dev.csv", index=False)
    test.to_csv(f"{out_dir}/splits/test.csv", index=False)

    # Quick report
    def stats(name, d):
        print(f"{name}: n={len(d)} | label counts=\n{d['label'].value_counts(normalize=True).round(3)}\n")
    stats("TRAIN", train); stats("DEV", dev); stats("TEST", test)

if __name__ == "__main__":
    main()
