#!/usr/bin/env python
# -*- coding: utf-8 -*-

""""
Τρέξιμο (ένα “canonical” run):

python src/08_2_make_splits.py \
  --data data/data_interim/creditcard_features_week8.csv \
  --outdir data/data_interim/splits_week8 \
  --target Class \
  --test-size 0.20 \
  --val-size 0.10 \
  --seed 42 \
  --drop-duplicates

"""

import argparse, json, hashlib
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def save_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="data/data_raw/creditcard.csv")
    ap.add_argument("--outdir", default="data/data_interim")
    ap.add_argument("--target", default="Class")
    ap.add_argument("--test-size", type=float, default=0.20)
    ap.add_argument("--val-size", type=float, default=0.00, help="0.00 means no val")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drop-duplicates", action="store_true")
    args = ap.parse_args()

    data_path = Path(args.data)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    if args.drop_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)

    if args.target not in df.columns:
        raise KeyError(f"Target '{args.target}' not found in CSV.")

    # κρατάμε μόνο numeric features (όπως δουλεύεις ήδη)
    y = df[args.target].astype(int)
    X = df.drop(columns=[args.target]).select_dtypes(include=["number"]).copy()

    # 1) split out test (locked)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )

    # 2) optional val split from trainval
    if args.val_size and args.val_size > 0:
        # val_size is fraction of FULL dataset; convert to fraction of trainval
        val_frac = args.val_size / (1.0 - args.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_frac, stratify=y_trainval, random_state=args.seed
        )
    else:
        X_train, y_train = X_trainval, y_trainval
        X_val, y_val = None, None

    # save CSVs (features + target)
    train_df = X_train.copy()
    train_df[args.target] = y_train.values
    save_csv(train_df, outdir / "train.csv")

    test_df = X_test.copy()
    test_df[args.target] = y_test.values
    save_csv(test_df, outdir / "test.csv")

    if X_val is not None:
        val_df = X_val.copy()
        val_df[args.target] = y_val.values
        save_csv(val_df, outdir / "val.csv")

    meta = {
        "source_csv": str(data_path.as_posix()),
        "source_sha256": sha256_file(data_path),
        "target": args.target,
        "seed": args.seed,
        "test_size": args.test_size,
        "val_size": args.val_size,
        "rows_total": int(len(df)),
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "fraud_rate_total": float(y.mean()),
        "fraud_rate_train": float(y_train.mean()),
        "fraud_rate_test": float(y_test.mean()),
        "features_count": int(X.shape[1]),
        "features": list(X.columns),
    }
    if X_val is not None:
        meta["rows_val"] = int(len(val_df))
        meta["fraud_rate_val"] = float(y_val.mean())

    (outdir / "split_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Saved:", outdir / "train.csv", outdir / "test.csv")
    if X_val is not None:
        print("Saved:", outdir / "val.csv")
    print("Saved:", outdir / "split_meta.json")

if __name__ == "__main__":
    main()
