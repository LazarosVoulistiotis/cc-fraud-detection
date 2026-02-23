# src/16_thresholding_and_metrics.py
"""
Week 16 — Final Thresholding & Metrics (VAL-selected threshold, TEST locked)

What this script does:
1) Load trained model (.joblib) and Week8 engineered splits (train/val/test).
2) Compute probabilities on VAL and TEST (no threshold tuning on test).
3) Select threshold on VAL using one of:
   - max_f1
   - max_f2 (beta=2)
   - precision_constraint (precision >= p, choose smallest threshold that satisfies it)
   - cost_based (minimize cost = fp_cost*FP + fn_cost*FN)
4) Evaluate on TEST at the selected threshold.
5) Save:
   - PR curve (VAL) + chosen point
   - Confusion matrix (TEST)
   - metrics.json + metrics_table.csv
   - threshold_candidates.csv (optional)

Example (Git Bash):
python src/16_thresholding_and_metrics.py \
  --model-path models/xgb_week8.joblib \
  --data-train data/data_interim/splits_week8/train.csv \
  --data-val   data/data_interim/splits_week8/val.csv \
  --data-test  data/data_interim/splits_week8/test.csv \
  --target-column Class \
  --policy precision_constraint --precision-min 0.80 \
  --figdir reports/figures/week16/precision_constraint_p80 \
  --outdir reports/week16_thresholding/precision_constraint_p80 \
  --seed 42
"""
import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_score,
    recall_score,
    f1_score,
)

# Config dataclass for command-line arguments
@dataclass
class Config:
    model_path: str
    data_train: str
    data_val: str
    data_test: str
    target_column: str
    policy: str
    beta: float
    precision_min: float
    fp_cost: float
    fn_cost: float
    figdir: str
    outdir: str
    seed: int

# CLI interface
def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Week 16 — Final thresholding & metrics (VAL threshold, TEST locked).")
    p.add_argument("--model-path", required=True)
    p.add_argument("--data-train", required=True)  # only for schema check; not used for thresholding
    p.add_argument("--data-val", required=True)
    p.add_argument("--data-test", required=True)
    p.add_argument("--target-column", default="Class")

    p.add_argument("--policy", default="cost_based",
                   choices=["cost_based", "max_f1", "max_fbeta", "precision_constraint"],
                   help="Threshold selection on VAL.")
    p.add_argument("--beta", type=float, default=2.0, help="Used when policy=max_fbeta (default beta=2).")
    p.add_argument("--precision-min", type=float, default=0.80, help="Used when policy=precision_constraint.")
    p.add_argument("--fp-cost", type=float, default=1.0, help="Used when policy=cost_based.")
    p.add_argument("--fn-cost", type=float, default=20.0, help="Used when policy=cost_based.")

    p.add_argument("--figdir", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    return Config(
        model_path=args.model_path,
        data_train=args.data_train,
        data_val=args.data_val,
        data_test=args.data_test,
        target_column=args.target_column,
        policy=args.policy,
        beta=args.beta,
        precision_min=args.precision_min,
        fp_cost=args.fp_cost,
        fn_cost=args.fn_cost,
        figdir=args.figdir,
        outdir=args.outdir,
        seed=args.seed,
    )

# Utility functions
def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)

# Get feature columns
def get_features(train_df: pd.DataFrame, target: str) -> list: # Το train είναι το “source of truth” για το τι features περιμένει το model. Αποφεύγεις mismatch errors τύπου “feature names mismatch” που είχες ήδη συναντήσει.
    feats = [c for c in train_df.columns if c != target]
    if not feats:
        raise ValueError("No features found.")
    return feats 

# proba_fraud(model, X): παίρνει score = P(fraud)
def proba_fraud(model, X: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(X)
    proba = np.asarray(proba)
    if proba.ndim != 2 or proba.shape[1] != 2:
        raise ValueError(f"predict_proba returned shape {proba.shape}, expected [n,2].")
    return proba[:, 1] # Αυτό είναι το “y_score” που χρησιμοποιείται σε PR curve, AUCs και thresholding.

# γενική Fβ συνάρτηση, Fβ = (1 + β^2) * ((P * R) / (β^2 * P) + R)
def fbeta(precision: np.ndarray, recall: np.ndarray, beta: float) -> np.ndarray:
    b2 = beta * beta
    denom = (b2 * precision + recall)
    # avoid division-by-zero
    denom = np.where(denom == 0, np.nan, denom)
    return (1 + b2) * (precision * recall) / denom

# threshold_sweep(y_true, y_score): φτιάχνει πίνακα threshold→metrics (στο VAL)
def threshold_sweep(y_true: np.ndarray, y_score: np.ndarray) -> pd.DataFrame:
    # Note: sklearn returns precision/recall arrays of length len(thresholds)+1
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    # Align to thresholds: drop the last precision/recall which has no threshold
    precision_t = precision[:-1]
    recall_t = recall[:-1]

    f1 = 2 * (precision_t * recall_t) / np.where((precision_t + recall_t) == 0, np.nan, (precision_t + recall_t))
    f2 = fbeta(precision_t, recall_t, beta=2.0)

    df = pd.DataFrame({
        "threshold": thresholds,
        "precision": precision_t,
        "recall": recall_t,
        "f1": f1,
        "f2": f2,
    })
    return df

# confusion_at_threshold(): (TP,FP,TN,FN) για συγκεκριμένο threshold
def confusion_at_threshold(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> Tuple[int, int, int, int]:
    y_pred = (y_score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp, fp, tn, fn

# select_threshold(): η καρδιά του “policy”
def select_threshold(cfg: Config, sweep: pd.DataFrame, y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, Dict]:
    policy = cfg.policy

    if policy == "max_f1":
        row = sweep.loc[sweep["f1"].idxmax()]
        thr = float(row["threshold"])
        meta = {"policy": policy, "criterion": "max_f1", "selected_row": row.to_dict()}

    elif policy == "max_fbeta":
        # recompute fbeta with chosen beta
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        precision_t = precision[:-1]
        recall_t = recall[:-1]
        f_b = fbeta(precision_t, recall_t, beta=cfg.beta)
        idx = int(np.nanargmax(f_b))
        thr = float(thresholds[idx])
        meta = {"policy": policy, "criterion": f"max_fbeta(beta={cfg.beta})",
                "precision": float(precision_t[idx]), "recall": float(recall_t[idx]), "f_beta": float(f_b[idx])}

    elif policy == "precision_constraint":
        # choose the smallest threshold that achieves precision >= precision_min
        ok = sweep[sweep["precision"] >= cfg.precision_min].sort_values("threshold")
        if ok.empty:
            # fallback: choose max precision (still report it)
            row = sweep.loc[sweep["precision"].idxmax()]
            thr = float(row["threshold"])
            meta = {"policy": policy, "criterion": f"precision>={cfg.precision_min} not achievable; fallback=max_precision",
                    "selected_row": row.to_dict()}
        else:
            row = ok.iloc[0]
            thr = float(row["threshold"])
            meta = {"policy": policy, "criterion": f"precision_constraint(>= {cfg.precision_min})",
                    "selected_row": row.to_dict()}

    elif policy == "cost_based":
        # cost = fp_cost*FP + fn_cost*FN
        costs = []
        for thr in sweep["threshold"].values:
            tp, fp, tn, fn = confusion_at_threshold(y_true, y_score, float(thr))
            cost = cfg.fp_cost * fp + cfg.fn_cost * fn
            costs.append(cost)
        sweep2 = sweep.copy()
        sweep2["cost"] = costs
        row = sweep2.loc[sweep2["cost"].idxmin()]
        thr = float(row["threshold"])
        meta = {"policy": policy, "criterion": f"min_cost(fp={cfg.fp_cost}, fn={cfg.fn_cost})",
                "selected_row": row.to_dict()}
        sweep[:] = sweep2  # keep cost column for export

    else:
        raise ValueError(f"Unknown policy: {policy}")

    return thr, meta

# τελικά metrics σε TEST για το επιλεγμένο threshold
def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> Dict:
    y_pred = (y_score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # F2 explicitly
    b = 2.0
    f2 = (1 + b*b) * (prec * rec) / max((b*b * prec + rec), 1e-12)

    mcc = matthews_corrcoef(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)

    return {
        "threshold": float(thr),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "f2": float(f2),
        "mcc": float(mcc),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "n": int(len(y_true)),
        "pos": int(y_true.sum()),
        "neg": int(len(y_true) - y_true.sum()),
    }

# PR curve στο VAL + σημείο του threshold
def plot_pr_curve(y_true: np.ndarray, y_score: np.ndarray, thr: float, outpath: str) -> None:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve (Validation)")

    # mark chosen point
    # Find closest threshold index
    if len(thresholds) > 0:
        idx = int(np.argmin(np.abs(thresholds - thr)))
        # recall/precision aligned to thresholds are precision[:-1], recall[:-1]
        p_pt = precision[:-1][idx]
        r_pt = recall[:-1][idx]
        plt.scatter([r_pt], [p_pt])
        plt.annotate(f"thr={thr:.4f}", (r_pt, p_pt), textcoords="offset points", xytext=(10, -10))

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

# Confusion matrix στο TEST με το επιλεγμένο threshold
def plot_confusion(tp: int, fp: int, tn: int, fn: int, outpath: str) -> None:
    cm = np.array([[tn, fp], [fn, tp]])
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix (Test)")
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])

    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

# Main function
def main() -> None:
    cfg = parse_args()
    ensure_dirs(cfg.figdir, cfg.outdir)
    np.random.seed(cfg.seed)

    model = joblib.load(cfg.model_path)
    train_df = pd.read_csv(cfg.data_train)
    val_df = pd.read_csv(cfg.data_val)
    test_df = pd.read_csv(cfg.data_test)

    if cfg.target_column not in val_df.columns or cfg.target_column not in test_df.columns:
        raise ValueError(f"Target column '{cfg.target_column}' not found.")

    features = get_features(train_df, cfg.target_column)

    X_val = val_df[features].to_numpy()
    y_val = val_df[cfg.target_column].astype(int).to_numpy()

    X_test = test_df[features].to_numpy()
    y_test = test_df[cfg.target_column].astype(int).to_numpy()

    s_val = proba_fraud(model, X_val)
    s_test = proba_fraud(model, X_test)

    sweep = threshold_sweep(y_val, s_val)
    thr, selection_meta = select_threshold(cfg, sweep, y_val, s_val)

    # Evaluate on TEST (locked)
    metrics_test = compute_metrics(y_test, s_test, thr)

    # Save artifacts
    with open(os.path.join(cfg.outdir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    with open(os.path.join(cfg.outdir, "threshold_selection.json"), "w", encoding="utf-8") as f:
        json.dump(selection_meta, f, indent=2)

    with open(os.path.join(cfg.outdir, "final_metrics_test.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_test, f, indent=2)

    # table (single row)
    pd.DataFrame([metrics_test]).to_csv(os.path.join(cfg.outdir, "final_metrics_test.csv"), index=False)

    # export sweep
    sweep.to_csv(os.path.join(cfg.outdir, "val_threshold_sweep.csv"), index=False)

    # plots
    plot_pr_curve(y_val, s_val, thr, os.path.join(cfg.figdir, "pr_curve_val_with_threshold.png"))
    plot_confusion(metrics_test["tp"], metrics_test["fp"], metrics_test["tn"], metrics_test["fn"],
                   os.path.join(cfg.figdir, "confusion_matrix_test.png"))

    print("[OK] Threshold selected on VAL and evaluated on TEST (locked).")
    print(f"Policy: {cfg.policy} | Selected threshold: {thr:.6f}")
    print("Saved:")
    print(f"- Figures: {cfg.figdir}")
    print(f"- Outputs: {cfg.outdir}")


if __name__ == "__main__":
    main()
