#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Τρέξιμο (πάνω στα canonical Week8 splits)

python src/09_1_logreg_from_splits.py \
  --train-csv data/data_interim/splits_week8/train.csv \
  --val-csv data/data_interim/splits_week8/val.csv \
  --test-csv data/data_interim/splits_week8/test.csv \
  --target Class \
  --outdir reports/week9_logreg_week8 \
  --figdir reports/figures/week9_logreg_week8 \
  --model-path models/logreg_week8.joblib \
  --scaler standard \
  --class-weight balanced \
  --threshold 0.50 \
  --optimize cost \
  --n-thresholds 101 \
  --cost-fp 1 \
  --cost-fn 20

LogReg training/eval using LOCKED splits (train/val/test CSVs).
- Threshold tuning γίνεται στο VAL (business σωστό)
- Final evaluation γίνεται στο TEST (κλειδωμένο)
"""

from __future__ import annotations

import argparse, json, sys, time, subprocess, platform
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)
import joblib
import logging


# --------------------------- Utils ---------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def make_jsonable(obj):
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: make_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_jsonable(v) for v in obj]
    return obj

def get_scaler(name: str):
    if name == "standard":
        return StandardScaler()
    if name == "robust":
        return RobustScaler()
    return "passthrough"

def setup_logging(log_path: Path):
    fmt = "%(asctime)s %(levelname)s:%(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(log_path, encoding="utf-8")],
    )

def get_git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True
        ).strip()
    except Exception:
        return "unknown"

def environment_info():
    import numpy, pandas, sklearn
    return {
        "python": sys.version.replace("\n"," "),
        "os": platform.platform(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__,
    }

# --------------------------- Plots ---------------------------
def plot_roc(y_true, y_score, outpath: Path, title: str):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC-AUC = {auc:.4f}")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_pr(y_true, y_score, outpath: Path, title: str):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    plt.figure()
    plt.plot(recall, precision, label=f"PR-AUC (AP) = {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_confusion(y_true, y_pred, outpath: Path, thr: float, title_prefix: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(f"{title_prefix} — threshold={thr:.2f}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0,1]); ax.set_xticklabels(["Non-fraud","Fraud"])
    ax.set_yticks([0,1]); ax.set_yticklabels(["Non-fraud","Fraud"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def plot_prec_rec_vs_threshold(sweep_df: pd.DataFrame, outpath: Path, title: str):
    plt.figure()
    plt.plot(sweep_df["threshold"], sweep_df["precision"], label="Precision")
    plt.plot(sweep_df["threshold"], sweep_df["recall"], label="Recall")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_cost_vs_threshold(sweep_df: pd.DataFrame, outpath: Path, title: str):
    if "cost_per_tx" not in sweep_df.columns:
        return
    plt.figure()
    plt.plot(sweep_df["threshold"], sweep_df["cost_per_tx"], label="Cost per transaction")
    plt.xlabel("Threshold")
    plt.ylabel("Cost")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

# --------------------------- Metrics & Sweep ---------------------------
def confusion_stats(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    tnr = 1.0 - fpr
    fnr = 1.0 - tpr
    bal_acc = (tpr + tnr) / 2.0
    return dict(tn=tn, fp=fp, fn=fn, tp=tp, tpr=tpr, fpr=fpr, tnr=tnr, fnr=fnr, bal_acc=bal_acc)

def threshold_metrics(y_true, y_score, thr: float, cost_fp: float, cost_fn: float):
    y_pred = (y_score >= thr).astype(int)
    stats = confusion_stats(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    total = len(y_true)
    cost_total = stats["fp"] * cost_fp + stats["fn"] * cost_fn
    cost_per_tx = cost_total / total if total else 0.0
    return {
        "threshold": float(thr),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "specificity": float(stats["tnr"]),
        "balanced_accuracy": float(stats["bal_acc"]),
        "tp": int(stats["tp"]), "fp": int(stats["fp"]), "tn": int(stats["tn"]), "fn": int(stats["fn"]),
        "cost_per_tx": float(cost_per_tx)
    }

def sweep_thresholds(y_true, y_score, n: int, cost_fp: float, cost_fn: float) -> pd.DataFrame:
    thrs = np.linspace(0.01, 0.99, n)
    rows = [threshold_metrics(y_true, y_score, float(t), cost_fp, cost_fn) for t in thrs]
    return pd.DataFrame(rows)

def pick_best_threshold(sweep_df: pd.DataFrame, mode: str) -> float | None:
    if mode == "none":
        return None
    if mode == "f1":
        return float(sweep_df.loc[sweep_df["f1"].idxmax(), "threshold"])
    if mode == "youden":
        return float(sweep_df.loc[sweep_df["balanced_accuracy"].idxmax(), "threshold"])
    if mode == "cost":
        return float(sweep_df.loc[sweep_df["cost_per_tx"].idxmin(), "threshold"])
    return None

def read_split(csv_path: Path, target: str):
    df = pd.read_csv(csv_path)
    if target not in df.columns:
        raise KeyError(f"Target '{target}' not found in {csv_path}")
    X = df.drop(columns=[target]).select_dtypes(include=["number"]).copy()
    y = df[target].astype(int).values
    return X, y

# --------------------------- Main ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True, type=str)
    ap.add_argument("--val-csv", required=True, type=str)
    ap.add_argument("--test-csv", required=True, type=str)
    ap.add_argument("--target", default="Class", type=str)

    ap.add_argument("--outdir", default="reports/week9_logreg_from_splits", type=str)
    ap.add_argument("--figdir", default="reports/figures/week9_logreg_from_splits", type=str)
    ap.add_argument("--model-path", default="models/logreg_from_splits.joblib", type=str)

    ap.add_argument("--scaler", choices=["none","standard","robust"], default="standard")
    ap.add_argument("--class-weight", choices=["none","balanced"], default="balanced")
    ap.add_argument("--threshold", default=0.50, type=float)

    ap.add_argument("--optimize", choices=["none","f1","youden","cost"], default="cost")
    ap.add_argument("--n-thresholds", default=101, type=int)
    ap.add_argument("--cost-fp", default=1.0, type=float)
    ap.add_argument("--cost-fn", default=20.0, type=float)

    args = ap.parse_args()

    outdir = Path(args.outdir); ensure_dir(outdir)
    figdir = Path(args.figdir); ensure_dir(figdir)
    model_path = Path(args.model_path); ensure_dir(model_path.parent)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    log_path = outdir / f"run_{run_id}.log"
    setup_logging(log_path)

    git_rev = get_git_rev()
    env = environment_info()

    logging.info("Run %s git=%s", run_id, git_rev)
    logging.info("Loading splits: train=%s val=%s test=%s", args.train_csv, args.val_csv, args.test_csv)

    X_train, y_train = read_split(Path(args.train_csv), args.target)
    X_val, y_val     = read_split(Path(args.val_csv), args.target)
    X_test, y_test   = read_split(Path(args.test_csv), args.target)

    scaler = get_scaler(args.scaler)
    cw = None if args.class_weight == "none" else "balanced"

    clf = LogisticRegression(max_iter=1000, class_weight=cw, solver="lbfgs")
    pipe = Pipeline([("scaler", scaler), ("clf", clf)])

    logging.info("Training Logistic Regression on TRAIN only")
    pipe.fit(X_train, y_train)

    joblib.dump(pipe, model_path)
    logging.info("Saved model: %s", model_path)

    # Scores
    val_score = pipe.predict_proba(X_val)[:, 1]
    test_score = pipe.predict_proba(X_test)[:, 1]

    # Threshold tuning on VAL (correct)
    val_sweep = sweep_thresholds(y_val, val_score, args.n_thresholds, args.cost_fp, args.cost_fn)
    val_sweep.to_csv(outdir / "threshold_sweep_val.csv", index=False)
    best_thr = pick_best_threshold(val_sweep, args.optimize)

    thr_default = float(args.threshold)
    thr_business = best_thr if best_thr is not None else thr_default

    # Evaluate on TEST using both thresholds
    def eval_on_test(thr: float):
        y_pred = (test_score >= thr).astype(int)
        return {
            "threshold": float(thr),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, test_score)),
            "pr_auc": float(average_precision_score(y_test, test_score)),
            "confusion": confusion_stats(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred, digits=4)
        }

    test_default = eval_on_test(thr_default)
    test_business = eval_on_test(thr_business)

    # Save reports
    (outdir / "classification_report_default.txt").write_text(test_default["classification_report"], encoding="utf-8")
    (outdir / "classification_report_business.txt").write_text(test_business["classification_report"], encoding="utf-8")

    # Plots
    plot_roc(y_test, test_score, figdir / "roc.png", "ROC — Logistic Regression (locked splits)")
    plot_pr(y_test, test_score, figdir / "pr.png", "PR — Logistic Regression (locked splits)")
    plot_confusion(y_test, (test_score >= thr_default).astype(int), figdir / "cm_default.png", thr_default, "Confusion (TEST)")
    plot_confusion(y_test, (test_score >= thr_business).astype(int), figdir / "cm_business.png", thr_business, "Confusion (TEST)")

    plot_prec_rec_vs_threshold(val_sweep, figdir / "val_prec_recall_vs_threshold.png", "VAL: Precision/Recall vs Threshold")
    plot_cost_vs_threshold(val_sweep, figdir / "val_cost_vs_threshold.png", "VAL: Cost vs Threshold")

    # (Optional) also sweep on TEST for analysis (NOT for choosing threshold)
    test_sweep = sweep_thresholds(y_test, test_score, args.n_thresholds, args.cost_fp, args.cost_fn)
    test_sweep.to_csv(outdir / "threshold_sweep_test.csv", index=False)

    metrics = {
        "model": "LogisticRegression(locked_splits)",
        "splits": {
            "train_csv": str(Path(args.train_csv).as_posix()),
            "val_csv": str(Path(args.val_csv).as_posix()),
            "test_csv": str(Path(args.test_csv).as_posix()),
            "target": args.target
        },
        "config": {
            "scaler": args.scaler,
            "class_weight": args.class_weight,
            "threshold_default": thr_default,
            "optimize": args.optimize,
            "n_thresholds": args.n_thresholds,
            "cost_fp": args.cost_fp,
            "cost_fn": args.cost_fn
        },
        "threshold_selected_on_val": best_thr,
        "test_eval_default": test_default,
        "test_eval_business": test_business,
        "artifacts": {
            "model_path": str(model_path),
            "metrics_json": str((outdir / "metrics.json").resolve()),
            "threshold_sweep_val_csv": str((outdir / "threshold_sweep_val.csv").resolve()),
            "threshold_sweep_test_csv": str((outdir / "threshold_sweep_test.csv").resolve()),
            "cr_default": str((outdir / "classification_report_default.txt").resolve()),
            "cr_business": str((outdir / "classification_report_business.txt").resolve()),
            "fig_roc": str((figdir / "roc.png").resolve()),
            "fig_pr": str((figdir / "pr.png").resolve()),
            "fig_cm_default": str((figdir / "cm_default.png").resolve()),
            "fig_cm_business": str((figdir / "cm_business.png").resolve()),
            "log_file": str(log_path.resolve()),
        },
        "run": {
            "run_id": run_id,
            "git_rev": git_rev,
            "env": env,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
    }

    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(make_jsonable(metrics), f, indent=2, ensure_ascii=False)

    logging.info("DONE. TEST default thr=%.2f: P=%.4f R=%.4f | business thr=%.4f: P=%.4f R=%.4f",
                 thr_default,
                 test_default["precision"], test_default["recall"],
                 thr_business,
                 test_business["precision"], test_business["recall"])

if __name__ == "__main__":
    main()
