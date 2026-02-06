#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Week 12 — XGBoost (Gradient Boosting) | Business-ready training & evaluation

Usage example (VAL run — choose best threshold on VAL):
python src/12_xgboost.py \
  --input-train data/data_interim/splits_week8/train.csv \
  --input-test  data/data_interim/splits_week8/val.csv \
  --target-column Class \
  --outdir reports/week12_xgb_week8_val \
  --figdir reports/figures/week12_xgb_week8_val \
  --model-path models/xgb_week8.joblib \
  --n-estimators 400 --max-depth 6 --learning-rate 0.05 \
  --subsample 0.8 --colsample-bytree 0.8 \
  --scale-pos-weight auto \
  --threshold 0.50 --optimize cost --n-thresholds 101 \
  --cost-fp 1 --cost-fn 20

Then read best threshold from VAL:
python - <<'PY'
import json
m=json.load(open('reports/week12_xgb_week8_val/metrics.json','r',encoding='utf-8'))
print("XGB VAL best_threshold =", m["best_threshold"])
print("XGB VAL best_threshold_metrics =", m["best_threshold_metrics"])
PY

Usage example (TEST run — evaluate with the VAL-selected threshold; no optimization on TEST):
python src/12_xgboost.py \
  --input-train data/data_interim/splits_week8/train.csv \
  --input-test  data/data_interim/splits_week8/test.csv \
  --target-column Class \
  --outdir reports/week12_xgb_week8_test \
  --figdir reports/figures/week12_xgb_week8_test \
  --model-path models/xgb_week8.joblib \
  --n-estimators 400 --max-depth 6 --learning-rate 0.05 \
  --subsample 0.8 --colsample-bytree 0.8 \
  --scale-pos-weight auto \
  --threshold <PUT_VAL_BEST_THR_HERE> --optimize none --n-thresholds 101 \
  --cost-fp 1 --cost-fn 20

What this script does:
- Train XGBoost on TRAIN, score on EVAL (VAL or TEST), apply a threshold policy, run threshold sweep (0.01..0.99),
  optionally pick a best threshold (cost/F1/Youden), and save reproducible artifacts:
  model, metrics.json, config.json, threshold_sweep.csv, classification_report.txt, feature_importances.csv, plots.

Business reminder:
- PR-AUC is often more meaningful than ROC-AUC under extreme class imbalance (fraud).
- Threshold is a policy decision: it controls fraud leakage (FN) vs workload/friction (FP).
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless export
import matplotlib.pyplot as plt

import joblib
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from xgboost import XGBClassifier


# --------------------------- Utils ---------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def setup_logging(log_path: Path) -> None:
    ensure_dir(log_path.parent)
    fmt = "%(asctime)s %(levelname)s:%(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )


def get_git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def environment_info() -> Dict[str, str]:
    import numpy, pandas, sklearn, xgboost
    return {
        "python": sys.version.replace("\n", " "),
        "os": platform.platform(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
    }


def _json_default(o: Any) -> Any:
    """Robust JSON serialization (fix numpy int64/float32 errors)."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


def load_xy(path: Path, target: str) -> Tuple[pd.DataFrame, np.ndarray, list]:
    df = pd.read_csv(path)
    if target not in df.columns:
        raise KeyError(f"Missing target column '{target}' in {path}")
    X = df.drop(columns=[target])
    y = df[target].astype(int).values
    return X, y, list(X.columns)


def align_eval_to_train(X_train: pd.DataFrame, X_eval: pd.DataFrame) -> pd.DataFrame:
    """Drop extra columns, add missing with 0, keep train order (schema-drift safe)."""
    train_cols = list(X_train.columns)
    eval_cols = list(X_eval.columns)

    extra = [c for c in eval_cols if c not in train_cols]
    if extra:
        X_eval = X_eval.drop(columns=extra)

    missing = [c for c in train_cols if c not in X_eval.columns]
    for c in missing:
        X_eval[c] = 0.0

    return X_eval[train_cols]


# --------------------------- Plots ---------------------------

def plot_roc(y_true: np.ndarray, y_score: np.ndarray, outpath: Path, title: str = "ROC Curve — XGBoost") -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_pr(y_true: np.ndarray, y_score: np.ndarray, outpath: Path, title: str = "Precision–Recall Curve — XGBoost") -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"PR-AUC (AP) = {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, outpath: Path, thr: float, title: str = "Confusion Matrix — XGBoost") -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm, interpolation="nearest", cmap="viridis")
    ax.set_title(f"{title} (thr={thr:.2f})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Non-fraud", "Fraud"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Non-fraud", "Fraud"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, cm[i, j], ha="center", va="center", color=color)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_prec_rec_vs_threshold(sweep_df: pd.DataFrame, outpath: Path) -> None:
    plt.figure(figsize=(6.5, 5))
    plt.plot(sweep_df["threshold"], sweep_df["precision"], label="Precision")
    plt.plot(sweep_df["threshold"], sweep_df["recall"], label="Recall")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision & Recall vs Threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_cost_vs_threshold(sweep_df: pd.DataFrame, outpath: Path) -> None:
    if "cost_per_tx" not in sweep_df.columns:
        return
    plt.figure(figsize=(6.5, 5))
    plt.plot(sweep_df["threshold"], sweep_df["cost_per_tx"], label="Cost per transaction")
    plt.xlabel("Threshold")
    plt.ylabel("Cost")
    plt.title("Cost vs Threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


# --------------------------- Metrics & Threshold Sweep ---------------------------

def confusion_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    tnr = 1.0 - fpr
    fnr = 1.0 - tpr
    bal_acc = (tpr + tnr) / 2.0
    return dict(
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        tpr=float(tpr), fpr=float(fpr), tnr=float(tnr), fnr=float(fnr),
        bal_acc=float(bal_acc)
    )


def threshold_metrics(y_true: np.ndarray, y_score: np.ndarray, thr: float, cost_fp: float, cost_fn: float) -> Dict[str, Any]:
    y_pred = (y_score >= thr).astype(int)
    stats = confusion_stats(y_true, y_pred)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, pos_label=1, average="binary", zero_division=0
    )

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
        "tp": int(stats["tp"]),
        "fp": int(stats["fp"]),
        "tn": int(stats["tn"]),
        "fn": int(stats["fn"]),
        "cost_per_tx": float(cost_per_tx),
    }


def sweep_thresholds(y_true: np.ndarray, y_score: np.ndarray, n: int, cost_fp: float, cost_fn: float) -> pd.DataFrame:
    thrs = np.linspace(0.01, 0.99, n)
    rows = [threshold_metrics(y_true, y_score, float(t), cost_fp, cost_fn) for t in thrs]
    return pd.DataFrame(rows)


def pick_best_threshold(sweep_df: pd.DataFrame, mode: str) -> Optional[float]:
    if mode == "none":
        return None
    if mode == "f1":
        best = sweep_df.loc[sweep_df["f1"].idxmax()]
        return float(best["threshold"])
    if mode == "youden":
        best = sweep_df.loc[sweep_df["balanced_accuracy"].idxmax()]
        return float(best["threshold"])
    if mode == "cost":
        best = sweep_df.loc[sweep_df["cost_per_tx"].idxmin()]
        return float(best["threshold"])
    return None


def recall_at_precision(y_true: np.ndarray, y_score: np.ndarray, target_p: float) -> float:
    p, r, _ = precision_recall_curve(y_true, y_score)
    mask = p >= target_p
    return float(r[mask].max()) if mask.any() else 0.0


def precision_at_recall(y_true: np.ndarray, y_score: np.ndarray, target_r: float) -> float:
    p, r, _ = precision_recall_curve(y_true, y_score)
    mask = r >= target_r
    return float(p[mask].max()) if mask.any() else 0.0


# --------------------------- CLI ---------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="XGBoost — business-ready training & evaluation (threshold policy).")
    ap.add_argument("--input-train", required=True, type=str, help="Training CSV (with target).")
    ap.add_argument("--input-test", required=True, type=str, help="Eval CSV (VAL or TEST, with target).")
    ap.add_argument("--target-column", default="Class", type=str)

    # Hyperparams
    ap.add_argument("--n-estimators", type=int, default=400)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample-bytree", type=float, default=0.8)
    ap.add_argument("--min-child-weight", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--reg-alpha", type=float, default=0.0)
    ap.add_argument("--reg-lambda", type=float, default=1.0)
    ap.add_argument("--max-delta-step", type=float, default=0.0)

    ap.add_argument("--tree-method", type=str, default="hist", help="hist | approx | exact | gpu_hist (if available)")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--random-state", type=int, default=42)

    # Imbalance
    ap.add_argument("--scale-pos-weight", type=str, default="auto", help="auto | none | float (e.g., 20)")
    ap.add_argument("--eval-metric", type=str, default="aucpr", help="aucpr (recommended) | auc | logloss ...")
    ap.add_argument("--early-stopping-rounds", type=int, default=0, help="Use ONLY on VAL runs (avoid on TEST).")

    # Outputs
    ap.add_argument("--outdir", default="reports/week12_xgb", type=str)
    ap.add_argument("--figdir", default="reports/figures/week12_xgb", type=str)
    ap.add_argument("--model-path", default="models/xgb.joblib", type=str)

    # Threshold policy
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument("--optimize", choices=["none", "f1", "youden", "cost"], default="none",
                    help="Pick best threshold from sweep (do on VAL, not on TEST).")
    ap.add_argument("--n-thresholds", type=int, default=101)
    ap.add_argument("--cost-fp", type=float, default=1.0)
    ap.add_argument("--cost-fn", type=float, default=20.0)

    return ap.parse_args()


def parse_scale_pos_weight(val: str, y_train: np.ndarray) -> Optional[float]:
    v = str(val).strip().lower()
    if v in {"none", "null", "off", "0"}:
        return None
    if v == "auto":
        pos = float(np.sum(y_train == 1))
        neg = float(np.sum(y_train == 0))
        return float(neg / pos) if pos > 0 else None
    return float(val)


# --------------------------- Main ---------------------------

def main() -> None:
    args = parse_args()

    outdir = Path(args.outdir); ensure_dir(outdir)
    figdir = Path(args.figdir); ensure_dir(figdir)
    model_path = Path(args.model_path); ensure_dir(model_path.parent)

    log_path = outdir / f"run_{time.strftime('%Y%m%d-%H%M%S')}.log"
    setup_logging(log_path)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    git_rev = get_git_rev()
    env = environment_info()

    if not (0.0 <= args.threshold <= 1.0):
        raise ValueError(f"Threshold must be in [0,1], got {args.threshold}")
    if args.n_thresholds < 3:
        raise ValueError("--n-thresholds must be >= 3")
    if args.cost_fp < 0 or args.cost_fn < 0:
        raise ValueError("Costs must be non-negative.")

    logging.info("Started run_id=%s git_rev=%s", run_id, git_rev)
    logging.info("Loading train/eval CSVs...")

    X_train, y_train, feature_names = load_xy(Path(args.input_train), args.target_column)
    X_eval,  y_eval,  _             = load_xy(Path(args.input_test),  args.target_column)
    X_eval = align_eval_to_train(X_train, X_eval)

    fraud_rate_eval = float(np.mean(y_eval)) if len(y_eval) else 0.0
    logging.info("Train: %d | Eval: %d | Fraud rate eval=%.5f", len(y_train), len(y_eval), fraud_rate_eval)

    spw = parse_scale_pos_weight(args.scale_pos_weight, y_train)

    xgb_params: Dict[str, Any] = dict(
        n_estimators=int(args.n_estimators),
        max_depth=int(args.max_depth),
        learning_rate=float(args.learning_rate),
        subsample=float(args.subsample),
        colsample_bytree=float(args.colsample_bytree),
        min_child_weight=float(args.min_child_weight),
        gamma=float(args.gamma),
        reg_alpha=float(args.reg_alpha),
        reg_lambda=float(args.reg_lambda),
        max_delta_step=float(args.max_delta_step),
        objective="binary:logistic",
        eval_metric=str(args.eval_metric),
        tree_method=str(args.tree_method),
        n_jobs=int(args.n_jobs),
        random_state=int(args.random_state),
    )
    if spw is not None:
        xgb_params["scale_pos_weight"] = float(spw)

    clf = XGBClassifier(**xgb_params)

    # Fit (optional early stopping — use only on VAL runs)
    t0 = time.perf_counter()
    if args.early_stopping_rounds and args.early_stopping_rounds > 0:
        logging.info("Training with early stopping: %d rounds (VAL only).", args.early_stopping_rounds)
        clf.fit(
            X_train, y_train,
            eval_set=[(X_eval, y_eval)],
            verbose=False,
            early_stopping_rounds=int(args.early_stopping_rounds),
        )
    else:
        clf.fit(X_train, y_train, verbose=False)
    fit_sec = time.perf_counter() - t0

    joblib.dump(clf, model_path)
    logging.info("Saved model: %s (fit_time=%.2fs)", model_path, fit_sec)

    # Scores
    y_score = clf.predict_proba(X_eval)[:, 1]

    # Default threshold evaluation
    thr = float(args.threshold)
    y_pred = (y_score >= thr).astype(int)

    acc = accuracy_score(y_eval, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_eval, y_pred, pos_label=1, average="binary", zero_division=0)
    roc_auc = roc_auc_score(y_eval, y_score)
    pr_auc = average_precision_score(y_eval, y_score)

    rec_at_p90 = recall_at_precision(y_eval, y_score, target_p=0.90)
    p_at_r90   = precision_at_recall(y_eval, y_score, target_r=0.90)

    (outdir / "classification_report.txt").write_text(
        classification_report(y_eval, y_pred, digits=4),
        encoding="utf-8"
    )

    # Feature importances (gain-based importance not exposed here; this is split-based importances)
    try:
        imp = getattr(clf, "feature_importances_", None)
        if imp is not None:
            imp_df = pd.DataFrame({"feature": feature_names, "importance": imp})
            imp_df.sort_values("importance", ascending=False).to_csv(outdir / "feature_importances.csv", index=False)
    except Exception as e:
        logging.warning("Could not export feature importances: %s", e)

    # Plots @ default threshold
    plot_roc(y_eval, y_score, figdir / "roc.png")
    plot_pr(y_eval, y_score, figdir / "pr.png")
    plot_confusion(y_eval, y_pred, figdir / "cm_default.png", thr=thr)

    # Threshold sweep
    sweep_df = sweep_thresholds(y_eval, y_score, int(args.n_thresholds), float(args.cost_fp), float(args.cost_fn))
    sweep_csv = outdir / "threshold_sweep.csv"
    sweep_df.to_csv(sweep_csv, index=False)

    plot_prec_rec_vs_threshold(sweep_df, figdir / "prec_recall_vs_threshold.png")
    plot_cost_vs_threshold(sweep_df, figdir / "cost_vs_threshold.png")

    best_thr = pick_best_threshold(sweep_df, args.optimize)
    best_metrics = None
    fig_conf_best_path = None

    if best_thr is not None:
        y_pred_best = (y_score >= best_thr).astype(int)
        fig_conf_best_path = figdir / "cm_best.png"
        plot_confusion(y_eval, y_pred_best, fig_conf_best_path, thr=float(best_thr))
        best_metrics = threshold_metrics(y_eval, y_score, float(best_thr), float(args.cost_fp), float(args.cost_fn))

    metrics: Dict[str, Any] = {
        "model": "XGBoost",
        "xgb_params": xgb_params,
        "threshold_default": thr,
        "metrics": {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "recall_at_precision_0.90": float(rec_at_p90),
            "precision_at_recall_0.90": float(p_at_r90),
        },
        "optimize": args.optimize,
        "best_threshold": best_thr,
        "best_threshold_metrics": best_metrics,
        "cost_fp": float(args.cost_fp),
        "cost_fn": float(args.cost_fn),
        "data": {
            "input_train": str(Path(args.input_train)),
            "input_test": str(Path(args.input_test)),
            "eval_size": int(len(y_eval)),
            "eval_fraud_rate": float(fraud_rate_eval),
        },
        "artifacts": {
            "model_path": str(model_path.resolve()),
            "metrics_json": str((outdir / "metrics.json").resolve()),
            "config_json": str((outdir / "config.json").resolve()),
            "classification_report": str((outdir / "classification_report.txt").resolve()),
            "feature_importances_csv": str((outdir / "feature_importances.csv").resolve()) if (outdir / "feature_importances.csv").exists() else None,
            "sweep_csv": str(sweep_csv.resolve()),
            "fig_roc": str((figdir / "roc.png").resolve()),
            "fig_pr": str((figdir / "pr.png").resolve()),
            "fig_cm_default": str((figdir / "cm_default.png").resolve()),
            "fig_cm_best": str(fig_conf_best_path.resolve()) if fig_conf_best_path else None,
            "fig_prec_rec_vs_thr": str((figdir / "prec_recall_vs_threshold.png").resolve()),
            "fig_cost_vs_thr": str((figdir / "cost_vs_threshold.png").resolve()),
            "log_file": str(log_path.resolve()),
        },
        "run": {
            "run_id": run_id,
            "git_rev": git_rev,
            "env": env,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "fit_time_sec": float(fit_sec),
        },
    }

    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=_json_default)

    with open(outdir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False, default=_json_default)

    md = f"""# Week 12 — XGBoost (Business-Oriented)

**Setup:** n_estimators={args.n_estimators}, max_depth={args.max_depth}, learning_rate={args.learning_rate},
subsample={args.subsample}, colsample_bytree={args.colsample_bytree},
scale_pos_weight={args.scale_pos_weight} (resolved={spw}),
threshold={thr}, optimize={args.optimize}, n_thresholds={args.n_thresholds},
cost_fp={args.cost_fp}, cost_fn={args.cost_fn}

## Metrics (Eval set — default threshold {thr:.2f})
- Accuracy: **{acc:.6f}**
- Precision (Fraud): **{prec:.4f}**
- Recall (Fraud): **{rec:.4f}**
- F1: **{f1:.4f}**
- ROC-AUC: **{roc_auc:.4f}**
- PR-AUC (AP): **{pr_auc:.4f}**
- Recall@Precision≥0.90: **{rec_at_p90:.4f}**
- Precision@Recall≥0.90: **{p_at_r90:.4f}**

{"## Best threshold ("+args.optimize+") = **"+str(round(best_thr,4))+"**" if best_thr is not None else "## Best threshold: (not requested)"}

## Artifacts
- Model: `{model_path.as_posix()}`
- Metrics: `{(outdir / "metrics.json").as_posix()}`
- Config: `{(outdir / "config.json").as_posix()}`
- Report: `{(outdir / "classification_report.txt").as_posix()}`
- Sweep: `{(outdir / "threshold_sweep.csv").as_posix()}`

## Figures (folder)
- `{figdir.as_posix()}`

> Note: Threshold tuning should be done on validation/CV. The test set remains locked for the final evaluation.
"""
    (outdir / "12_xgb_summary.md").write_text(md, encoding="utf-8")

    logging.info(
        "DONE run_id=%s | thr=%.2f | Acc=%.6f P=%.4f R=%.4f F1=%.4f ROC-AUC=%.4f PR-AUC=%.4f",
        run_id, thr, acc, prec, rec, f1, roc_auc, pr_auc
    )


if __name__ == "__main__":
    main()
