#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Για να δεις τι επιλογές (CLI arguments) δέχεται:
python src/11_random_forest.py --help

Χρήση 2 runs (ένα baseline + ένα “balanced+conservative”):

Run A — baseline:
python src/11_random_forest.py \
  --input-train data/data_interim/train.csv \
  --input-test  data/data_interim/test.csv \
  --target-column Class \
  --outdir reports/week11_rf_runA_baseline \
  --figdir reports/figures/week11/runA \
  --model-path models/rf_runA_baseline.joblib \
  --n-estimators 100 --max-depth None --min-samples-leaf 1 \
  --class-weight none \
  --threshold 0.50 --optimize none --n-thresholds 101 \
  --cost-fp 1.0 --cost-fn 20.0

Run F — balanced + conservative:
python src/11_random_forest.py \
  --input-train data/data_interim/train.csv \
  --input-test  data/data_interim/test.csv \
  --target-column Class \
  --outdir reports/week11_rf_runF_balanced_conservative \
  --figdir reports/figures/week11/runF \
  --model-path models/rf_runF_balanced.joblib \
  --n-estimators 300 --max-depth 12 --min-samples-leaf 3 \
  --class-weight balanced \
  --threshold 0.50 --optimize cost --n-thresholds 101 \
  --cost-fp 1.0 --cost-fn 20.0

!)Tο script με 1 πρόταση
Είναι ένα reproducible training + evaluation pipeline για Random Forest που:
-εκπαιδεύει μοντέλο,
-βγάζει probabilities,
-εφαρμόζει threshold (decision policy),
-κάνει threshold sweep (0.01–0.99),
-μπορεί να επιλέξει “best threshold” με βάση F1 / Youden / Cost,
-σώζει artifacts (μοντέλο, metrics.json, plots, feature importances)

2)Ροή εκτέλεσης (σαν “mental diagram”)
-Parse args (CLI → hyperparameters / paths / business knobs)
-Load train/test από CSV
-Build RandomForestClassifier (+ class_weight handling)
-Fit
-Predict proba (y_score = P(class=1))
-Apply threshold (y_pred = y_score ≥ thr)
-Compute KPIs (Accuracy, Precision/Recall/F1, ROC-AUC, PR-AUC, “SLA” metrics)
-Plots (ROC, PR, Confusion)
-Threshold sweep → CSV + plots (Prec/Rec vs thr, Cost vs thr)
-Pick best threshold (optional) + confusion + metrics
-Save outputs (metrics.json, config.json, report.txt, md summary)
  
Σημειώσεις “business”:
- Random Forest = ensemble (bagging) → μειώνει variance → πιο σταθερές αποφάσεις.
- Παράγει threshold sweep (0.01..0.99) και μπορεί να επιλέξει “βέλτιστο” threshold με βάση F1/Youden/Cost.
- Σώζει artifacts (μοντέλο, metrics.json, config.json, classification report, feature importances, plots).
"""

from __future__ import annotations

import argparse, json, sys, time, subprocess, platform
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") # χωρίς display (για servers)
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, confusion_matrix, classification_report
)

import joblib
import logging


# --------------------------- Utils ---------------------------

def ensure_dir(p: Path):
    """Δημιουργεί directory αν δεν υπάρχει (ώστε να μη σκάσουν τα saves)."""
    p.mkdir(parents=True, exist_ok=True)


def setup_logging(log_path: Path):
    """
    Logging σε console + αρχείο.
    Βοήθεια για report evidence (run trace + params + αποτελέσματα).
    """
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
    """Πιάνει το git commit (reproducibility). Αν δεν υπάρχει git, γυρνάει unknown."""
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def environment_info():
    """Καταγραφή versions packages για reproducibility στο metrics.json."""
    import numpy, pandas, sklearn
    return {
        "python": sys.version.replace("\n"," "),
        "os": platform.platform(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__,
    }


def load_xy(path: Path, target: str):
    """
    Φορτώνει CSV και χωρίζει features/target.
    Περιμένει ότι το train/test CSV έχουν ήδη γίνει preprocess (πχ scaling) σε προηγούμενες εβδομάδες.
    """
    df = pd.read_csv(path)
    if target not in df.columns:
        raise KeyError(f"Missing target column '{target}' in {path}")
    X = df.drop(columns=[target])
    y = df[target].astype(int).values
    return X, y, list(X.columns)


def parse_bool(v: str) -> bool:
    """Μετατροπή string CLI σε bool."""
    return str(v).strip().lower() in {"1", "true", "yes", "y", "t"}


def parse_max_features(v: str):
    """
    max_features μπορεί να είναι:
    - 'sqrt', 'log2', None
    - float (π.χ. 0.5 για 50% των features)
    """
    v = str(v).strip()
    if v.lower() in {"none", "null"}:
        return None
    if v in {"sqrt", "log2"}:
        return v
    try:
        return float(v)
    except Exception:
        return v  # αφήνουμε sklearn να το χειριστεί


def parse_max_depth(v: str):
    """max_depth από CLI: δέχεται None/none ή int."""
    v = str(v).strip()
    if v.lower() in {"none", "null"}:
        return None
    return int(v)


# --------------------------- Plots ---------------------------

def plot_roc(y_true, y_score, outpath: Path, title="ROC Curve — Random Forest"):
    """ROC curve: χρήσιμο για overall separability (όχι πάντα top metric σε imbalance)."""
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


def plot_pr(y_true, y_score, outpath: Path, title="Precision–Recall Curve — Random Forest"):
    """
    PR curve: πιο “fraud-friendly” σε imbalanced data.
    Το AP (Average Precision) είναι PR-AUC.
    """
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


def plot_confusion(y_true, y_pred, outpath: Path, thr: float, title="Confusion Matrix — Random Forest"):
    """
    Confusion matrix (2x2) με labels [0,1] για σιγουριά.
    TP = fraud stopped, FN = fraud missed, FP = friction/cost, TN = ok.
    """
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


def plot_prec_rec_vs_threshold(sweep_df: pd.DataFrame, outpath: Path):
    """Trade-off plot: Precision & Recall καθώς αλλάζει το threshold."""
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


def plot_cost_vs_threshold(sweep_df: pd.DataFrame, outpath: Path):
    """Business plot: expected cost per transaction vs threshold (αν υπάρχει)."""
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

def confusion_stats(y_true, y_pred):
    """
    Υπολογίζει βασικά stats από confusion matrix.
    labels=[0,1] ώστε να είναι πάντα 2x2.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Recall / TPR = TP / (TP + FN)
    tpr = tp / (tp + fn) if (tp + fn) else 0.0

    # FPR = FP / (FP + TN)
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    # Specificity / TNR = 1 - FPR
    tnr = 1.0 - fpr

    # FNR = 1 - TPR
    fnr = 1.0 - tpr

    # Balanced accuracy = (TPR + TNR) / 2
    bal_acc = (tpr + tnr) / 2.0

    return dict(tn=tn, fp=fp, fn=fn, tp=tp, tpr=tpr, fpr=fpr, tnr=tnr, fnr=fnr, bal_acc=bal_acc)


def threshold_metrics(y_true, y_score, thr: float, cost_fp: float | int, cost_fn: float | int):
    """
    Μετατρέπει probabilities -> predictions με threshold
    και υπολογίζει:
    - precision/recall/f1 για fraud (pos_label=1)
    - confusion stats
    - cost per transaction (FP*cost_fp + FN*cost_fn)/N
    """
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
        "tp": int(stats["tp"]), "fp": int(stats["fp"]), "tn": int(stats["tn"]), "fn": int(stats["fn"]),
        "cost_per_tx": float(cost_per_tx),
    }


def sweep_thresholds(y_true, y_score, n: int, cost_fp: float | int, cost_fn: float | int) -> pd.DataFrame:
    """Δοκιμάζει thresholds από 0.01 μέχρι 0.99 και γυρνάει DataFrame με metrics."""
    thrs = np.linspace(0.01, 0.99, n)
    rows = [threshold_metrics(y_true, y_score, float(t), cost_fp, cost_fn) for t in thrs]
    return pd.DataFrame(rows)


def pick_best_threshold(sweep_df: pd.DataFrame, mode: str) -> float | None:
    """
    Επιλογή threshold:
    - f1: max F1
    - youden: max balanced accuracy (ισοδύναμο με max (TPR - FPR))
    - cost: min cost_per_tx
    """
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


def recall_at_precision(y_true, y_score, target_p: float) -> float:
    """Μέγιστο recall που μπορούμε να πετύχουμε κρατώντας precision >= target_p."""
    p, r, _ = precision_recall_curve(y_true, y_score)
    mask = p >= target_p
    return float(r[mask].max()) if mask.any() else 0.0


def precision_at_recall(y_true, y_score, target_r: float) -> float:
    """Μέγιστο precision που μπορούμε να πετύχουμε κρατώντας recall >= target_r."""
    p, r, _ = precision_recall_curve(y_true, y_score)
    mask = r >= target_r
    return float(p[mask].max()) if mask.any() else 0.0


# --------------------------- CLI & Main pipeline ---------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Random Forest — business-ready baseline (training & evaluation).")
    ap.add_argument("--input-train", required=True, type=str, help="Path to training CSV (with target column).")
    ap.add_argument("--input-test", required=True, type=str, help="Path to test CSV (with target column).")
    ap.add_argument("--target-column", default="Class", type=str)

    # Hyperparams RF (controlled experimentation knobs)
    ap.add_argument("--n-estimators", type=int, default=100)
    ap.add_argument("--max-depth", type=lambda v: None if v=="None" else int(v), default=None)
    ap.add_argument("--min-samples-split", type=int, default=2)
    ap.add_argument("--min-samples-leaf", type=int, default=1)
    ap.add_argument("--max-features", type=str, default="sqrt", help="sqrt | log2 | None | float (π.χ. 0.5)")
    ap.add_argument("--bootstrap", type=parse_bool, default=True)
    ap.add_argument("--n-jobs", type=int, default=-1)

    # Imbalance handling (business knob: “πόσο τιμωρείς το FN”)
    ap.add_argument("--class-weight", choices=["none","balanced","balanced_subsample","custom"], default="none")
    ap.add_argument("--pos-weight", type=float, default=10.0, help="Χρήση μόνο όταν class-weight=custom (βάρος για fraud).")

    ap.add_argument("--random-state", type=int, default=42)

    # Outputs & metadata
    ap.add_argument("--outdir", default="reports/week11_rf_baseline", type=str)
    ap.add_argument("--figdir", default="reports/figures/week11", type=str)
    ap.add_argument("--model-path", default="models/rf_baseline.joblib", type=str)

    # Thresholding & cost (risk policy)
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument("--optimize", choices=["none","f1","youden","cost"], default="none",
                    help="Κριτήριο επιλογής «βέλτιστου» threshold από sweep.")
    ap.add_argument("--n-thresholds", type=int, default=101, help="Σημεία sweep (0.01..0.99).")
    ap.add_argument("--cost-fp", type=float, default=1.0, help="Κόστος False Positive.")
    ap.add_argument("--cost-fn", type=float, default=20.0, help="Κόστος False Negative.")
    return ap.parse_args()


def main():
    args = parse_args()

    outdir = Path(args.outdir); ensure_dir(outdir)
    figdir = Path(args.figdir); ensure_dir(figdir)
    model_path = Path(args.model_path); ensure_dir(model_path.parent)

    # Log file per run (ώστε να έχεις audit trail)
    log_path = outdir / f"run_{time.strftime('%Y%m%d-%H%M%S')}.log"
    setup_logging(log_path)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    git_rev = get_git_rev()
    env = environment_info()

    # Validation ορισμάτων
    if not (0.0 <= args.threshold <= 1.0):
        raise ValueError(f"Threshold must be in [0,1], got {args.threshold}")
    if args.n_thresholds < 3:
        raise ValueError("--n-thresholds must be >= 3")
    if args.cost_fp < 0 or args.cost_fn < 0:
        raise ValueError("Costs must be non-negative.")

    logging.info("Ξεκίνησε run_id=%s git_rev=%s", run_id, git_rev)

    # 1) Load data
    logging.info("Φόρτωση train/test")
    X_train, y_train, feature_names = load_xy(Path(args.input_train), args.target_column)
    X_test,  y_test,  _             = load_xy(Path(args.input_test),  args.target_column)

    logging.info("Train: %d | Test: %d | Fraud rate test=%.5f",
                 len(y_train), len(y_test), float(np.mean(y_test)))

    # 2) class_weight handling
    cw = None
    if args.class_weight == "balanced":
        cw = "balanced"
    elif args.class_weight == "balanced_subsample":
        cw = "balanced_subsample"
    elif args.class_weight == "custom":
        cw = {0: 1.0, 1: float(args.pos_weight)}

    # 3) Build model
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=parse_max_features(args.max_features),
        bootstrap=bool(args.bootstrap),
        n_jobs=args.n_jobs,
        class_weight=cw,
        random_state=args.random_state,
    )

    # 4) Train + save model artifact
    logging.info("Εκπαίδευση Random Forest...")
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    fit_sec = time.perf_counter() - t0

    joblib.dump(clf, model_path)
    logging.info("Αποθηκεύτηκε μοντέλο: %s (fit_time=%.2fs)", model_path, fit_sec)

    # 5) Predict probabilities (score)
    # IMPORTANT: Αυτό είναι το “knob” που συνδέεται με threshold tuning.
    y_score = clf.predict_proba(X_test)[:, 1]

    # 6) Apply default threshold
    thr = float(args.threshold)
    y_pred = (y_score >= thr).astype(int)

    # 7) Core metrics
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, pos_label=1, average="binary", zero_division=0)
    roc_auc = roc_auc_score(y_test, y_score)
    pr_auc = average_precision_score(y_test, y_score)

    # 8) SLA-style KPIs (ωραία για “operational targets”)
    rec_at_p90 = recall_at_precision(y_test, y_score, target_p=0.90)
    p_at_r90   = precision_at_recall(y_test, y_score, target_r=0.90)

    # 9) Save classification report (human-readable)
    (outdir / "classification_report.txt").write_text(
        classification_report(y_test, y_pred, digits=4),
        encoding="utf-8"
    )

    # 10) Feature importances (RF έχει built-in)
    try:
        imp_df = pd.DataFrame({"feature": feature_names, "importance": clf.feature_importances_})
        imp_df.sort_values("importance", ascending=False).to_csv(outdir / "feature_importances.csv", index=False)
    except Exception as e:
        logging.warning("Αδυναμία εξαγωγής feature importances: %s", e)

    # 11) Plots @ default threshold
    plot_roc(y_test, y_score, figdir / "week11_rf_roc.png")
    plot_pr(y_test, y_score, figdir / "week11_rf_pr.png")
    plot_confusion(y_test, y_pred, figdir / "week11_rf_confusion_default.png", thr=thr)

    # 12) Threshold sweep
    sweep_df = sweep_thresholds(
        y_true=y_test,
        y_score=y_score,
        n=int(args.n_thresholds),
        cost_fp=args.cost_fp,
        cost_fn=args.cost_fn
    )
    sweep_csv = outdir / "threshold_sweep.csv"
    sweep_df.to_csv(sweep_csv, index=False)

    plot_prec_rec_vs_threshold(sweep_df, figdir / "week11_rf_prec_recall_vs_threshold.png")
    plot_cost_vs_threshold(sweep_df, figdir / "week11_rf_cost_vs_threshold.png")

    # 13) Best threshold (optional)
    best_thr = pick_best_threshold(sweep_df, args.optimize)
    best_metrics = None
    fig_conf_best_path = None

    if best_thr is not None:
        y_pred_best = (y_score >= best_thr).astype(int)
        fig_conf_best_path = figdir / "week11_rf_confusion_best.png"
        plot_confusion(y_test, y_pred_best, fig_conf_best_path, thr=best_thr)
        best_metrics = threshold_metrics(y_test, y_score, best_thr, args.cost_fp, args.cost_fn)

    # 14) Save metrics.json (machine-readable evidence)
    metrics = {
        "model": "RandomForest",
        "parameters": clf.get_params(),
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
        "artifacts": {
            "model_path": str(model_path.resolve()),
            "metrics_json": str((outdir / "metrics.json").resolve()),
            "classification_report": str((outdir / "classification_report.txt").resolve()),
            "feature_importances_csv": str((outdir / "feature_importances.csv").resolve()),
            "sweep_csv": str(sweep_csv.resolve()),
            "fig_roc": str((figdir / "week11_rf_roc.png").resolve()),
            "fig_pr": str((figdir / "week11_rf_pr.png").resolve()),
            "fig_conf_default": str((figdir / "week11_rf_confusion_default.png").resolve()),
            "fig_conf_best": str(fig_conf_best_path.resolve()) if fig_conf_best_path else None,
            "fig_prec_rec_vs_thr": str((figdir / "week11_rf_prec_recall_vs_threshold.png").resolve()),
            "fig_cost_vs_thr": str((figdir / "week11_rf_cost_vs_threshold.png").resolve()),
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
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # 15) Save config.json (ακριβώς τα CLI args που έτρεξαν)
    with open(outdir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    # 16) Markdown summary (report-friendly)
    md = f"""# Week 11 — Random Forest (Business-Oriented)

**Setup:** n_estimators={args.n_estimators}, max_depth={args.max_depth}, min_samples_split={args.min_samples_split},
min_samples_leaf={args.min_samples_leaf}, max_features={args.max_features}, bootstrap={args.bootstrap},
class_weight={args.class_weight}, random_state={args.random_state},
threshold={thr}, optimize={args.optimize}, n_thresholds={args.n_thresholds}, cost_fp={args.cost_fp}, cost_fn={args.cost_fn}

## Μετρικές (Test set — default threshold {thr:.2f})
- Accuracy: **{acc:.6f}**
- Precision: **{prec:.4f}**
- Recall: **{rec:.4f}**
- F1: **{f1:.4f}**
- ROC-AUC: **{roc_auc:.4f}**
- PR-AUC (AP): **{pr_auc:.4f}**
- Recall@Precision≥0.90: **{rec_at_p90:.4f}**
- Precision@Recall≥0.90: **{p_at_r90:.4f}**

{"## Best threshold ("+args.optimize+") = **"+str(round(best_thr,4))+"**" if best_thr is not None else "## Best threshold: (not requested)"}

## Artifacts
- Model: `{model_path.as_posix()}`
- Metrics: `{(outdir / "metrics.json").as_posix()}`
- Report: `{(outdir / "classification_report.txt").as_posix()}`
- Importances: `{(outdir / "feature_importances.csv").as_posix()}`
- Sweep: `{(outdir / "threshold_sweep.csv").as_posix()}`

## Figures (folder)
- `{figdir.as_posix()}`

> Note: In production, threshold tuning should be done on validation/CV; test set remains locked.
"""
    (outdir / "11_rf_baseline_summary.md").write_text(md, encoding="utf-8")

    logging.info(
        "Ολοκληρώθηκε run_id=%s | thr=%.2f | Acc=%.6f P=%.4f R=%.4f F1=%.4f ROC-AUC=%.4f PR-AUC=%.4f",
        run_id, thr, acc, prec, rec, f1, roc_auc, pr_auc
    )


if __name__ == "__main__":
    main()
