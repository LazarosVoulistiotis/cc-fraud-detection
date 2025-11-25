# --------------------------- Shebang, encoding, docstring (CLI παράδειγμα) ---------------------------
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Χρήση (παράδειγμα):
python src/10_decision_tree.py \
  --input-train data/data_interim/train.csv \
  --input-test  data/data_interim/test.csv \
  --target-column Class \
  --outdir reports/week10_dt_baseline \
  --figdir reports/figures/week10 \
  --model-path models/dt_baseline.joblib \
  --max-depth None --min-samples-split 2 --min-samples-leaf 1 \
  --class-weight none --random-state 42 \
  --threshold 0.50 --optimize none --n-thresholds 101 \
  --cost-fp 1.0 --cost-fn 20.0

Σημειώσεις “business”:
- Παράγει threshold sweep (0.01..0.99) και μπορεί να επιλέξει “βέλτιστο” threshold
  με κριτήριο F1, Youden (balanced accuracy proxy) ή κόστος (FP/FN).
- Σώζει artifacts (μοντέλο, metrics.json, config.json, classification report, feature importances,
  plots ROC/PR/Confusion, Precision/Recall vs thr, Cost vs thr) και metadata run (git rev, env).
- Για παραγωγή: threshold tuning σε validation/CV, το test μένει κλειστό.
"""

# --------------------------- Imports & backend ---------------------------

from __future__ import annotations

import argparse, json, sys, time, subprocess, platform
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless export
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, confusion_matrix, classification_report
)

import joblib
import logging

# --------------------------- Utils ---------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def setup_logging(log_path: Path):
    ensure_dir(log_path.parent)
    fmt = "%(asctime)s %(levelname)s:%(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8")
        ],
    )


def get_git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
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


def load_xy(path: Path, target: str):
    df = pd.read_csv(path)
    if target not in df.columns:
        raise KeyError(f"Missing target column '{target}' in {path}")
    X = df.drop(columns=[target])
    y = df[target].astype(int).values
    return X, y, list(X.columns)


# --------------------------- Plots ---------------------------

def plot_roc(y_true, y_score, outpath: Path, title="ROC Curve — Decision Tree"):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {auc:.4f}")
    plt.plot([0,1], [0,1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
# Κατάλληλο για σταθερού κόστους προβλήματα. Σε ισχυρή ανισορροπία κλάσεων (fraud) η PR-AUC είναι πιο σχετική, αλλά η ROC δίνει μια “μακροσκοπική” εικόνα.

def plot_pr(y_true, y_score, outpath: Path, title="Precision–Recall Curve — Decision Tree"):
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
# PR-AUC συνδέεται με ποιότητα alerts και φόρτο ομάδας ελέγχου (false positives).

def plot_confusion(y_true, y_pred, outpath: Path, thr: float, title="Confusion Matrix — Decision Tree"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm, interpolation="nearest", cmap="viridis")
    ax.set_title(f"{title} (thr={thr:.2f})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0,1]); ax.set_xticklabels(["Non-fraud","Fraud"])
    ax.set_yticks([0,1]); ax.set_yticklabels(["Non-fraud","Fraud"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i,j] > cm.max()/2 else "black"
            ax.text(j, i, cm[i, j], ha="center", va="center", color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
# Εδώ κάνεις κόστος-ανάγνωση με μια ματιά: FP (άδικα alerts) vs FN (χαμένες απάτες).

def plot_prec_rec_vs_threshold(sweep_df: pd.DataFrame, outpath: Path):
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
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn) if (tp + fn) else 0.0  # recall
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    tnr = 1.0 - fpr  # specificity
    fnr = 1.0 - tpr
    bal_acc = (tpr + tnr) / 2.0
    return dict(tn=tn, fp=fp, fn=fn, tp=tp, tpr=tpr, fpr=fpr, tnr=tnr, fnr=fnr, bal_acc=bal_acc)
# “πόσες απάτες δεν μας ξεφεύγουν”, FPR = “θόρυβος στο SOC/ops”.

def threshold_metrics(y_true, y_score, thr: float, cost_fp: float|int, cost_fn: float|int):
    y_pred = (y_score >= thr).astype(int)
    stats = confusion_stats(y_true, y_pred)
    prec = precision_recall_fscore_support(y_true, y_pred, pos_label=1, average="binary", zero_division=0)[0]
    rec  = stats["tpr"]
    f1   = precision_recall_fscore_support(y_true, y_pred, pos_label=1, average="binary", zero_division=0)[2]
    total = len(y_true)
    cost_total = stats["fp"] * cost_fp + stats["fn"] * cost_fn # cost_fp: κόστος άδικου alert (π.χ. χειροκίνητος έλεγχος, εμπειρία πελάτη), cost_fn: κόστος χαμένης απάτης (συνήθως πολύ μεγαλύτερο). Αυτό σε αφήνει να κάνεις data-driven επιλογή threshold, όχι “0.50 επειδή έτσι”. 
    cost_per_tx = cost_total / total if total else 0.0
    return {
        "threshold": thr,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "specificity": float(stats["tnr"]),
        "balanced_accuracy": float(stats["bal_acc"]),
        "tp": int(stats["tp"]), "fp": int(stats["fp"]), "tn": int(stats["tn"]), "fn": int(stats["fn"]),
        "cost_per_tx": float(cost_per_tx)
    }


def sweep_thresholds(y_true, y_score, n: int, cost_fp: float|int, cost_fn: float|int) -> pd.DataFrame:
    thrs = np.linspace(0.01, 0.99, n)
    rows = [threshold_metrics(y_true, y_score, float(t), cost_fp, cost_fn) for t in thrs]
    return pd.DataFrame(rows)


def pick_best_threshold(sweep_df: pd.DataFrame, mode: str) -> float | None:
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
    p, r, _ = precision_recall_curve(y_true, y_score)
    mask = p >= target_p
    return float(r[mask].max()) if mask.any() else 0.0


def precision_at_recall(y_true, y_score, target_r: float) -> float:
    p, r, _ = precision_recall_curve(y_true, y_score)
    mask = r >= target_r
    return float(p[mask].max()) if mask.any() else 0.0
# είναι SLA-style KPIs: “Αν ζητήσω Precision ≥ 0.90, μέχρι τι Recall μπορώ να φτάσω;” “Αν απαιτώ Recall ≥ 0.90, ποιο Precision επιτυγχάνω;” Αυτά δεν “κλειδώνουν” συγκεκριμένο threshold· απαντούν στο ερώτημα πολιτικής
# "SLA-style KPIs" refers to Key Performance Indicators (KPIs) that are specifically designed to measure performance against the formal commitments outlined in a Service Level Agreement (SLA). While SLAs are the contractual promises (e.g., "99.9% uptime guaranteed"), the associated KPIs are the metrics (e.g., monthly uptime percentage tracked in a dashboard) used to monitor, verify, and report whether those promises are being met. 

# --------------------------- CLI & Main pipeline ---------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Decision Tree — business-ready baseline (training & evaluation).")
    ap.add_argument("--input-train", required=True, type=str, help="Path to training CSV (with target column).")
    ap.add_argument("--input-test", required=True, type=str, help="Path to test CSV (with target column).")
    ap.add_argument("--target-column", default="Class", type=str)

    # Hyperparams DT
    ap.add_argument("--criterion", choices=["gini","entropy","log_loss"], default="gini")
    ap.add_argument("--max-depth", type=lambda v: None if v=="None" else int(v), default=None)
    ap.add_argument("--min-samples-split", type=int, default=2)
    ap.add_argument("--min-samples-leaf", type=int, default=1)
    ap.add_argument("--class-weight", choices=["none","balanced"], default="none")
    ap.add_argument("--random-state", type=int, default=42)

    # Outputs & metadata
    ap.add_argument("--outdir", default="reports/week10_dt_baseline", type=str)
    ap.add_argument("--figdir", default="reports/figures/week10", type=str)
    ap.add_argument("--model-path", default="models/dt_baseline.joblib", type=str)

    # Thresholds & business
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
    logging.info("Φόρτωση train/test")
    X_train, y_train, feature_names = load_xy(Path(args.input_train), args.target_column)
    X_test,  y_test,  _             = load_xy(Path(args.input_test),  args.target_column)
    logging.info("Train: %d rows | Test: %d rows | Fraud rate test=%.5f",
                 len(y_train), len(y_test), float(np.mean(y_test)))

    # Model
    cw = None if args.class_weight == "none" else "balanced"
    clf = DecisionTreeClassifier(
        criterion=args.criterion,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        class_weight=cw,
        random_state=args.random_state,
    )

    logging.info("Εκπαίδευση Decision Tree...")
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    fit_sec = time.perf_counter() - t0
    joblib.dump(clf, model_path)
    logging.info("Αποθηκεύτηκε μοντέλο: %s (fit_time=%.2fs)", model_path, fit_sec)

    # Predict
    y_score = clf.predict_proba(X_test)[:, 1]
    thr = float(args.threshold)
    y_pred = (y_score >= thr).astype(int)
    # Κρατάς probabilities και δίνεις ερμηνεία με threshold (default 0.50).

    # Core metrics
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, pos_label=1, average="binary", zero_division=0)
    roc_auc = roc_auc_score(y_test, y_score)
    pr_auc = average_precision_score(y_test, y_score)

    # KPIs τύπου SLA
    rec_at_p90 = recall_at_precision(y_test, y_score, target_p=0.90)
    p_at_r90   = precision_at_recall(y_test, y_score, target_r=0.90)
    # Τα δύο SLA-KPIs απαντούν σε επιχειρησιακά constraints.

    # Classification report
    clf_rep = classification_report(y_test, y_pred, digits=4)
    with open(outdir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(clf_rep)

    # Feature importances
    try:
        imp = clf.feature_importances_
        imp_df = pd.DataFrame({"feature": feature_names, "importance": imp})
        imp_df.sort_values("importance", ascending=False).to_csv(outdir / "feature_importances.csv", index=False)
    except Exception as e:
        logging.warning("Αδυναμία εξαγωγής feature importances: %s", e)
    # Explainability σε rule-based μοντέλο: ποια χαρακτηριστικά “κόβουν” πιο συχνά.

    # Plots (default thr)
    plot_roc(y_test, y_score, figdir / "week10_dt_roc.png")
    plot_pr(y_test, y_score, figdir / "week10_dt_pr.png")
    plot_confusion(y_test, y_pred, figdir / "week10_dt_confusion_default.png", thr=thr)

    # Threshold sweep
    sweep_df = sweep_thresholds(y_true=y_test, y_score=y_score, n=int(args.n_thresholds),
                                cost_fp=args.cost_fp, cost_fn=args.cost_fn)
    sweep_csv = outdir / "threshold_sweep.csv"
    sweep_df.to_csv(sweep_csv, index=False)
    plot_prec_rec_vs_threshold(sweep_df, figdir / "week10_dt_prec_recall_vs_threshold.png")
    plot_cost_vs_threshold(sweep_df, figdir / "week10_dt_cost_vs_threshold.png")

    # Best threshold (optional)
    best_thr = pick_best_threshold(sweep_df, args.optimize)
    best_metrics = None
    if best_thr is not None:
        y_pred_best = (y_score >= best_thr).astype(int)
        plot_confusion(y_test, y_pred_best, figdir / "week10_dt_confusion_best.png", thr=best_thr)
        best_metrics = threshold_metrics(y_test, y_score, best_thr, args.cost_fp, args.cost_fn)
    # Όταν οι proba έχουν διαβάθμιση, εδώ θα δεις ουσιαστικό κέρδος.

    # Metrics.json με metadata
    metrics = {
        "model": "DecisionTree(baseline)",
        "parameters": clf.get_params(),
        "test_size_note": "external provided split (train/test csv)",
        "threshold_default": thr,
        "metrics": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "recall_at_precision_0.90": rec_at_p90,
            "precision_at_recall_0.90": p_at_r90
        },
        "optimize": args.optimize,
        "best_threshold": best_thr,
        "best_threshold_metrics": best_metrics,
        "cost_fp": args.cost_fp,
        "cost_fn": args.cost_fn,
        "artifacts": {
            "model_path": str(model_path.resolve()),
            "metrics_json": str((outdir / "metrics.json").resolve()),
            "classification_report": str((outdir / "classification_report.txt").resolve()),
            "feature_importances_csv": str((outdir / "feature_importances.csv").resolve()),
            "sweep_csv": str(sweep_csv.resolve()),
            "fig_roc": str((figdir / "week10_dt_roc.png").resolve()),
            "fig_pr": str((figdir / "week10_dt_pr.png").resolve()),
            "fig_conf_default": str((figdir / "week10_dt_confusion_default.png").resolve()),
            "fig_conf_best": str((figdir / "week10_dt_confusion_best.png").resolve()),
            "fig_prec_rec_vs_thr": str((figdir / "week10_dt_prec_recall_vs_threshold.png").resolve()),
            "fig_cost_vs_thr": str((figdir / "week10_dt_cost_vs_threshold.png").resolve()),
            "log_file": str(log_path.resolve())
        },
        "run": {
            "run_id": run_id,
            "git_rev": git_rev,
            "env": env,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
    }
    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Save full config (CLI args)
    with open(outdir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    # Markdown summary
    md = f"""# Week 10 — Decision Tree Baseline (Business-Oriented)

**Setup:** criterion={args.criterion}, max_depth={args.max_depth}, min_samples_split={args.min_samples_split},
min_samples_leaf={args.min_samples_leaf}, class_weight={args.class_weight}, random_state={args.random_state},
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

{"## Βέλτιστο threshold ("+args.optimize+") = **"+str(round(best_thr,4))+"**" if best_thr is not None else "## Βέλτιστο threshold: (δεν ζητήθηκε)"}

## Εικόνες
- ROC: `reports/figures/week10/week10_dt_roc.png`
- PR: `reports/figures/week10/week10_dt_pr.png`
- Confusion (default): `reports/figures/week10/week10_dt_confusion_default.png`
- Confusion (best): `reports/figures/week10/week10_dt_confusion_best.png`
- Precision/Recall vs Threshold: `reports/figures/week10/week10_dt_prec_recall_vs_threshold.png`
- Cost vs Threshold: `reports/figures/week10/week10_dt_cost_vs_threshold.png`

## Αρχεία
- Μοντέλο: `models/dt_baseline.joblib`
- Μετρικές: `reports/week10_dt_baseline/metrics.json`
- Classification report: `reports/week10_dt_baseline/classification_report.txt`
- Feature importances: `reports/week10_dt_baseline/feature_importances.csv`
- Threshold Sweep: `reports/week10_dt_baseline/threshold_sweep.csv`
- Config (CLI args): `reports/week10_dt_baseline/config.json`
- Log: `{log_path.name}`

> Σημείωση: Το threshold tuning εδώ είναι για **αναλυτική αναφορά** πάνω στο test.
> Σε παραγωγική ροή, ορίζουμε validation/CV για επιλογή threshold και κρατάμε το test “κλειστό”.
"""
    with open(outdir / "10_dt_baseline_summary.md", "w", encoding="utf-8") as f:
        f.write(md)

    logging.info("Ολοκληρώθηκε run_id=%s. Μετρικές (thr=%.2f): Acc=%.6f P=%.4f R=%.4f F1=%.4f ROC-AUC=%.4f PR-AUC=%.4f",
                 run_id, thr, acc, prec, rec, f1, roc_auc, pr_auc)


if __name__ == "__main__":
    main()

"""Πώς να το “διαβάσεις” για Business αποφάσεις
Κομμάτι	                    Ερώτηση που απαντά
PR-AUC & PR curve	        Πόσο “καθαρά” είναι τα alerts (precision) όταν ανεβάζω recall;
Confusion @ thr	            Πόσα fraud χάνω (FN) και πόσο φορτώνω την ομάδα (FP);
Cost vs Threshold	        Ποιο threshold ελαχιστοποιεί το κόστος;
Recall@Precision≥X	        Αν ζητήσω ποιότητα alerts (Precision), μέχρι που φτάνει η κάλυψη (Recall);
Precision@Recall≥Y	        Αν απαιτώ κάλυψη (Recall), τι ποιότητα alerts στέλνω (Precision);
Feature importances	        Γιατί το μοντέλο δίνει αυτά τα alerts; Πού να εστιάσω σε feature engineering;
metrics.json + artifacts	Audit trail, reproducibility, σύγκριση runs, BI ingestion."""