# run python src/02_logreg_baseline.py --data data/data_raw/creditcard.csv --balanced --C 0.1 --recall-target 0.90

# Logistic Regression baseline για credit card fraud (business-ready)

from __future__ import annotations
# By adding the __future__ import, the interpreter will no longer interpret annotations at evaluation time, 
# making the code compatible with both past and future Python versions. This rule respects the target-version setting.
import json # για αποθήκευση reports σε JSON
import argparse # καθαρό CLI.
import logging # logging αντί print για ευέλικτη παρακολούθηση
from pathlib import Path # cross-platform paths

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless export
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import make_pipeline 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay,
    precision_recall_curve, confusion_matrix
)

# --------------------------- CLI & Defaults ---------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Logistic Regression baseline (balanced) + reports/plots",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", type=Path, required=True,
                   help="Path στο creditcard.csv")
    p.add_argument("--images", type=Path, default=Path("reports/figures/week3"),
                   help="Φάκελος για αποθήκευση εικόνων (ROC/PR/CM)")
    p.add_argument("--reports", type=Path, default=Path("reports/week3_metrics"),
                   help="Φάκελος για CSV/JSON reports")
    p.add_argument("--test-size", type=float, default=0.30,
                   help="Ποσοστό test set")
    p.add_argument("--seed", type=int, default=42, help="Random state")
    p.add_argument("--C", type=float, default=1.0,
                   help="Inverse regularization strength της LogisticRegression")
    p.add_argument("--balanced", action="store_true",
                   help="Χρήση class_weight='balanced'")
    p.add_argument("--recall-target", type=float, default=None,
                   help="Αν δοθεί, tuner βρίσκει threshold ώστε Recall≥target")
                    # Επειδή στο fraud το Recall (μην χάνουμε απάτες) έχει προτεραιότητα, γίνεται data-driven επιλογή 
                    # threshold και όχι τυφλά 0.5
    return p

# --------------------------- Logging ---------------------------

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
# Σταθερή δομή logs (timestamp + επίπεδο). Σε παραγωγή συχνά θα στέλνεται σε αρχείο ή stdout του container.
# --------------------------- Utils ---------------------------

CANDIDATE_TARGETS = ["Class", "class", "is_fraud", "fraud"]

def find_target(df: pd.DataFrame) -> str:
    for c in CANDIDATE_TARGETS:
        if c in df.columns:
            return c
    raise ValueError(f"Δεν βρέθηκε target column. Δοκίμασε ένα από: {CANDIDATE_TARGETS}")

def choose_threshold_by_recall(y_true, proba, recall_target: float):
    """Επιλέγει threshold με Recall ≥ στόχο και όσο γίνεται υψηλότερο Precision."""
    pr, rc, thr = precision_recall_curve(y_true, proba)
    # Επιστρέφει precision array, recall array, και thresholds (μήκος len(rc) - 1)
    candidates = [(t, p, r) for p, r, t in zip(pr[:-1], rc[:-1], thr) if r >= recall_target]
    # Δοκιμάζει όλα τα διαθέσιμα thresholds και κρατά μόνο εκείνα με Recall ≥ στόχος.
    if not candidates:
        # fallback: μέγιστο recall που είναι εφικτό
        best_i = int(np.argmax(rc))
        t = thr[min(best_i, len(thr)-1)] if len(thr) else 0.5
        return float(t), float(pr[best_i]), float(rc[best_i])
    # προτίμηση στο καλύτερο precision· σε ισοβαθμία μικρότερο threshold
    t, p, r = max(candidates, key=lambda x: (x[1], -x[0]))
    return float(t), float(p), float(r)

def ensure_dirs(*paths: Path):
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# --------------------------- Main ---------------------------

def main():
    setup_logging() # ενιαία μορφή logs (timestamp, level)
    args = build_parser().parse_args() # παίρνει όλες τις παραμέτρους (paths, C, balanced, recall-target κ.λπ.)
    ensure_dirs(args.images, args.reports) # δημιουργεί (αν δεν υπάρχουν) τους φακέλους για figures/reports, ώστε τα saves να μη σπάνε
    # Guard clause αν λείπει το CSV
    if not args.data.exists():
        raise FileNotFoundError(f"Λείπει το {args.data}. Βάλε το CSV στο data/data_raw/")

    # Load
    df = pd.read_csv(args.data, low_memory=False)
    target = find_target(df)

    # Clean (προληπτικά)
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    if removed:
        logging.info("Αφαιρέθηκαν %d διπλότυπα", removed) # Αποφεύγεις να «μετράς» πολλές φορές το ίδιο event. Το log βοηθά traceability

    y = pd.to_numeric(df[target], errors="raise").astype(int)
    X = df.drop(columns=[target]).select_dtypes(include=["number"]).copy()

    # Info
    pos_rate = y.mean()
    logging.info("Dataset loaded: shape=%s, target=%s, fraud_rate=%.6f (~%.3f%%)", # Το base rate (fraud rate) είναι κρίσιμο σε imbalanced προβλήματα
                 df.shape, target, pos_rate, pos_rate * 100)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )
    logging.info("Train: %s | Test: %s", X_train.shape, X_test.shape) # stratify=y: διατηρείς το ποσοστό fraud και στα δύο splits. Απαραίτητο για σταθερά metrics.

    # Pipeline
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight=("balanced" if args.balanced else None),
            max_iter=1000,
            solver="lbfgs",
            C=args.C,
            random_state=args.seed,
        ),
    )
    clf.fit(X_train, y_train)
    logging.info("Pipeline trained (C=%.4g, balanced=%s)", args.C, bool(args.balanced))

    # Predictions & metrics @0.5
    y_proba = clf.predict_proba(X_test)[:, 1] # Κρατάς πιθανότητες θετικής κλάσης (χρήσιμες για curves & tuning)
    y_pred = (y_proba >= 0.5).astype(int) # baseline labels με το κλασικό cutoff 0.5

    metrics_05 = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "average_precision": average_precision_score(y_test, y_proba),
        "threshold": 0.5,
        "C": args.C,
        "balanced": bool(args.balanced),
        "test_size": args.test_size,
        "seed": args.seed,
    }
    logging.info(
        "Metrics @0.5 | Acc=%.4f Prec=%.4f Rec=%.4f F1=%.4f ROC-AUC=%.4f AP=%.4f",
        metrics_05["accuracy"], metrics_05["precision"], metrics_05["recall"],
        metrics_05["f1"], metrics_05["roc_auc"], metrics_05["average_precision"]
    )

    # Save metrics @0.5
    metrics_csv = args.reports / "week3_day1_metrics.csv"
    pd.DataFrame([metrics_05]).to_csv(metrics_csv, index=False)
    save_json(metrics_05, args.reports / "week3_day1_metrics.json")
    logging.info("Saved metrics → %s", metrics_csv)

    # Plots (ROC/PR/CM) @0.5
    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, y_proba, name="LogReg", ax=ax_roc)
    ax_roc.set_title("ROC Curve — Logistic Regression")
    fig_roc.tight_layout()
    roc_path = args.images / "roc_curve.png"
    fig_roc.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close(fig_roc)

    fig_pr, ax_pr = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_test, y_proba, name="LogReg", ax=ax_pr)
    ax_pr.set_title("Precision–Recall Curve — Logistic Regression")
    fig_pr.tight_layout()
    pr_path = args.images / "pr_curve.png"
    fig_pr.savefig(pr_path, dpi=150, bbox_inches="tight")
    plt.close(fig_pr)

    fig_cm, ax_cm = plt.subplots(figsize=(5.2, 4.4))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax_cm, colorbar=True)
    ax_cm.set_title("Confusion Matrix (threshold 0.5)")
    fig_cm.tight_layout()
    cm_path = args.images / "confusion_matrix_thr0_5.png"
    fig_cm.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig_cm)

    logging.info("Saved plots → %s, %s, %s", roc_path, pr_path, cm_path)

    # Optional: threshold tuning by recall target
    if args.recall_target is not None:
        thr, p_at_thr, r_at_thr = choose_threshold_by_recall(y_test, y_proba, args.recall_target)
        y_pred_thr = (y_proba >= thr).astype(int)

        # extra KPIs
        cm = confusion_matrix(y_test, y_pred_thr)
        tn, fp, fn, tp = cm.ravel() # cm.ravel() σπάει τον πίνακα σε TN, FP, FN, TP
        fpr = fp / (fp + tn) if (fp + tn) else 0.0 # FPR (False Positive Rate) = FP / (FP + TN). Δείχνει πόσα “καθαρά” samples χτυπά το σύστημα ως απάτη. Χρήσιμο για operations (κόστος)
        specificity = tn / (tn + fp) if (tn + fp) else 0.0 # Specificity (True Negative Rate) = TN / (TN + FP). Συμπλήρωμα του FPR

        metrics_thr = {
            "accuracy": accuracy_score(y_test, y_pred_thr),
            "precision": precision_score(y_test, y_pred_thr, zero_division=0),
            "recall": recall_score(y_test, y_pred_thr, zero_division=0),
            "f1": f1_score(y_test, y_pred_thr, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "average_precision": average_precision_score(y_test, y_proba),
            "threshold": float(thr),
            "recall_target": args.recall_target,
            "FPR": fpr,
            "Specificity": specificity,
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
            "Predicted_Pos": int(y_pred_thr.sum()),
            "Support_Pos": int((y_test == 1).sum()),
            "Support_Neg": int((y_test == 0).sum()),
            "C": args.C,
            "balanced": bool(args.balanced),
            "test_size": args.test_size,
            "seed": args.seed,
        }

        # Save tuned metrics
        tuned_csv = args.reports / "week3_day1_threshold_selection.csv"
        pd.DataFrame([metrics_thr]).to_csv(tuned_csv, index=False)
        save_json(metrics_thr, args.reports / "week3_day1_threshold_selection.json")
        logging.info(
            "Tuned @thr=%.4f | Acc=%.4f Prec=%.4f Rec=%.4f F1=%.4f FPR=%.4f Spec=%.4f",
            thr, metrics_thr["accuracy"], metrics_thr["precision"],
            metrics_thr["recall"], metrics_thr["f1"], fpr, specificity
        )

        # PR curve με σημείωση threshold
        pr_vals, rc_vals, thr_vals = precision_recall_curve(y_test, y_proba)
        fig_pr2, ax_pr2 = plt.subplots(figsize=(6, 5))
        ax_pr2.plot(rc_vals, pr_vals, label=f"AP={metrics_thr['average_precision']:.3f}")
        if len(thr_vals):
            idx = int(np.argmin(np.abs(thr_vals - thr)))
            ax_pr2.scatter(rc_vals[idx], pr_vals[idx], marker="o")
            ax_pr2.annotate(
                f"thr={thr:.3f}\nP={pr_vals[idx]:.2f}, R={rc_vals[idx]:.2f}",
                (rc_vals[idx], pr_vals[idx]), textcoords="offset points", xytext=(10, -10)
            )
        ax_pr2.set_xlabel("Recall"); ax_pr2.set_ylabel("Precision")
        ax_pr2.set_title(f"Precision–Recall (tuned @thr={thr:.3f})")
        ax_pr2.legend(loc="upper right", frameon=True)
        fig_pr2.tight_layout()
        pr_marked = args.images / f"pr_curve_tuned_thr{thr:.3f}.png"
        fig_pr2.savefig(pr_marked, dpi=150, bbox_inches="tight")
        plt.close(fig_pr2)

        # Confusion Matrix @ tuned thr
        fig_cm2, ax_cm2 = plt.subplots(figsize=(5.2, 4.4))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred_thr, ax=ax_cm2, colorbar=True)
        ax_cm2.set_title(f"Confusion Matrix (thr={thr:.3f})")
        fig_cm2.tight_layout()
        cm_tuned = args.images / f"confusion_matrix_thr{thr:.3f}.png"
        fig_cm2.savefig(cm_tuned, dpi=150, bbox_inches="tight")
        plt.close(fig_cm2)

        logging.info("Saved tuned plots → %s, %s", pr_marked, cm_tuned)


if __name__ == "__main__":
    main()
