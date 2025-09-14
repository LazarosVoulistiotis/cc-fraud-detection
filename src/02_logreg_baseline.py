# src/01_logreg_baseline.py

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # για να μην χρειάζεται GUI
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay
)

# ---- ΡΥΘΜΙΣΕΙΣ ----
DATA_PATH = Path("data/data_raw/creditcard.csv")
IMAGES_DIR = Path("images/week3")
REPORTS_DIR = Path("reports")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATE_TARGETS = ["Class", "class", "is_fraud", "fraud"]
TEST_SIZE = 0.30
RANDOM_STATE = 42


def main():
    # ---- Φόρτωση dataset ----
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Λείπει το {DATA_PATH}. Βάλε το CSV στο data/data_raw/")

    df = pd.read_csv(DATA_PATH)
    target = next((c for c in CANDIDATE_TARGETS if c in df.columns), None)
    if target is None:
        raise ValueError(f"Δεν βρέθηκε target column. Δοκίμασε ένα από: {CANDIDATE_TARGETS}")

    df = df.drop_duplicates().reset_index(drop=True)
    y = df[target].astype(int)
    X = df.drop(columns=[target])

    print(f"Target: {target}")
    print("Shape:", df.shape)
    print("Positive rate:", y.mean())

    # ---- Split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # ---- Pipeline ----
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=RANDOM_STATE
        )
    )
    clf.fit(X_train, y_train)

    # ---- Predictions ----
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "average_precision": average_precision_score(y_test, y_proba),  # PR-AUC
    }

    # ---- Save metrics ----
    metrics_path = REPORTS_DIR / "week3_day1_metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    print(f"Saved metrics → {metrics_path}")

    # ---- Plots ----
    # ROC
    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, y_proba, name="LogReg (balanced)", ax=ax_roc)
    ax_roc.set_title("ROC Curve — Logistic Regression (balanced)")
    fig_roc.savefig(IMAGES_DIR / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig_roc)

    # Precision–Recall
    fig_pr, ax_pr = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_test, y_proba, name="LogReg (balanced)", ax=ax_pr)
    ax_pr.set_title("Precision–Recall Curve — Logistic Regression (balanced)")
    fig_pr.savefig(IMAGES_DIR / "pr_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig_pr)

    # Confusion Matrix
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax_cm)
    ax_cm.set_title("Confusion Matrix (threshold 0.5)")
    fig_cm.savefig(IMAGES_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig_cm)

    print("Saved plots →", IMAGES_DIR)


if __name__ == "__main__":
    main()
