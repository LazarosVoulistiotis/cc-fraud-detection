
"""
Logistic Regression (L2) για credit card fraud:
- Grid σε C
- Metrics @ threshold=0.5
- Επιλογή καλύτερου C με PR-AUC (Average Precision)
- Threshold tuning για στόχο Recall
- Αποθήκευση ROC/PR καμπυλών & Confusion Matrix
- Αποθήκευση συνοπτικών metrics σε CSV

Τρέξιμο:
    python src/03_reg_threshold.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

# Headless backend (για Windows/CI/χωρίς οθόνη). ΠΡΕΠΕΙ πριν το pyplot import.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)

# -------------------- ΡΥΘΜΙΣΕΙΣ --------------------

# Ρίζα του repo, ανεξάρτητα από το current working dir
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "data_raw" / "creditcard.csv"
IMAGES_DIR = BASE_DIR / "reports" / "figures" /"week3"
REPORTS_DIR = BASE_DIR / "reports" / "week3_metrics"

TARGET_CANDIDATES = ["Class", "class", "is_fraud"]
CS = [0.01, 0.1, 1, 10]         # grid σε C (λογαριθμική κλίμακα)
TEST_SIZE = 0.30
RANDOM_STATE = 42
RECALL_TARGET = 0.90            # στόχος για threshold tuning


# -------------------- HELPERS --------------------

def find_target(df: pd.DataFrame) -> str:
    """Εντόπισε τη στήλη-στόχο από πιθανές ονομασίες."""
    for c in TARGET_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"Δεν βρήκα target σε {TARGET_CANDIDATES}. Δώσε σωστή στήλη στόχου (0/1).")


def ensure_outdirs():
    """Δημιούργησε φακέλους εξόδου αν δεν υπάρχουν και επέστρεψέ τους."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return IMAGES_DIR, REPORTS_DIR


def print_metrics_table(rows, title="Metrics per C"):
    """
    Εμφάνισε metrics ως πίνακα.
    - Σε Jupyter: styled display
    - Σε script/κονσόλα: απλό print με to_string()
    """
    dfm = pd.DataFrame(rows)
    try:
        # δούλεψε ως styled ΜΟΝΟ αν τρέχουμε μέσα σε IPython/Jupyter
        from IPython import get_ipython
        shell = get_ipython()
        if shell is not None:  # Jupyter/IPython session
            from IPython.display import display
            display(dfm.style.hide(axis="index").set_caption(title))
            return
    except Exception:
        pass

    # κονσόλα / κανονικό script fallback
    print(f"\n{title}")
    print(dfm.to_string(index=False))



def choose_threshold_by_recall(y_true, proba, recall_target=0.90):
    """
    Βρες το threshold με recall ≥ recall_target, μεγιστοποιώντας precision.
    Αν δεν υπάρχει τέτοιο threshold, επέστρεψε το σημείο με μέγιστο recall.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, proba)

    candidates = []
    for i in range(len(thresholds)):  # thresholds έχει len(precision)-1
        r = recall[i]
        p = precision[i]
        thr = thresholds[i]
        if r >= recall_target:
            candidates.append((thr, p, r))

    if not candidates:
        best_i = int(np.argmax(recall))
        best_thr = thresholds[min(best_i, len(thresholds) - 1)] if len(thresholds) > 0 else 0.5
        return float(best_thr), float(precision[best_i]), float(recall[best_i])

    # Διάλεξε με μέγιστο precision, και αν ισοβαθμεί, μικρότερο threshold
    best_thr, best_p, best_r = max(candidates, key=lambda x: (x[1], -x[0]))
    return float(best_thr), float(best_p), float(best_r)


# -------------------- MAIN --------------------

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}. Put the dataset in data/data_raw/")

    out_images, out_reports = ensure_outdirs()

    # Φόρτωση & βασικός καθαρισμός
    df = pd.read_csv(DATA_PATH)
    y_col = find_target(df)
    y = df[y_col].astype(int)
    X = df.drop(columns=[y_col]).select_dtypes(include=[np.number])

    # Train/Test split με stratify
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # ---------------- Grid σε C ----------------
    rows = []
    best_key = None
    best_score = -np.inf
    proba_per_C = {}

    for C in CS:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                penalty="l2",
                C=C,
                solver="lbfgs",
                random_state=RANDOM_STATE
            ))
        ])

        # Εκπαίδευση
        pipe.fit(X_train, y_train)

        # Πιθανότητες και labels@0.5
        proba = pipe.predict_proba(X_test)[:, 1]
        proba_per_C[C] = proba  # κράτα για μετέπειτα ανάλυση
        pred_05 = (proba >= 0.5).astype(int)

        # Μετρικές
        acc = accuracy_score(y_test, pred_05)
        prec = precision_score(y_test, pred_05, zero_division=0)
        rec = recall_score(y_test, pred_05, zero_division=0)
        f1 = f1_score(y_test, pred_05, zero_division=0)
        rocauc = roc_auc_score(y_test, proba)
        ap = average_precision_score(y_test, proba)

        rows.append({
            "C": C,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4),
            "ROC-AUC": round(rocauc, 4),
            "PR-AUC(AP)": round(ap, 4),
            "Positives@0.5": int(pred_05.sum())
        })

        # --- Plots (ROC & PR) ---
        # ROC
        fpr, tpr, _ = roc_curve(y_test, proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC-AUC={rocauc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve (C={C})"); plt.legend()
        plt.savefig(out_images / f"roc_C{C}.png", dpi=150, bbox_inches="tight"); plt.close()

        # PR
        pr, rc, _ = precision_recall_curve(y_test, proba)
        plt.figure()
        plt.plot(rc, pr, label=f"AP={ap:.3f}")
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve (C={C})"); plt.legend()
        plt.savefig(out_images / f"pr_C{C}.png", dpi=150, bbox_inches="tight"); plt.close()

        # Επιλογή best C με βάση PR-AUC
        if ap > best_score:
            best_score = ap
            best_key = C

    # Παρουσίαση/Αποθήκευση metrics table
    print_metrics_table(rows, title="Metrics per C @ threshold=0.5")
    pd.DataFrame(rows).sort_values("PR-AUC(AP)", ascending=False)\
        .to_csv(out_reports / "week3_day2_metrics.csv", index=False)

    print(f"\n>>> Best C by PR-AUC(AP): {best_key}")

    # ---------------- Threshold tuning στο best C ----------------
    best_C = best_key
    proba_best = proba_per_C[best_C]

    thr, p_at_thr, r_at_thr = choose_threshold_by_recall(
        y_test, proba_best, recall_target=RECALL_TARGET
    )
    pred_thr = (proba_best >= thr).astype(int)

    # Μετρικές στο επιλεγμένο threshold
    acc = accuracy_score(y_test, pred_thr)
    prec = precision_score(y_test, pred_thr, zero_division=0)
    rec = recall_score(y_test, pred_thr, zero_division=0)
    f1 = f1_score(y_test, pred_thr, zero_division=0)
    rocauc = roc_auc_score(y_test, proba_best)
    ap = average_precision_score(y_test, proba_best)

    print(f"\nThreshold tuning (target Recall ≥ {RECALL_TARGET:.2f}):")
    print(f"Selected threshold = {thr:.4f}")
    print(
        f"Metrics @thr: Accuracy={acc:.4f}  Precision={prec:.4f}  "
        f"Recall={rec:.4f}  F1={f1:.4f}  ROC-AUC={rocauc:.4f}  PR-AUC(AP)={ap:.4f}"
    )

    # PR curve με σημείωση του threshold
    pr, rc, thresholds = precision_recall_curve(y_test, proba_best)
    plt.figure()
    plt.plot(rc, pr, label=f"AP={ap:.3f}")
    if len(thresholds) > 0:
        idx = int(np.argmin(np.abs(thresholds - thr)))
        plt.scatter(rc[idx], pr[idx], marker="o")
        plt.annotate(
            f"thr={thr:.3f}\nP={pr[idx]:.2f}, R={rc[idx]:.2f}",
            (rc[idx], pr[idx]), textcoords="offset points", xytext=(10, -10)
        )
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"PR Curve (best C={best_C})"); plt.legend()
    plt.savefig(out_images / f"pr_bestC{best_C}_thr.png", dpi=150, bbox_inches="tight"); plt.close()

    # Confusion matrix + heatmap-like
    cm = confusion_matrix(y_test, pred_thr)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (None, None, None, None)
    print(f"\nConfusion matrix @thr={thr:.3f}:\n{cm}")
    print(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title(f"Confusion Matrix @thr={thr:.3f} (C={best_C})")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Pred 0", "Pred 1"])
    plt.yticks(tick_marks, ["True 0", "True 1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.xlabel("Predicted label"); plt.ylabel("True label")
    plt.savefig(out_images / f"cm_bestC{best_C}_thr.png", dpi=150, bbox_inches="tight"); plt.close()

    # (προαιρετικό) αποθήκευση classification report
    report_str = classification_report(y_test, pred_thr, zero_division=0)
    with open(out_reports / f"classification_report_bestC{best_C}_thr.txt", "w", encoding="utf-8") as f:
        f.write(report_str)


if __name__ == "__main__":
    main()
