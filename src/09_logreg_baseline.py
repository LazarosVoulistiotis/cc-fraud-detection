#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Χρήση:
python src/09_logreg_baseline.py --data data/data_raw/creditcard.csv \
    --outdir reports/week9_baseline --figdir reports/figures/week9 \
    --model-path models/logreg_baseline.joblib --test-size 0.20 --seed 42 \
    --scaler standard --class-weight balanced --threshold 0.50 \
    --optimize none --n-thresholds 101 --cost-fp 1.0 --cost-fn 20.0

Σημείωση “business”:
- Το default threshold (0.50) αξιολογείται ΚΑΙ γίνεται sweep σε [0.01..0.99] για να βρεις
  εναλλακτικό threshold βάσει αντικειμενικού στόχου:
  --optimize none|f1|youden|cost
- Το κόστος ορίζεται με --cost-fp και --cost-fn (ανά περιστατικό). Τα αποτελέσματα κόστους
  είναι “αναλυτικά-reporting” πάνω στο test. Για παραγωγή: κάνε tuning σε validation/CV.
"""
from __future__ import annotations # αναβολή αξιολόγησης type hints σε strings

import argparse, json, sys, time, subprocess, platform # CLI parsing, serialization, σύστημα & metadata (git hash, OS, timestamps)
from pathlib import Path
import numpy as np # αριθμητικοί υπολογισμοί
import pandas as pd # φόρτωση & επεξεργασία δεδομένων
import matplotlib # γραφικά
matplotlib.use("Agg")  # backend για headless export PNG. Χρήσιμο σε servers/CI χωρίς GUI
import matplotlib.pyplot as plt # pyplot μόνο για αποθήκευση γραφημάτων
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)
import joblib # αποθήκευση ολόκληρου Pipeline (scaler+μοντέλο) ως ένα αρχείο
import logging # ενιαίο logging σε κονσόλα + αρχείο για audit trail
# όλα τα artifacts (plots, json, txt, joblib) μπορούν να συλλεχθούν σε CI/MLflow/DAGs και να τεκμηριώσουν το run end-to-end

# --------------------------- Utils ---------------------------
# Μικρό guard που δημιουργεί όλους τους γονικούς φακέλους αν λείπουν (parents=True) και δεν σκάει αν υπάρχουν (exist_ok=True).
# Business νόημα: αποφεύγεις flaky runs επειδή “έλειπε ο φάκελος figures”. ("Flaky runs" refers to automated tests that randomly pass or fail without any changes to the underlying code, indicating unreliability. This can cause issues in software development by creating distrust in test results, wasting time on reruns, and potentially letting actual bugs slip through. Causes range from environmental inconsistencies and race conditions to poor test code and infrastructure issues, and identifying and addressing these is crucial for reliable testing pipelines)
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def get_scaler(name: str):
    if name == "standard":
        return StandardScaler() # για κανονικοποίηση με μέσο/τυπική απόκλιση (καλό baseline)
    if name == "robust":
        return RobustScaler() # αν φοβάσαι heavy-tailed/outliers
    return "passthrough" # για να μην εφαρμόσεις καθόλου scaler μέσα στο Pipeline

def setup_logging(log_path: Path): # ενιαίο logging σε κονσόλα + αρχείο
    fmt = "%(asctime)s %(levelname)s:%(message)s" # χρονική σήμανση, σοβαρότητα, μήνυμα
    logging.basicConfig(
        level=logging.INFO, 
        format=fmt, 
        handlers=[
            logging.StreamHandler(sys.stdout), # StreamHandler στο stdout για live παρακολούθηση
            logging.FileHandler(log_path, encoding="utf-8") # FileHandler σε log_path για αποτύπωμα run
        ],
    )

def get_git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True
        ).strip()
    except Exception:
        return "unknown" 
    # Καλεί git rev-parse --short HEAD και γυρνά το τρέχον commit. Σε σφάλμα (π.χ. δεν είναι git repo ή λείπει το git) επιστρέφει "unknown".
    # Business νόημα: κάθε run συνδέεται με συγκεκριμένη έκδοση κώδικα. Αυτό είναι χρυσός για reproducibility και audit (ποιο commit παρήγαγε αυτό το μοντέλο;).

def environment_info():
    import numpy, pandas, sklearn
    return {
        "python": sys.version.replace("\n"," "),
        "os": platform.platform(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__,
    }
    # Business νόημα: αποτύπωση περιβάλλοντος. Αν αύριο αλλάξεις scikit-learn και “γλιστρήσουν” τα αποτελέσματα, θα ξέρεις γιατί.

# --------------------------- Plots ---------------------------
# Η ROC είναι threshold-free. Καλό σήμα για rank ordering του μοντέλου. Σε έντονο class imbalance μπορεί να φαίνεται υπεραισιόδοξη, γι’ αυτό δεν την παρουσιάζουμε ποτέ μόνη της.
def plot_roc(y_true, y_score, outpath: Path): # Υπολογίζει ROC curve (FPR vs TPR) και ROC-AUC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC-AUC = {auc:.4f}")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Logistic Regression (baseline)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close() # Χρησιμεύει για management slides, αλλά για απάτη η PR-AUC “λέει την αλήθεια” για τα false alarms

# Σε datasets με 0.1–0.2% fraud, η PR είναι η σωστή διοπτρα: πόσα από τα alerts είναι όντως απάτες (precision) και τι ποσοστό απάτης πιάνουμε (recall).
def plot_pr(y_true, y_score, outpath: Path): # Ζωγραφίζει Precision–Recall και αναφέρει Average Precision (AP) = PR-AUC.
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    plt.figure()
    plt.plot(recall, precision, label=f"PR-AUC (AP) = {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve — Logistic Regression (baseline)")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close() # Η baseline precision σε PR ισούται περίπου με το fraud rate. Αν το AP σου < fraud rate, έχουμε πρόβλημα

# Είναι η πιο “χειροπιαστή” εικόνα για TP/FP/FN/TN. Για fraud, τα FN καίνε λεφτά, τα FP καίνε λειτουργικό χρόνο
def plot_confusion(y_true, y_pred, outpath: Path, thr: float): # Confusion Matrix στο συγκεκριμένο threshold (π.χ. 0.50).
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(f"Confusion Matrix — threshold={thr:.2f}")
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
    plt.close(fig) # Καλό να δείχνεις δύο matrices: default (0.50) και βέλτιστο threshold (π.χ. cost-optimal)

# Δείχνει πώς αλλάζουν precision και recall καθώς ανεβοκατεβάζεις το threshold.
# Αποκαλύπτει το trade-off που θα κουβεντιάσεις με business/ops: περισσότερα catches (recall) ⇒ περισσότερα false alarms (precision πέφτει)
def plot_prec_rec_vs_threshold(sweep_df: pd.DataFrame, outpath: Path):
    plt.figure()
    plt.plot(sweep_df["threshold"], sweep_df["precision"], label="Precision") 
    plt.plot(sweep_df["threshold"], sweep_df["recall"], label="Recall")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision & Recall vs Threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close() # Συνήθως, όταν ανεβαίνει το threshold, precision ↑, recall ↓

# Γράφημα κόστους ανά συναλλαγή vs threshold, εφόσον έχεις υπολογίσει cost_per_tx.
# Εκεί κερδίζεις τους stakeholders. Δείχνεις πού ελαχιστοποιείται το κόστος με βάση τους συντελεστές --cost-fp, --cost-fn
def plot_cost_vs_threshold(sweep_df: pd.DataFrame, outpath: Path):
    if "cost_per_tx" not in sweep_df.columns:
        return
    plt.figure()
    plt.plot(sweep_df["threshold"], sweep_df["cost_per_tx"], label="Cost per transaction")
    plt.xlabel("Threshold")
    plt.ylabel("Cost")
    plt.title("Cost vs Threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close() # Συνήθως FN ≫ FP στο κόστος. Βάλε ρεαλιστικά νούμερα (π.χ. 1€ ανά FP έλεγχος, 50–500€ ανά FN απώλεια)

# --------------------------- Metrics & Sweep ---------------------------
# Εξάγει TN, FP, FN, TP και παραγώγες
def confusion_stats(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn) if (tp + fn) else 0.0  # recall
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    tnr = 1.0 - fpr  # specificity
    fnr = 1.0 - tpr
    bal_acc = (tpr + tnr) / 2.0
    return dict(tn=tn, fp=fp, fn=fn, tp=tp, tpr=tpr, fpr=fpr, tnr=tnr, fnr=fnr, bal_acc=bal_acc) # Balanced accuracy «ζυγίζει» και τις δύο πλευρές: τι πιάνεις (TPR) και πόσο καθαρός μένεις (TNR).

# Μετατρέπει πιθανότητες σε labels στο συγκεκριμένο thr
def threshold_metrics(y_true, y_score, thr: float, cost_fp: float|int, cost_fn: float|int):
    y_pred = (y_score >= thr).astype(int)
    stats = confusion_stats(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    # απλό επιχειρηματικό κόστος ανά συναλλαγή
    total = len(y_true)
    cost_total = stats["fp"] * cost_fp + stats["fn"] * cost_fn
    cost_per_tx = cost_total / total if total else 0.0 # cost_per_tx = (FP·cost_fp + FN·cost_fn) / N
    return {
        "threshold": thr,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "specificity": stats["tnr"],
        "balanced_accuracy": stats["bal_acc"],
        "tp": stats["tp"], "fp": stats["fp"], "tn": stats["tn"], "fn": stats["fn"],
        "cost_per_tx": cost_per_tx
    } # Έχεις ενιαίο λεξιλόγιο κόστους για να συζητήσεις thresholds με risk/ops/CS.

# # Φτιάχνει ένα grid από thresholds (0.01..0.99) με n σημεία και υπολογίζει τα μετρικά για το καθένα
def sweep_thresholds(y_true, y_score, n: int, cost_fp: float|int, cost_fn: float|int) -> pd.DataFrame:
    thrs = np.linspace(0.01, 0.99, n)
    rows = [threshold_metrics(y_true, y_score, float(t), cost_fp, cost_fn) for t in thrs]
    return pd.DataFrame(rows)

def pick_best_threshold(sweep_df: pd.DataFrame, mode: str) -> float | None:
    if mode == "none":
        return None # δεν διαλέγει (μένεις στο default)
    if mode == "f1":
        best = sweep_df.loc[sweep_df["f1"].idxmax()]
        return float(best["threshold"]) # max F1 — καλή ισορροπία, αλλά μπορεί να μην αντανακλά επιχειρηματικό στόχο
    if mode == "youden":
        # Youden = TPR - FPR -> εδώ χρησιμοποιούμε balanced_accuracy proxy: max
        best = sweep_df.loc[sweep_df["balanced_accuracy"].idxmax()]
        return float(best["threshold"]) # max balanced accuracy (proxy του Youden’s J = TPR − FPR)
    if mode == "cost":
        best = sweep_df.loc[sweep_df["cost_per_tx"].idxmin()]
        return float(best["threshold"]) # "cost": ελαχιστοποίηση κόστους — η πιο business-native επιλογή αν έχεις σωστά coefficients
    return None # Δείχνεις μεθοδολογική διαφάνεια: “άλλο metric, άλλο threshold”. Δεν βγαίνει “ένα ιερό” threshold· επιλέγεις αυτό που ταιριάζει στον στόχο

# Αν θέλω ≥90% precision, πόσο recall πετυχαίνω;
def recall_at_precision(y_true, y_score, target_p: float) -> float:
    p, r, _ = precision_recall_curve(y_true, y_score)
    mask = p >= target_p
    return float(r[mask].max()) if mask.any() else 0.0
# Αν θέλω ≥90% recall, τι precision μπορώ να κρατήσω;
def precision_at_recall(y_true, y_score, target_r: float) -> float:
    p, r, _ = precision_recall_curve(y_true, y_score)
    mask = r >= target_r
    return float(p[mask].max()) if mask.any() else 0.0
# Αυτό είναι SLA-style metric. Μεταφράζεται άμεσα σε πολιτικές λειτουργίας (π.χ. “κρατάμε precision>=0.9 για να μη ζαλίζουμε το back-office”).
# Ένα "SLA-style metric" (μετρική τύπου SLA) είναι ένας δείκτης που χρησιμοποιείται για τη μέτρηση της παρεχόμενης υπηρεσίας, σύμφωνα με τους όρους ενός Συμφωνητικού Επιπέδου Υπηρεσιών (SLA). Αυτοί οι δείκτες ορίζονται σε ένα συμβόλαιο μεταξύ παρόχου και πελάτη (ή μεταξύ τμημάτων μιας εταιρείας) και περιλαμβάνουν μετρήσεις όπως ο χρόνος απόκρισης, η διαθεσιμότητα ενός συστήματος ή ο χρόνος επίλυσης ενός αιτήματος. Η παρακολούθηση αυτών των μετρικών βοηθά στη διασφάλιση της ποιότητας της υπηρεσίας και στον καθορισμό κυρώσεων ή διορθωτικών ενεργειών εάν οι συμφωνημένοι στόχοι δεν επιτευχθούν

# --------------------------- Main ---------------------------

def main():
    ap = argparse.ArgumentParser() # CLI args
    ap.add_argument("--data", required=True, type=str) # path στο CSV dataset
    ap.add_argument("--outdir", default="reports/week9_baseline", type=str) # όπου θα αποθηκευτούν τα αποτελέσματα
    ap.add_argument("--figdir", default="reports/figures/week9", type=str) # όπου θα αποθηκευτούν τα γραφήματα
    ap.add_argument("--model-path", default="models/logreg_baseline.joblib", type=str) # όπου θα αποθηκευτεί το μοντέλο
    ap.add_argument("--test-size", default=0.2, type=float) # ποσοστό test set
    ap.add_argument("--seed", default=42, type=int) # για αναπαραγωγιμότητα split
    ap.add_argument("--scaler", choices=["none","standard","robust"], default="standard") # συνειδητή επιλογή κανονικοποίησης
    ap.add_argument("--class-weight", choices=["none","balanced"], default="balanced") # baseline για imbalance χωρίς resampling
    ap.add_argument("--threshold", default=0.50, type=float) # αρχικό decision cutoff (A "decision cutoff" is a threshold used to decide between two or more options, commonly seen in finance, medicine, and machine learning. It represents the point at which a decision is made, such as whether to invest in a security, a diagnostic test result is positive, or a predictive model should classify an item as belonging to one class or another. The appropriate cutoff point is often determined by specific criteria, like risk tolerance, required return, or the need to balance different outcomes (e.g., sensitivity vs. specificity))
    ap.add_argument("--optimize", choices=["none","f1","youden","cost"], default="none",
                    help="Κριτήριο επιλογής «βέλτιστου» threshold από sweep.")
    ap.add_argument("--n-thresholds", default=101, type=int, help="Σημεία sweep (0.01..0.99).") # πόσα thresholds να δοκιμάσουμε στο sweep (0.01..0.99)
    ap.add_argument("--cost-fp", default=1.0, type=float, help="Κόστος False Positive.") # κόστος ανά false positive (π.χ. κόστος χειρισμού alert)
    ap.add_argument("--cost-fn", default=20.0, type=float, help="Κόστος False Negative.") # κόστος ανά false negative (π.χ. απώλεια από fraud)
    args = ap.parse_args() # ανάγνωση CLI args
    # Business νόημα: το εργαλείο εκφράζει πολιτικές απόφασης (thresholds) και κόστη ρητά. Μπορείς να το κολλάς αυτούσιο σε SOP/Runbook.

    # --- Validation ορισμάτων
    # Ελέγχεις ότι threshold ∈ [0,1], test_size ∈ (0,1), και n_thresholds ≥ 3.Αυτό αποτρέπει “ήσυχες” αστοχίες (π.χ. αρνητικό threshold ή test 95%).
    if not (0.0 <= args.threshold <= 1.0):
        raise ValueError(f"Threshold must be in [0,1], got {args.threshold}")
    if not (0.0 < args.test_size < 1.0):
        raise ValueError(f"test_size must be in (0,1), got {args.test_size}")
    if args.n_thresholds < 3:
        raise ValueError("--n-thresholds must be >= 3")

    # --- Paths & logging
    data_path = Path(args.data)
    outdir = Path(args.outdir); ensure_dir(outdir)
    figdir = Path(args.figdir); ensure_dir(figdir)
    model_path = Path(args.model_path); ensure_dir(model_path.parent)
    log_path = outdir / f"run_{time.strftime('%Y%m%d-%H%M%S')}.log"
    ensure_dir(log_path.parent)
    setup_logging(log_path) # Φτιάχνεις μοναδικό log αρχείο ανά run (run_YYYYmmdd-HHMMSS.log) και το περνάς στο setup_logging
    # Business νόημα: κάθε run αφήνει αποτύπωμα (audit trail). Σε incident review ξέρεις τι τρέξαμε, πότε, με ποια config.

    # --- Metadata run(reproducibility & audit trail)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    git_rev = get_git_rev()
    env = environment_info()
    # Όταν αλλάξει scikit-learn minor version και “γλιστρήσουν” 3 δεκαδικά στο AP, θα ξέρεις γιατί
    
    # --- Log start
    logging.info("Ξεκίνησε run_id=%s git_rev=%s", run_id, git_rev)
    logging.info("Φόρτωση dataset από %s", data_path)

    # --- Load data(sanity check για στήλη Class)
    df = pd.read_csv(data_path)
    if "Class" not in df.columns:
        raise KeyError("Δεν βρέθηκε η στήλη Class στο dataset.")
    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int).values

    # --- Split data
    logging.info("Stratified split (test_size=%.2f, seed=%d)", args.test_size, args.seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    ) # Με stratify=y κρατάς ίδιες αναλογίες απάτης σε train/test. Αυτό είναι κρίσιμο όταν το prevalence είναι ~0.17%.
    # Business νόημα: αξιολογείς στο test ίδιες συνθήκες πληθυσμού με το training.

    # --- Pipeline (leakage-proof) & model 
    scaler = get_scaler(args.scaler) # επιλογή scaler από CLI
    cw = None if args.class_weight == "none" else "balanced" # επιλογή class_weight από CLI
    clf = LogisticRegression(max_iter=1000, class_weight=cw, solver="lbfgs") # σταθερό baseline, L2 regularization (L2 regularization is a machine learning technique that prevents overfitting by adding a penalty term to the cost function that is proportional to the square of the model's weights. This discourages large weights, leading to a simpler model that is better at generalizing to new data. The strength of the penalty is controlled by a hyperparameter, usually denoted by lambda (\(\lambda \)), and it's also known as Ridge regularization.)
    pipe = Pipeline([("scaler", scaler), ("clf", clf)]) # ο scaler “μαθαίνει” μόνο στο train, άρα όχι data leakage.

    # --- Fit 
    logging.info("Εκπαίδευση Logistic Regression (baseline)")
    pipe.fit(X_train, y_train) # εκπαίδευση pipeline (scaler + μοντέλο)

    # --- Save model
    joblib.dump(pipe, model_path) # αποθήκευση ολόκληρου pipeline (scaler + μοντέλο) ως ένα αρχείο .joblib
    logging.info("Αποθηκεύτηκε μοντέλο: %s", model_path)

    # --- Predict & default threshold
    y_score = pipe.predict_proba(X_test)[:, 1] # παίρνεις πιθανότητες fraud
    thr = float(args.threshold) # default threshold από CLI
    y_pred = (y_score >= thr).astype(int) # μετατροπή σε labels με το default threshold

    # --- Core metrics (threshold-free & threshold-based)
    precision = precision_score(y_test, y_pred, zero_division=0) # precision = TP / (TP + FP) Threshold-based
    recall = recall_score(y_test, y_pred, zero_division=0) # recall = TP / (TP + FN) Threshold-based
    f1 = f1_score(y_test, y_pred, zero_division=0) # F1 = 2·(precision·recall)/(precision+recall) Threshold-based
    roc_auc = roc_auc_score(y_test, y_score) # ROC-AUC Threshold-free
    pr_auc = average_precision_score(y_test, y_score) # PR-AUC (AP) Threshold-free

    # KPIs τύπου SLA (Τα Key Performance Indicators (KPIs) είναι οι συγκεκριμένοι δείκτες που μετρούν την απόδοση ενός οργανισμού, ενώ το Service Level Agreement (SLA) είναι η επίσημη συμφωνία μεταξύ ενός παρόχου υπηρεσιών και του πελάτη που ορίζει τους στόχους απόδοσης και τις συνέπειες για τη μη επίτευξή τους. Τα SLAs θέτουν τις βασικές προσδοκίες απόδοσης, ενώ τα KPIs μετρούν εάν αυτές οι προσδοκίες επιτυγχάνονται. Για παράδειγμα, ένα SLA μπορεί να ορίζει ότι η διαθεσιμότητα ενός συστήματος πρέπει να είναι τουλάχιστον \(99.9\%\) και το αντίστοιχο KPI θα είναι η πραγματική διαθεσιμότητα που μετράται.
    rec_at_p90 = recall_at_precision(y_test, y_score, target_p=0.90) 
    p_at_r90 = precision_at_recall(y_test, y_score, target_r=0.90) 
    # Απευθείας χρήσιμα σε policy συζητήσεις: “αν ζητάμε ≥90% precision, πόσο recall πιάνουμε;”

    # --- Classification report & coefficients
    clf_rep = classification_report(y_test, y_pred, digits=4)
    with open(outdir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(clf_rep)
    # Caveat: οι V1–V28 είναι PCA components, χωρίς ανθρώπινη ερμηνεία. Το Amount συνήθως βγάζει νόημα. Γράψ’ το ρητά στο report για να μη νομίζει κανείς ότι “ανακαλύψαμε αιτιότητα”.
    try:
        coef = pipe.named_steps["clf"].coef_[0]
        coef_df = pd.DataFrame({"feature": X.columns, "coef": coef, "abs_coef": np.abs(coef)})
        coef_df.sort_values("abs_coef", ascending=False).to_csv(outdir / "coefficients_sorted.csv", index=False)
    except Exception as e:
        logging.warning("Αδυναμία εξαγωγής coefficients: %s", e)

    # --- Plots (ROC/PR/Confusion)
    plot_roc(y_test, y_score, figdir / "week9_logreg_roc.png")
    plot_pr(y_test, y_score, figdir / "week9_logreg_pr.png")
    plot_confusion(y_test, y_pred, figdir / "week9_logreg_confusion_default.png", thr=thr)

    # --- Threshold sweep (για reporting/επιχειρηματική οπτική)
    sweep_df = sweep_thresholds( 
        y_true=y_test, y_score=y_score,
        n=int(args.n_thresholds),
        cost_fp=args.cost_fp, cost_fn=args.cost_fn
    ) # παράγει DataFrame με precision/recall/F1/specificity/balanced_accuracy/TP/FP/TN/FN και cost_per_tx για κάθε threshold στο grid
    sweep_csv = outdir / "threshold_sweep.csv"
    sweep_df.to_csv(sweep_csv, index=False)

    # επιπλέον plots
    plot_prec_rec_vs_threshold(sweep_df, figdir / "week9_prec_recall_vs_threshold.png")
    plot_cost_vs_threshold(sweep_df, figdir / "week9_cost_vs_threshold.png")
    # Business νόημα: εδώ “μιλάς” χρήμα. Με ρεαλιστικά --cost-fp & --cost-fn μπορείς να εισηγηθείς πολιτική (π.χ. thr=0.76) στη διοίκηση με cost-driven επιχειρηματολογία.

    # επιλογή “βέλτιστου” threshold (αν ζητηθεί)
    # Αν --optimize ≠ none, διαλέγεις best_thr: 
    # f1: μέγιστο F1 
    # youden: μέγιστο balanced accuracy (proxy του Youden’s J)
    # cost: ελάχιστο cost_per_tx — το πιο business-oriented κριτήριο.
    best_thr = pick_best_threshold(sweep_df, args.optimize)
    best_metrics = None
    if best_thr is not None:
        y_pred_best = (y_score >= best_thr).astype(int)
        plot_confusion(y_test, y_pred_best, figdir / "week9_logreg_confusion_best.png", thr=best_thr) # Υπολογίζεις και δεύτερο confusion για best_thr + best_metrics dict (με TP/FP/TN/FN/precision/recall/F1/specificity/bal_acc/cost)
        best_metrics = threshold_metrics(y_test, y_score, best_thr, args.cost_fp, args.cost_fn)
    # Το script ήδη σημειώνει στο Markdown ότι αυτό γίνεται πάνω στο test για reporting· σε παραγωγή το tuning πρέπει να γίνεται σε validation/CV, με το test “κλειστό”

    # --- Συγκεντρωτικά metrics.json με metadata
    metrics = {
        "model": "LogisticRegression(baseline)",
        "test_size": args.test_size,
        "scaler": args.scaler,
        "class_weight": args.class_weight,
        "threshold_default": thr,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "recall_at_precision_0.90": rec_at_p90,
        "precision_at_recall_0.90": p_at_r90,
        "optimize": args.optimize,
        "best_threshold": best_thr,
        "best_threshold_metrics": best_metrics,
        "cost_fp": args.cost_fp,
        "cost_fn": args.cost_fn,
        "artifacts": {
            "model_path": str(model_path),
            "metrics_json": str((outdir / "metrics.json").resolve()),
            "classification_report": str((outdir / "classification_report.txt").resolve()),
            "coefficients_csv": str((outdir / "coefficients_sorted.csv").resolve()),
            "sweep_csv": str(sweep_csv.resolve()),
            "fig_roc": str((figdir / "week9_logreg_roc.png").resolve()),
            "fig_pr": str((figdir / "week9_logreg_pr.png").resolve()),
            "fig_conf_default": str((figdir / "week9_logreg_confusion_default.png").resolve()),
            "fig_conf_best": str((figdir / "week9_logreg_confusion_best.png").resolve()),
            "fig_prec_rec_vs_thr": str((figdir / "week9_prec_recall_vs_threshold.png").resolve()),
            "fig_cost_vs_thr": str((figdir / "week9_cost_vs_threshold.png").resolve()),
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

    # --- Αποθήκευση πλήρους config (CLI args)
    with open(outdir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    # --- Markdown summary
    md = f"""# Week 9 — Baseline Logistic Regression (Business-Oriented)

**Setup:** scaler={args.scaler}, class_weight={args.class_weight}, threshold={thr},
test_size={args.test_size}, seed={args.seed}, optimize={args.optimize},
n_thresholds={args.n_thresholds}, cost_fp={args.cost_fp}, cost_fn={args.cost_fn}

## Μετρικές (Test set — default threshold {thr:.2f})
- Precision: **{precision:.4f}**
- Recall: **{recall:.4f}**
- F1: **{f1:.4f}**
- ROC-AUC: **{roc_auc:.4f}**
- PR-AUC (AP): **{pr_auc:.4f}**
- Recall@Precision≥0.90: **{rec_at_p90:.4f}**
- Precision@Recall≥0.90: **{p_at_r90:.4f}**

{"## Βέλτιστο threshold ("+args.optimize+") = **"+str(round(best_thr,4))+"**" if best_thr is not None else "## Βέλτιστο threshold: (δεν ζητήθηκε)"}

## Εικόνες
- ROC: `reports/figures/week9/week9_logreg_roc.png`
- PR: `reports/figures/week9/week9_logreg_pr.png`
- Confusion (default): `reports/figures/week9/week9_logreg_confusion_default.png`
- Confusion (best): `reports/figures/week9/week9_logreg_confusion_best.png`
- Precision/Recall vs Threshold: `reports/figures/week9/week9_prec_recall_vs_threshold.png`
- Cost vs Threshold: `reports/figures/week9/week9_cost_vs_threshold.png`

## Αρχεία
- Μοντέλο: `models/logreg_baseline.joblib`
- Μετρικές: `reports/week9_baseline/metrics.json`
- Classification report: `reports/week9_baseline/classification_report.txt`
- Συντελεστές: `reports/week9_baseline/coefficients_sorted.csv`
- Threshold Sweep: `reports/week9_baseline/threshold_sweep.csv`
- Config (CLI args): `reports/week9_baseline/config.json`
- Log: `{log_path.name}`

> Σημείωση: Το threshold tuning εδώ είναι για **αναλυτική αναφορά** πάνω στο test.
> Σε παραγωγική ροή, ορίζουμε validation/CV set για επιλογή threshold και κρατάμε το test “κλειστό”.
"""
    with open(outdir / "09_logreg_baseline_summary.md", "w", encoding="utf-8") as f:
        f.write(md)

    logging.info("Ολοκληρώθηκε run_id=%s. Μετρικές (default thr=%.2f): P=%.4f R=%.4f F1=%.4f ROC-AUC=%.4f PR-AUC=%.4f",
                 run_id, thr, precision, recall, f1, roc_auc, pr_auc)
# --------------------------- Εκτέλεση main ---------------------------
if __name__ == "__main__":
    main()
