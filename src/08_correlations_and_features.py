#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
ΣΚΟΠΟΣ
- Εβδομάδα 8: Συσχετίσεις & Μηχανική Χαρακτηριστικών (Credit Card Fraud).
- Business-oriented artifacts για README/Report/CI: διαγράμματα (PNG), σύνοψη (JSON), εμπλουτισμένο CSV.

ΧΡΗΣΗ 
Ενεργοποίηση venv
Git Bash / WSL / Linux / macOS (Bash/Zsh): source .venv/Scripts/activate
CMD: .venv\Scripts\activate

(Git Bash / WSL / Linux / macOS (Bash/Zsh)):
python src/08_correlations_and_features.py \
--data data/data_raw/creditcard.csv \
--target Class \
--reports reports \
--images reports/figures/week8 \
--out data/data_interim/creditcard_features_week8.csv \
--top-n 15 \
--corr-method pearson \
--corr-sample 0 \
--mi-topn 25

(CMD)
python src/08_correlations_and_features.py ^
--data data\data_raw\creditcard.csv ^
--target Class ^
--reports reports ^
--images reports\figures\week8 ^
--out data\data_interim\creditcard_features_week8.csv ^
--top-n 15 ^
--corr-method pearson ^
--corr-sample 0 ^
--mi-topn 25

Γρήγορος έλεγχος ότι όλα βγήκαν
Στο Git Bash, τρέξε:
ls -la reports/figures/week8
ls -la data/data_interim
head -n 40 reports/week8_corr_summary.json

*Σημειώσεις:*
- Δεν χρησιμοποιείται seaborn — μόνο matplotlib (headless PNG export).
- Mutual Information (MI) απαιτεί scikit-learn. Αν δεν υπάρχει, γίνεται graceful skip.
"""

from __future__ import annotations # Ενεργοποιεί deferred (string) annotations για type hints, βελτιώνοντας ταχύτερα imports/συμβατότητες.

import argparse # CLI flags (reproducibility σε pipelines)
import json
import math # για σταθερές/συναρτήσεις (π.χ. pi)
import logging # τυπικό, μη παρεμβατικό log σε stdout/CI
import platform #
import getpass 
import time # platform/getpass/time: μεταδεδομένα run (OS, χρήστης, timestamps)
import subprocess # εκτέλεση git rev-parse για να δούμε SHA (ιχνηλασιμότητα)
import sys # για sys.exit() σε fatal σφάλματα I/O/εισόδων
from pathlib import Path # cross-platform διαχείριση paths (ανθεκτικότερο από raw strings)
from typing import Dict, Optional # type hints (καθαρότητα κώδικα, ευκολότερο review)
# αριθμητικοί πίνακες & πίνακες δεδομένων—βάση για EDA/featurization
import numpy as np
import pandas as pd
# Matplotlib (headless) & ρυθμίσεις γραφημάτων
import matplotlib
matplotlib.use("Agg")  # Headless PNG export (τρέχει σε server/CI χωρίς GUI)
import matplotlib.pyplot as plt

# Default πιο καθαρές εικόνες χωρίς να γιγαντώνονται τα αρχεία
plt.rcParams.update({"figure.dpi": 150}) # figure.dpi=150: καθαρότερα διαγράμματα στο report, με λογικό μέγεθος αρχείων

# Προαιρετική εισαγωγή MI. Αν λείπει scikit-learn, κάνουμε graceful degradation.
try:
    from sklearn.feature_selection import mutual_info_classif
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False
# αντοχή σε περιβάλλοντα όπου δεν έχουν περάσει όλα τα deps· το script συνεχίζει χωρίς να “σκάει”

# ----------------------------- ΒΟΗΘΗΤΙΚΑ ----------------------------- #

def ensure_dir(p: Path) -> None:
    """Δημιουργεί τον φάκελο (και τους γονικούς) αν δεν υπάρχουν."""
    p.mkdir(parents=True, exist_ok=True)
# λιγότερα I/O σφάλματα, πιο ομαλή εκτέλεση σε CI

def git_short_sha() -> Optional[str]:
    """Επιστρέφει το τρέχον git short SHA (αν υπάρχει repo), αλλιώς None."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        return sha or None
    except Exception:
        return None
# ιχνηλασιμότητα—στο JSON summary/logs ξέρεις την ακριβή έκδοση κώδικα που παρήγαγε τα artifacts

def coerce_binary_target(df: pd.DataFrame, target: str) -> pd.Series:
    """
    Επιστρέφει τη στήλη στόχου ως binary numeric (0/1) αν είναι εφικτό.
    - Στο Kaggle dataset το 'Class' είναι ήδη 0/1 (numeric), οπότε αυτό είναι mostly no-op.
    - Αν δεν είναι numeric, θα προσπαθήσει map σε 0/1 όταν υπάρχουν ακριβώς δύο μοναδικές τιμές.
    """
    s = df[target] # αρχική στήλη στόχου
    if pd.api.types.is_numeric_dtype(s): # αποφεύγει άσκοπες μετατροπές
        return s
    # Attempt to coerce: δυαδική στήλη με δύο μοναδικές τιμές
    uniques = s.dropna().unique()
    if len(uniques) == 2:
        mapping = {uniques[0]: 0, uniques[1]: 1}
        return s.map(mapping) # επιτρέπει ομοιόμορφους υπολογισμούς (corr/MI) ακόμη κι αν έρθει data με labels αντί για 0/1.
    # Τελευταία προσπάθεια: to_numeric (αν έχει '0'/'1' ως strings)
    try:
        return pd.to_numeric(s, errors="raise")
    except Exception:
        raise ValueError(
            f"Target '{target}' δεν είναι numeric και δεν μπορεί να μετατραπεί σε binary 0/1."
        )
    # αποφεύγεις σιωπηλά λάθη/λανθασμένες μετρικές—καλύτερα fail-fast με καθαρό error.

# ----------------------------- PLOTS ----------------------------- #

def plot_corr_heatmap(
    df_numeric: pd.DataFrame,
    save_path: Path,
    title: str,
    vmin: float = -1.0,
    vmax: float = 1.0
) -> None: # Ορίζει συνάρτηση που παίρνει μόνο αριθμητικό DF, path αποθήκευσης, τίτλο και range χρωματικής κλίμακας (default −1..1)
    """
    Heatmap συσχετίσεων για όλες τις αριθμητικές στήλες.
    Business αναγκαιότητα: γρήγορη εικόνα πολυσυσχετίσεων & clusters.
    """
    corr = df_numeric.corr(numeric_only=True) # αποφεύγονται σφάλματα από μη-numeric και παίρνεις standard μέτρο συσχέτισης
    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(corr.values, vmin=vmin, vmax=vmax, cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.columns, fontsize=7)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.026, pad=0.02)
    cbar.ax.set_ylabel("ρ", rotation=0, labelpad=10)
    fig.tight_layout()
    ensure_dir(save_path.parent)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    # Αν έχεις πολλά features, σκέψου να δείξεις heatmap μόνο για υποσύνολο (π.χ. top-variance columns) ή να εφαρμόσεις Spearman σε heavy-tails.

def plot_corr_with_target_barh(
    df_numeric: pd.DataFrame,
    target: str,
    save_path: Path,
    top_n: int
) -> Dict[str, float]: # Φτιάχνει οριζόντιο bar chart με τα top-N |corr| μεταξύ κάθε feature και του target. Επιστρέφει signed corr (με πρόσημο) για το JSON
    """
    Οριζόντιο bar chart με τις κορυφαίες |corr| με τον στόχο.
    Επιστρέφει dict με ΠΡΟΣΗΜΑ (όχι μόνο απόλυτη τιμή) για report-συμπεράσματα.
    """
    corr_df = df_numeric.corr(numeric_only=True)
    if target not in corr_df.columns:
        raise KeyError(
            f"Το target '{target}' δεν βρέθηκε στο correlation matrix. Είναι σίγουρα numeric;"
        )

    corr_s = corr_df[target].drop(labels=[target])  # συσχέτιση κάθε feature με target
    corr_abs_top = corr_s.abs().sort_values(ascending=False).head(top_n) # Ταξινομεί κατά απόλυτη τιμή, κρατά τα top-N (ranking για feature prioritization)

    # Προετοιμασία plot
    fig, ax = plt.subplots(figsize=(10, 6))
    # Σειρά από κάτω προς τα πάνω για να φαίνεται το υψηλότερο στην κορυφή
    ax.barh(corr_abs_top.index[::-1], corr_abs_top.values[::-1])
    ax.set_xlabel(f"|corr| με {target}")
    ax.set_title(f"Top {top_n} απόλυτες συσχετίσεις με {target}")
    fig.tight_layout()
    ensure_dir(save_path.parent)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    # Επιστρέφουμε τα signed corr (για JSON σύνοψη)
    signed_top = corr_s[corr_abs_top.index].sort_values(key=np.abs, ascending=False)
    return signed_top.to_dict()
    # στο report περιγράφεις κατεύθυνση σχέσης (θετική/αρνητική), όχι μόνο μέγεθος
    # Προσοχή: Συσχέτιση ≠ αιτιότητα. Σε ανισόρροπα δεδομένα οι τιμές συχνά είναι χαμηλές αλλά χρήσιμες ως ένδειξη

def plot_fraud_rate_by_hour(
    df_with_hour: pd.DataFrame,
    target: str,
    save_path: Path
) -> Dict[int, float]: # Γραφικό fraud rate ανά ώρα 0–23 + επιστροφή dict {hour: rate} για JSON
    """
    Γράφημα fraud rate ανά ώρα (0–23). Επιστρέφει μικρή σύνοψη dict {hour: fraud_rate}.
    Business value: time-aware insights για adaptive thresholds/rules.
    """
    if "Hour" not in df_with_hour.columns:
        return {} # Αν δεν έχεις feature Hour, βγες ήσυχα (graceful)
    g = (
        df_with_hour.groupby("Hour")[target] # ομαδοποίηση ανά ώρα
        .agg(["mean", "count", "sum"]) # υπολογισμός μέσου (fraud rate), πλήθους, αθροίσματος (αριθμός απάτης)
        .rename(columns={"mean": "fraud_rate"}) # μετονομασία για σαφήνεια
        .sort_index() # για 0→23 σειρά
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(g.index, g["fraud_rate"], marker="o")
    ax.set_xticks(range(0, 24, 1))
    ax.set_xlabel("Ώρα ημέρας (0–23)")
    ax.set_ylabel("Fraud rate")
    ax.set_title("Ποσοστό απάτης ανά ώρα")
    fig.tight_layout()
    ensure_dir(save_path.parent)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    return {int(h): float(fr) for h, fr in g["fraud_rate"].round(6).to_dict().items()}
    # αν δεις spikes σε συγκεκριμένες ώρες, μπορείς να εφαρμόσεις time-aware thresholds ή κανόνες (π.χ. πιο αυστηρό screening 01:00–04:00), να κατανείμεις βάρδιες/πόρους monitoring κ.λπ.
    # Πρόσεχε: ερμήνευε μαζί με count—μην βασίζεσαι σε λίγες παρατηρήσεις.

# ----------------------------- FEATURE ENGINEERING ----------------------------- #

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Προσθέτει:
      - Hour από Time (με modulo 86400, bucket ανά ώρα)
      - Κυκλική κωδικοποίηση Hour (hour_sin, hour_cos)
      - Amount_log1p (σταθεροποίηση ασυμμετρίας)
    """
    out = df.copy() # Κάνει αντίγραφο για να μη μεταβάλλει το αρχικό DF (ασφαλέστερο σε pipelines)

    # Time -> Hour (0–23) & κυκλική κωδικοποίηση
    if "Time" in out.columns:
        # Time = δευτερόλεπτα από 1η συναλλαγή (2 ημέρες στο Kaggle). Χρησιμοποιούμε 24*3600 = 86400.
        hour = ((out["Time"] % 86400) // 3600).astype(int)
        out["Hour"] = hour
        theta = 2 * math.pi * (hour / 24.0)
        out["hour_sin"] = np.sin(theta)
        out["hour_cos"] = np.cos(theta) 
        # αυτά συχνά βελτιώνουν linear/trees μοντέλα (καλύτερη “κατανόηση” ότι 23:00 και 00:00 είναι κοντά)

    # Amount -> Amount_log1p (αν δεν υπάρχει)
    if "Amount" in out.columns and "Amount_log1p" not in out.columns:
        out["Amount_log1p"] = np.log1p(out["Amount"].clip(lower=0))
        # σταθεροποιεί training, βελτιώνει συμπεριφορά μοντέλων & thresholding
    return out
    # Αν θες πιο “λεπτή” ώρα (π.χ. 48 buckets μισάωρων), άλλαξε τον διαιρέτη

# ----------------------------- MUTUAL INFORMATION ----------------------------- #
# Mutual information is a measure of the inherent dependence expressed in the joint distribution of and relative to the marginal distribution of and under the assumption of independence.
def compute_mutual_information(
    df_numeric: pd.DataFrame,
    target: str,
    topn: int,
    sample: int = 0,
) -> Optional[pd.Series]:
    """
    Υπολογίζει Mutual Information (MI) μεταξύ (X=num_στήλες) και y=target.
    - Κάνει fillna(0) για απλότητα (εναλλακτικά: imputer πριν).
    - Αν οριστεί sample>0, παίρνει τυχαίο υποσύνολο για επιτάχυνση.
    Επιστρέφει pd.Series ταξινομημένο φθίνουσα (topn).
    """
    if not SKLEARN_AVAILABLE:
        logging.warning("scikit-learn δεν είναι εγκατεστημένο: παράλειψη MI.")
        return None # Αν δεν υπάρχει scikit-learn, γίνεται graceful skip. (Το script συνεχίζει κανονικά.

    num_cols = [c for c in df_numeric.columns if c != target]
    if len(num_cols) == 0:
        logging.warning("Δεν βρέθηκαν αριθμητικές στήλες (πλην target) για MI.")
        return None # Παίρνει όλες τις numeric στήλες εκτός target. Αν δεν υπάρχουν, σταματά

    X = df_numeric[num_cols].fillna(0)
    y_raw = df_numeric[target]
    if not pd.api.types.is_numeric_dtype(y_raw):
        logging.warning("Target για MI δεν είναι numeric — παράλειψη MI.")
        return None # Γεμίζει κενά με 0 (απλή επιλογή για να μη “σκάσει” ο αλγόριθμος—σε production θα έβαζες imputer). Ελέγχει ότι ο στόχος είναι numeric (binary 0/1)

    # Προαιρετικό sampling για ταχύτητα
    if sample and len(X) > sample:
        X = X.sample(sample, random_state=42)
        y_raw = y_raw.loc[X.index] # Sampling για ταχύτητα σε μεγάλα σύνολα (reproducible με random_state)

    try:
        mi_vals = mutual_info_classif(X.values, y_raw.values, random_state=42)
        mi = pd.Series(mi_vals, index=num_cols).sort_values(ascending=False)
        return mi.head(topn)
    except Exception as e:
        logging.warning(f"Αποτυχία υπολογισμού MI: {e}")
        return None
    # Συμπληρώνει την εικόνα των συσχετίσεων όταν υπάρχουν μη γραμμικές σχέσεις (π.χ. thresholds, interactions)

# ----------------------------- MAIN ----------------------------- #

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser() # CLI flags για ευκολότερη εκτέλεση σε pipelines/CI
    ap.add_argument("--data", required=True, type=str, help="Διαδρομή προς creditcard.csv")
    ap.add_argument("--target", default="Class", type=str, help="Όνομα στήλης-στόχου (binary)")
    ap.add_argument("--reports", default="reports", type=str, help="Φάκελος report artifacts (JSON/CSV)")
    ap.add_argument("--images", default="reports/figures/week8", type=str, help="Φάκελος εικόνων (PNGs)")
    ap.add_argument("--out", default="data/data_interim/creditcard_features_week8.csv", type=str, help="Εμπλουτισμένο CSV out")
    ap.add_argument("--top-n", default=15, type=int, help="Πλήθος κορυφαίων |corr| να εμφανιστούν/επιστραφούν")
    ap.add_argument("--corr-method", default="pearson", choices=["pearson", "spearman"], help="Μέθοδος συσχέτισης")
    ap.add_argument("--corr-sample", default=0, type=int, help="Sampling Ν για corr/plots (0 = χωρίς sampling)")
    ap.add_argument("--mi-topn", default=25, type=int, help="Top-N Mutual Information features (0 = skip)")
    ap.add_argument("--no-plots", action="store_true", help="Απενεργοποίηση όλων των plots (χρήσιμο σε CI)")
    return ap.parse_args()


def main():
    args = parse_args()

    # --- Setup paths ---
    data_path = Path(args.data)
    reports_dir = Path(args.reports)
    images_dir = Path(args.images)
    out_path = Path(args.out)
    summary_dir = reports_dir / "week8_corr_summary"

    ensure_dir(reports_dir); ensure_dir(images_dir); ensure_dir(out_path.parent)

    # --- Logging metadata (business-ready auditability) ---
    # Auditability is the ability of a system, process, or organization to be independently and accurately reviewed to verify its activities and ensure compliance with rules, regulations, and standards. It is achieved through accessible records, transparent operations, and thorough documentation, which allow for the tracing of actions, decisions, and changes. This is a critical concept in fields like finance and AI, where accountability, transparency, and compliance are essential
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_meta = {
        "runner": getpass.getuser(), # (ποιος)τρέχων χρήστης OS
        "python": platform.python_version(), # (που) έκδοση Python
        "platform": platform.platform(), # (πότε) πληροφορίες OS
        "script": Path(__file__).name, # όνομα script
        "git_sha": git_short_sha(), # (τι) git SHA (αν υπάρχει)
        "start_ts": int(time.time()), # timestamp έναρξης
        "args": vars(args), # παράμετροι CLI
    }
    logging.info(f"Run meta: {json.dumps(run_meta, ensure_ascii=False)}")
    # ιχνηλασιμότητα (SOX/εσωτερικά audits), reproducibility σε reports.

    # --- Load data ---
    if not data_path.exists():
        logging.error(f"Το αρχείο δεδομένων δεν υπάρχει: {data_path}")
        sys.exit(1)

    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        logging.error(f"Αποτυχία φόρτωσης CSV: {e}")
        sys.exit(1) # Καθαρό catch των I/O/parse σφαλμάτων → predictable CI.

    target = args.target
    if target not in df.columns:
        logging.error(f"Target '{target}' δεν βρέθηκε στο dataset.")
        sys.exit(1) # Schema guard (συχνό failure point σε πραγματικά data streams).

    # Διασφάλιση binary-αριθμητικού target για corr/MI
    try:
        df[target] = coerce_binary_target(df, target)
    except ValueError as e:
        logging.error(str(e))
        sys.exit(1) # Εξασφαλίζεις ότι downstream μετρικές (corr/MI) έχουν σωστό τύπο. Business: data hygiene, αποφυγή σιωπηλών bugs.

    rows, cols = df.shape
    fraud_rate_overall = float(df[target].mean()) if rows > 0 else 0.0
    logging.info(f"Shape={rows}x{cols}, FraudRate={fraud_rate_overall:.6f}")
    # Γρήγορο health-check (μέγεθος & βασικός δείκτης κινδύνου). Βάζεις fraud rate στο log για monitoring.

    # Έλεγχος NaNs (top10) για data quality αναφορά
    try:
        nan_ratio = df.isna().mean().sort_values(ascending=False)
        nan_ratio.head(10).to_csv(summary_dir / "week8_nan_ratio_top10.csv", header=["nan_ratio"])
    except Exception as e:
        logging.warning(f"Αποτυχία αποθήκευσης nan_ratio: {e}")
    # mini-report για ελλείψεις. Business: τεκμηρίωση ποιότητας (καίριο για risk)

    # Προετοιμασία numeric DF (και optional sampling για corr/MI)
    df_numeric = df.select_dtypes(include=[np.number])
    if args.corr_sample and len(df_numeric) > args.corr_sample:
        df_corr_mi = df_numeric.sample(args.corr_sample, random_state=42)
        logging.info(f"Χρησιμοποιείται sampling {args.corr_sample} γραμμών για corr/MI.")
    else:
        df_corr_mi = df_numeric
    # Στοχεύεις μόνο numeric features (όπως απαιτεί corr/MI). Sampling = ελεγχόμενη επιτάχυνση σε μεγάλα data.

    # --- 1) Heatmap συσχετίσεων ---
    images_paths = {}
    if not args.no_plots:
        try:
            # Αν method == 'spearman', θα φαίνεται μόνο στον τίτλο (η pandas.corr με method ανατίθεται αργότερα)
            # Εδώ το heatmap είναι πάνω σε pearson του df_numeric (σταθερή εικόνα όλων των numeric).
            # Για συνέπεια με method, μπορούμε να εφαρμόσουμε το method που επέλεξε ο χρήστης:
            corr = df_corr_mi.corr(method=args.corr_method, numeric_only=True)
            fig, ax = plt.subplots(figsize=(16, 14))
            im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
            ax.set_yticklabels(corr.columns, fontsize=7)
            ax.set_title(f"Correlation heatmap ({args.corr_method})")
            cbar = fig.colorbar(im, ax=ax, fraction=0.026, pad=0.02)
            cbar.ax.set_ylabel("ρ", rotation=0, labelpad=10)
            fig.tight_layout()
            heatmap_path = images_dir / "corr_heatmap.png"
            fig.savefig(heatmap_path, bbox_inches="tight")
            plt.close(fig)
            images_paths["corr_heatmap"] = str(heatmap_path)
        except Exception as e:
            logging.warning(f"Αποτυχία δημιουργίας heatmap: {e}")
    # Υπολογίζεις corr (με επιλεγμένη μέθοδο) στο (πιθανά) δειγματοληπτημένο numeric DF, ζωγραφίζεις το matrix, σώζεις PNG.
    # Business: οπτική χαρτογράφηση σχέσεων/συστάδων, έτοιμη για report.

    # --- 2) |corr| με την κλάση (με πρόσημα) + CSV ---
    top_corr_signed_dict: Dict[str, float] = {}
    try:
        # Corr με επιλεγμένη μέθοδο (pearson/spearman)
        corr_full = df_corr_mi.corr(method=args.corr_method, numeric_only=True)
        if target not in corr_full.columns:
            raise KeyError(f"Target '{target}' απουσιάζει από corr matrix.")
        corr_with_target = corr_full[target].drop(labels=[target]).sort_values(key=np.abs, ascending=False)
        # Αποθήκευση CSV με όλες τις συσχετίσεις προς το target (signed)
        corr_with_target.to_csv(summary_dir / "week8_corr_with_target.csv", header=["corr"])
    # Υπολογίζεις signed συσχέτιση για όλα τα features προς τον στόχο και την αποθηκεύεις (CSV). Business: machine-readable ranking για downstream auto-reports/plots.
    
        # Plot top-N (barh)
        if not args.no_plots:
            top_corr_signed_dict = plot_corr_with_target_barh(
                df_corr_mi, target, images_dir / "corr_with_target.png", top_n=args.top_n
            )
            images_paths["corr_with_target"] = str(images_dir / "corr_with_target.png")
        else:
            top_corr_signed_dict = corr_with_target.head(args.top_n).to_dict()
    except Exception as e:
        logging.warning(f"Αποτυχία corr προς target: {e}")
    # Φτιάχνεις bar chart (ή μόνο dict σε CI). Επιστρέφεις top-N signed για το JSON. Business: γρήγορη ιεράρχηση για stakeholders.

    # --- 3) Feature engineering ---
    df_feats = engineer_features(df)
    # Προσθέτεις Hour, hour_sin, hour_cos, Amount_log1p. Business: καλύτερη αναπαράσταση χρονικών/ποσοτικών patterns για μοντέλα & rules

    # --- 4) Fraud-rate ανά ώρα ---
    fraud_rate_by_hour_dict: Dict[int, float] = {} # Επιστρέφεται στο JSON summary
    counts_by_hour_dict: Optional[Dict[int, int]] = None # Επιστρέφεται στο JSON summary
    try:
        if "Hour" in df_feats.columns:
            # Plot
            if not args.no_plots:
                fraud_rate_by_hour_dict = plot_fraud_rate_by_hour(
                    df_feats, target, images_dir / "fraud_rate_by_hour.png"
                )
                images_paths["fraud_rate_by_hour"] = str(images_dir / "fraud_rate_by_hour.png")
            else:
                g = (
                    df_feats.groupby("Hour")[target]
                    .mean()
                    .sort_index()
                    .round(6)
                )
                fraud_rate_by_hour_dict = {int(h): float(fr) for h, fr in g.to_dict().items()}
            # Counts per hour (για context)
            counts_by_hour_series = df_feats.groupby("Hour")[target].size().sort_index()
            counts_by_hour_dict = {int(h): int(c) for h, c in counts_by_hour_series.to_dict().items()}
    except Exception as e:
        logging.warning(f"Αποτυχία fraud-rate-by-hour: {e}")
    # Παράγεις operational insight (fraud rate 0–23) και παρέχεις counts για context (μην ερμηνεύεις spikes με 3 δείγματα).
    # Operational insight is the process of using data analysis to understand and improve business operations, identify inefficiencies, and make better-informed decisions. It involves analyzing data from various sources to monitor key performance indicators (KPIs), perform root cause analysis on issues, and implement changes to increase efficiency and productivity. For example, it can be used to analyze production speed, quality control, and resource usage, or to monitor and troubleshoot IT infrastructure and systems. 

    # --- 5) Αποθήκευση εμπλουτισμένου dataset ---
    try:
        df_feats.to_csv(out_path, index=False)
    except Exception as e:
        logging.error(f"Αποτυχία αποθήκευσης εμπλουτισμένου CSV: {e}")
        sys.exit(1) # interim dataset για επόμενα βήματα (training/EDA). Fail-fast σε I/O errors

    # --- 6) Mutual Information (προαιρετικό, αν υπάρχει sklearn και mi-topn>0) ---
    mi_topn_series: Optional[pd.Series] = None
    if args.mi_topn and args.mi_topn > 0:
        mi_topn_series = compute_mutual_information(
            df_corr_mi, target, topn=args.mi_topn, sample=args.corr_sample
        )
        if mi_topn_series is not None:
            try:
                mi_topn_series.to_csv(summary_dir  / "week8_mutual_info_topN.csv", header=["mi"])
            except Exception as e:
                logging.warning(f"Αποτυχία αποθήκευσης MI CSV: {e}")
    # Υπολογίζεις MI top-N (αν διαθέσιμο) & export. Business: συμπληρώνεις corr με non-linear dependencies.

    # --- 7) JSON σύνοψη (για README/report ingest) ---
    summary = {
        "script_version": git_short_sha() or "week8_v2",
        "data_path": str(data_path),
        "rows": int(rows),
        "cols": int(cols),
        "fraud_rate_overall": fraud_rate_overall,
        "corr": {
            "method": args.corr_method,
            "sample": int(args.corr_sample),
            "top_n": int(args.top_n),
            "top_corr_with_target_signed": top_corr_signed_dict,  # dict feature->corr (με πρόσημο)
            "csv_all_signed": str((summary_dir  / "week8_corr_with_target.csv").as_posix()),
        },
        "engineered_features": [c for c in ["Hour", "hour_sin", "hour_cos", "Amount_log1p"] if c in df_feats.columns],
        "fraud_by_hour": {
            "rates": fraud_rate_by_hour_dict or None,
            "counts": counts_by_hour_dict or None,
        },
        "mi": {
            "computed": bool(mi_topn_series is not None),
            "topn": int(args.mi_topn),
            "csv": str((summary_dir  / "week8_mutual_info_topN.csv").as_posix()) if mi_topn_series is not None else None,
        },
        "images": images_paths if not args.no_plots else {},
        "out_dataset": str(out_path.as_posix()),
        "nan_ratio_top10_csv": str((summary_dir  / "week8_nan_ratio_top10.csv").as_posix()),
        "run_meta": run_meta,
        "end_ts": int(time.time()),
        "duration_sec": int(time.time() - run_meta["start_ts"]),
    } # Ένα ενιαίο machine-readable summary για το README/τελικό report: τι τρέξαμε, με ποια ρύθμιση, τι παράχθηκε, πόσο κράτησε, ποιος το έτρεξε, ποια έκδοση κώδικα. Business: audit, reproducibility, governance.

    try:
        with open(summary_dir  / "week8_corr_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.warning(f"Αποτυχία αποθήκευσης summary JSON: {e}") # Καθαρή αποθήκευση του summary. Αν αποτύχει, δεν ρίχνεις όλο το run (warning).

    logging.info("✅ Ολοκληρώθηκε η Εβδομάδα 8 (artifacts δημιουργήθηκαν επιτυχώς).")
    print("[OK] Week 8 artifacts created.")
    # Δίνεις και log και stdout μήνυμα — βοηθά σε ανθρώπινη ανάγνωση και σε scripts που ψάχνουν συγκεκριμένο token.

if __name__ == "__main__":
    main()
# Entry point συμβατό με import (ώστε να μπορείς να το καλέσεις από tests χωρίς να εκτελεστεί).