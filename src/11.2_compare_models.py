#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Week 11 — Model Comparison (LogReg vs Decision Tree vs Random Forest)
--------------------------------------------------------------------

Σκοπός:
- Φτιάχνει δίκαιη σύγκριση 3 μοντέλων με ίδια KPIs και ίδια “decision policy”.
- Policy A: threshold=0.50 (default)
- Policy B: threshold που ελαχιστοποιεί cost/transaction (cost_fp, cost_fn)

Σημαντικό (feature schema drift):
- Τα models (Week9/Week10) μπορεί να έχουν εκπαιδευτεί με διαφορετικά features από το τρέχον test.csv.
- Αυτό το script κάνει feature alignment per-model:
  - drop extra columns (unseen at fit)
  - add missing columns (seen at fit but missing now) filled with 0.0
  - enforce column order exactly as during fit

Usage (με τα δικά σου paths):
python src/12_compare_models.py \
  --test-csv data/data_interim/test.csv \
  --target Class \
  --outdir reports/week11_model_comparison \
  --cost-fp 1 --cost-fn 20 --n-thresholds 101 \
  --logreg models/logreg_baseline.joblib \
  --dt models/dt_baseline.joblib \
  --rf models/rf_runF_balanced.joblib
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


# --------------------------- Data ---------------------------

def load_xy(csv_path: Path, target: str):
    df = pd.read_csv(csv_path)
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in {csv_path}")
    X = df.drop(columns=[target])
    y = df[target].astype(int).values
    return X, y


# --------------------- Feature Alignment ---------------------

def get_expected_features(model):
    """
    Επιστρέφει τα feature names που είδε το μοντέλο στο fit (αν υπάρχουν).
    Για Pipeline: ψάχνει τα steps από το τέλος προς τα πίσω.
    """
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if hasattr(model, "named_steps"):
        for step in reversed(list(model.named_steps.values())):
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)

    return None


def align_X_to_model(model, X: pd.DataFrame):
    """
    Ευθυγραμμίζει τα features του X με αυτά που περιμένει το μοντέλο:
    - αφαιρεί extra columns (που δεν υπήρχαν στο fit)
    - προσθέτει missing columns με 0.0
    - επιβάλλει σωστή σειρά στηλών
    """
    expected = get_expected_features(model)
    if expected is None:
        return X, {"dropped": [], "missing_filled": [], "expected_count": None}

    X_aligned = X.copy()

    dropped = [c for c in X_aligned.columns if c not in expected]
    if dropped:
        X_aligned = X_aligned.drop(columns=dropped)

    missing = [c for c in expected if c not in X_aligned.columns]
    if missing:
        for c in missing:
            X_aligned[c] = 0.0

    X_aligned = X_aligned[expected]
    return X_aligned, {"dropped": dropped, "missing_filled": missing, "expected_count": len(expected)}


# --------------------------- Metrics ---------------------------

def threshold_metrics(y_true, y_score, thr: float, cost_fp: float, cost_fn: float):
    y_pred = (y_score >= thr).astype(int)

    # labels=[0,1] για σιγουριά ότι θα είναι πάντα 2x2
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, pos_label=1, average="binary", zero_division=0
    )

    cost = (fp * cost_fp + fn * cost_fn) / len(y_true)

    return {
        "threshold": float(thr),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "cost_per_tx": float(cost),
    }


def pick_best_threshold_cost(y_true, y_score, n: int, cost_fp: float, cost_fn: float):
    """
    Επιλέγει threshold που ελαχιστοποιεί cost/tx.
    """
    thrs = np.linspace(0.01, 0.99, n)
    rows = [threshold_metrics(y_true, y_score, float(t), cost_fp, cost_fn) for t in thrs]
    df = pd.DataFrame(rows)
    best = df.loc[df["cost_per_tx"].idxmin()]
    return float(best["threshold"]), df


def get_scores(model, X: pd.DataFrame):
    """
    Παίρνει continuous score για thresholding:
    - predict_proba αν υπάρχει
    - αλλιώς decision_function normalised σε [0,1]
    - αλλιώς predict (0/1) ως fallback
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        # binary: παίρνουμε την πιθανότητα της θετικής κλάσης
        return proba[:, 1]

    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        s = (s - np.min(s)) / (np.max(s) - np.min(s) + 1e-12)
        return s

    return model.predict(X).astype(float)


# --------------------------- Main ---------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Compare LogReg vs DT vs RF with shared decision policy.")
    ap.add_argument("--test-csv", required=True, help="Path to test CSV with target column.")
    ap.add_argument("--target", default="Class", help="Target column name.")
    ap.add_argument("--outdir", default="reports/week11_model_comparison", help="Output directory.")
    ap.add_argument("--cost-fp", type=float, default=1.0, help="Cost of False Positive.")
    ap.add_argument("--cost-fn", type=float, default=20.0, help="Cost of False Negative.")
    ap.add_argument("--n-thresholds", type=int, default=101, help="Threshold sweep points (0.01..0.99).")

    ap.add_argument("--logreg", required=True, help="Path to Logistic Regression joblib (can be Pipeline).")
    ap.add_argument("--dt", required=True, help="Path to Decision Tree joblib.")
    ap.add_argument("--rf", required=True, help="Path to Random Forest joblib.")
    ap.add_argument("--verbose", action="store_true", help="Print extra debug info (feature alignment).")

    return ap.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    X_test, y_test = load_xy(Path(args.test_csv), args.target)

    # Load models
    models = {
        "LogReg": joblib.load(args.logreg),
        "DecisionTree": joblib.load(args.dt),
        "RandomForest": joblib.load(args.rf),
    }

    rows = []
    sweeps = {}

    for name, m in models.items():
        # Align features to what THIS model expects (handles schema drift)
        X_use, info = align_X_to_model(m, X_test)

        if args.verbose and (info["dropped"] or info["missing_filled"]):
            print(f"\n[{name}] feature alignment:")
            if info["expected_count"] is not None:
                print(f"  expected_features: {info['expected_count']}")
            if info["dropped"]:
                print("  dropped(extra):", info["dropped"])
            if info["missing_filled"]:
                print("  filled_missing_with_0:", info["missing_filled"])

        # Continuous scores
        y_score = get_scores(m, X_use)

        # Threshold-independent metrics (ranking quality)
        roc_auc = roc_auc_score(y_test, y_score)
        pr_auc = average_precision_score(y_test, y_score)

        # Policy A: default thr=0.50
        m50 = threshold_metrics(y_test, y_score, 0.50, args.cost_fp, args.cost_fn)

        # Policy B: cost-optimal threshold
        best_thr, sweep_df = pick_best_threshold_cost(y_test, y_score, args.n_thresholds, args.cost_fp, args.cost_fn)
        mbest = threshold_metrics(y_test, y_score, best_thr, args.cost_fp, args.cost_fn)

        rows.append({
            "model": name,
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),

            "thr_0.50_precision": m50["precision"],
            "thr_0.50_recall": m50["recall"],
            "thr_0.50_f1": m50["f1"],
            "thr_0.50_tp": m50["tp"],
            "thr_0.50_fp": m50["fp"],
            "thr_0.50_fn": m50["fn"],
            "thr_0.50_cost_per_tx": m50["cost_per_tx"],

            "thr_cost_best": float(best_thr),
            "thr_cost_precision": mbest["precision"],
            "thr_cost_recall": mbest["recall"],
            "thr_cost_f1": mbest["f1"],
            "thr_cost_tp": mbest["tp"],
            "thr_cost_fp": mbest["fp"],
            "thr_cost_fn": mbest["fn"],
            "thr_cost_cost_per_tx": mbest["cost_per_tx"],

            "dropped_features": ",".join(info["dropped"]) if info["dropped"] else "",
            "missing_features_filled": ",".join(info["missing_filled"]) if info["missing_filled"] else "",
        })

        sweeps[name] = sweep_df

    comp = pd.DataFrame(rows).sort_values("thr_cost_cost_per_tx")
    comp_path = outdir / "model_comparison.csv"
    comp.to_csv(comp_path, index=False)

    # Save sweeps
    for name, df in sweeps.items():
        df.to_csv(outdir / f"{name}_threshold_sweep.csv", index=False)

    # Save JSON
    (outdir / "model_comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print("\n=== Model comparison saved ===")
    print(f"- {comp_path.as_posix()}")
    print(comp.to_string(index=False))


if __name__ == "__main__":
    main()
