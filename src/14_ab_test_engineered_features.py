# src/14_ab_test_engineered_features.py
"""
Είναι ένα A/B test harness που:
- παίρνει τα ίδια Week 13 splits
- εκπαιδεύει το ίδιο μοντέλο 2 φορές:
    A) baseline features
    B) baseline + engineered features (week14)
- επιλέγει threshold από validation με constraint precision ≥ 0.90
- μετρά τελικό αποτέλεσμα στο test
- αποθηκεύει τα αποτελέσματα σε CSV/JSON
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix,
)

from feature_engineering_week14 import Week14FeatureEngineer
# fit μόνο στο train (amount stats), transform παντού

def load_split_csv(path: Path, target: str) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in {path}")
    X = df.drop(columns=[target])
    y = df[target].values
    return X, y # ακριβώς τα ίδια splits για baseline και engineered → δίκαιη σύγκριση


def select_threshold_precision_constraint(y_true: np.ndarray, y_score: np.ndarray, precision_target: float = 0.90) -> dict:
    """
    Choose threshold on VALIDATION:
    - among points where precision >= target, pick the one with max recall.
    Returns dict with chosen threshold + achieved precision/recall.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    # precision/recall arrays have len = len(thresholds)+1
    # align to thresholds by ignoring last precision/recall point
    precision_t = precision[:-1]
    recall_t = recall[:-1] 

    ok = precision_t >= precision_target
    if not np.any(ok):
        # no threshold meets constraint
        best_idx = int(np.argmax(precision_t))  # fallback: max precision
        return {
            "threshold": float(thresholds[best_idx]),
            "precision": float(precision_t[best_idx]),
            "recall": float(recall_t[best_idx]),
            "meets_constraint": False,
        }

    best_idx = int(np.argmax(recall_t[ok]))
    ok_indices = np.where(ok)[0]
    chosen = ok_indices[best_idx]


    return {
        "threshold": float(thresholds[chosen]),
        "precision": float(precision_t[chosen]),
        "recall": float(recall_t[chosen]),
        "meets_constraint": True,
    } 
# Επιλέγεις το threshold που: κρατά precision ≥ target και δίνει max recall. (“Θέλω όσο το δυνατόν περισσότερα fraud, αλλά μόνο αν κρατάω false alarms χαμηλά.”)

def train_xgb(X_train: pd.DataFrame, y_train: np.ndarray, seed: int, tuned_params: dict | None = None):
    from xgboost import XGBClassifier

    # imbalance handling
    pos = max(1, int((y_train == 1).sum()))
    neg = max(1, int((y_train == 0).sum()))
    default_spw = neg / pos

    if tuned_params is None:
        tuned_params = {}

    # defaults (μόνο αν δεν υπάρχουν ήδη στο tuned_params)
    tuned_params = dict(tuned_params)  # copy
    tuned_params.setdefault("n_estimators", 600)
    tuned_params.setdefault("max_depth", 4)
    tuned_params.setdefault("learning_rate", 0.05)
    tuned_params.setdefault("subsample", 0.8)
    tuned_params.setdefault("colsample_bytree", 0.8)
    tuned_params.setdefault("reg_lambda", 1.0)
    tuned_params.setdefault("reg_alpha", 0.0)
    tuned_params.setdefault("min_child_weight", 1.0)
    tuned_params.setdefault("gamma", 0.0)
    tuned_params.setdefault("objective", "binary:logistic")
    tuned_params.setdefault("eval_metric", "aucpr")
    tuned_params.setdefault("tree_method", "hist")
    tuned_params.setdefault("scale_pos_weight", default_spw)

    model = XGBClassifier(
        **tuned_params,
        random_state=seed,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    return model


def evaluate(model, X: pd.DataFrame, y: np.ndarray, threshold: float) -> dict:
    proba = model.predict_proba(X)[:, 1]
    # Metrics independent of threshold
    pr_auc = float(average_precision_score(y, proba))
    roc_auc = float(roc_auc_score(y, proba))
    # Apply threshold → binary predictions
    y_hat = (proba >= threshold).astype(int)
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y, y_hat).ravel()
    # Precision/Recall από confusion matrix
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    return {
        "threshold": float(threshold),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "precision": float(precision),
        "recall": float(recall),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def run_one(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    seed: int,
    precision_target: float,
    tuned_params: dict | None = None,
) -> dict:
    model = train_xgb(X_train, y_train, seed=seed, tuned_params=tuned_params)

    val_scores = model.predict_proba(X_val)[:, 1]
    thr_info = select_threshold_precision_constraint(y_val, val_scores, precision_target=precision_target)

    test_metrics = evaluate(model, X_test, y_test, threshold=thr_info["threshold"])
    return {"threshold_selection": thr_info, "test_metrics": test_metrics}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-path", default="data/data_interim/train.csv")
    ap.add_argument("--val-path", default="data/data_interim/val.csv")
    ap.add_argument("--test-path", default="data/data_interim/test.csv")
    ap.add_argument("--target", default="Class")
   # ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42], help="List of random seeds (e.g., --seeds 41 42 43)")
    ap.add_argument("--tuned-params", default=None, help="Path to JSON with tuned XGB params from Week 13")
    ap.add_argument("--no-amount-scaled", action="store_true", help="Disable amount_scaled feature for engineered run")
    ap.add_argument("--precision-target", type=float, default=0.90)
    ap.add_argument("--outdir", default="reports/week14_ab_engineered")
    args = ap.parse_args()

    print("SEEDS ARGPARSE:", args.seeds)

    tuned_params = None
    if args.tuned_params:
     tuned_params = json.loads(Path(args.tuned_params).read_text(encoding="utf-8"))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Load Week 13 splits (CSV)
    X_train, y_train = load_split_csv(Path(args.train_path), args.target)
    X_val, y_val = load_split_csv(Path(args.val_path), args.target)
    X_test, y_test = load_split_csv(Path(args.test_path), args.target)

    # Keep baseline feature schema = original columns only (defensive)
    original_cols = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
    baseline_cols = [c for c in original_cols if c in X_train.columns]

    X_train_base = X_train[baseline_cols]
    X_val_base = X_val[baseline_cols]
    X_test_base = X_test[baseline_cols]

    rows = []

    for seed in args.seeds:
        print("RUNNING SEED:", seed)
    # 2) Baseline run (no engineered features)
        baseline = run_one(
            X_train_base, y_train,
            X_val_base, y_val,
            X_test_base, y_test,
            seed=seed,
            precision_target=args.precision_target,
            tuned_params=tuned_params,
    )

    # 3) Engineered (optional: no amount_scaled)
        fe = Week14FeatureEngineer(
            use_amount=True,
            use_time=True,
            use_amount_scaled=(not args.no_amount_scaled),
        ).fit(X_train_base)

        X_train_eng = fe.transform(X_train_base)
        X_val_eng   = fe.transform(X_val_base)
        X_test_eng  = fe.transform(X_test_base)

        engineered = run_one(
            X_train_eng, y_train,
            X_val_eng, y_val,
            X_test_eng, y_test,
            seed=seed,
            precision_target=args.precision_target,
            tuned_params=tuned_params,
        )

    # 4) Compare + export
        for name, obj in [("baseline", baseline), ("engineered", engineered)]:
            m = obj["test_metrics"]
            rows.append({
                "seed": seed,
                "run": name,
                "thr_meets_precision_constraint": obj["threshold_selection"]["meets_constraint"],
                "thr_val_precision": obj["threshold_selection"]["precision"],
                "thr_val_recall": obj["threshold_selection"]["recall"],
                **m,
            })

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "ab_results_by_seed.csv", index=False)

    summary = df.groupby("run")[["pr_auc","roc_auc","precision","recall","fp","fn","tp"]].agg(["mean","std"])
    summary.to_csv(outdir / "ab_results_summary.csv")

    print("\n=== WEEK 14 A/B RESULTS (BY SEED) ===")
    print(df.to_string(index=False))
    print("\n=== SUMMARY (mean ± std) ===")
    print(summary)
    print(f"\nSaved: {outdir / 'ab_results_by_seed.csv'}")
    print(f"Saved: {outdir / 'ab_results_summary.csv'}")


if __name__ == "__main__":
    main()
