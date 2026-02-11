""" Evaluation Harness - Utility functions for splits, metrics, thresholding, tuning, and evaluation.
επιστημονική σύγκριση “γιατί αυτό το μοντέλο είναι καλύτερο;”
-ίδια split
-ίδια scoring
-ίδιο threshold logic
-καλύτερο PR-AUC
-καλύτερο recall υπό precision constraint
"""
import time # for runtime measurement
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass # for structured config/params if desired (not used in this example but can be helpful)
from typing import Dict, Any, Optional, Tuple, List # for type hints

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_recall_curve, roc_curve,
    precision_score, recall_score, f1_score, fbeta_score,
    confusion_matrix
)

from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

# Set a global random state for reproducibility across all functions that use randomness (splits, tuning, etc.)
RANDOM_STATE = 42

# ---------------------------
# JSON safe helper (fix for numpy types in params)
# ---------------------------
def _json_safe(x):
    """
    Convert numpy types (np.int64, np.float64, etc.)
    to native Python types so json.dumps works.
    """
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    return x

# ---------------------------
# 1) Splits
# ---------------------------
# Stratified split into train/valid/test. valid_size είναι ποσοστό του (train+valid) chunk, όχι του συνολικού.
def stratified_train_valid_test_split(
    X, y,
    test_size=0.20, 
    valid_size=0.20, 
    random_state=RANDOM_STATE
):
    """
    Επιστρέφει: X_train, X_valid, X_test, y_train, y_valid, y_test
    valid_size είναι ποσοστό του (train+valid) chunk, όχι του συνολικού.
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_temp, y_temp, test_size=valid_size, stratify=y_temp, random_state=random_state
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test


# ---------------------------
# 2) Metrics + Thresholding
# ---------------------------
# Υπολογίζει όλα τα βασικά metrics από τα probabilities και ένα threshold για binarization.
def compute_metrics_from_probs(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    # Για τις καμπύλες και AUC, χρησιμοποιούμε τα probabilities χωρίς thresholding
    pr_auc = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    # Για τα precision/recall/f1, χρησιμοποιούμε τα binary predictions με το threshold που επιλέξαμε
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)
    # confusion matrix για να πάρουμε tn, fp, fn, tp
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "threshold": float(threshold),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "f2": float(f2),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

# Επιλογή threshold που μεγιστοποιεί Fβ στο validation set
def pick_threshold_max_fbeta(y_true, y_prob, beta=2.0) -> Tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # precision/recall έχουν length = len(thresholds)+1
    # υπολογίζουμε Fβ σε όλα τα σημεία (εκτός του τελευταίου χωρίς threshold)
    fbeta = (1 + beta**2) * (precision[:-1] * recall[:-1]) / np.maximum(
        (beta**2 * precision[:-1] + recall[:-1]), 1e-12
    )
    best_idx = int(np.argmax(fbeta))
    return float(thresholds[best_idx]), float(fbeta[best_idx])
    #(“αν θέλω να πιάνω όσο περισσότερες απάτες γίνεται, ποιο threshold είναι σωστό;”)

# Επιλογή threshold που ικανοποιεί constraint στην precision (π.χ. precision >= 0.90) και από αυτά παίρνουμε το max recall.
def pick_threshold_for_precision_constraint(
    y_true, y_prob, target_precision=0.90
) -> Tuple[Optional[float], Dict[str, Any]]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    # thresholds aligns with precision[:-1], recall[:-1]
    mask = precision[:-1] >= target_precision
    if not np.any(mask):
        return None, {"note": "No threshold reaches target precision."}

    # από αυτά που πιάνουν precision, παίρνουμε το max recall
    candidate_idx = np.where(mask)[0]
    best_idx = candidate_idx[np.argmax(recall[:-1][candidate_idx])]
    thr = float(thresholds[best_idx])

    info = {
        "target_precision": float(target_precision),
        "achieved_precision": float(precision[:-1][best_idx]),
        "achieved_recall": float(recall[:-1][best_idx]),
    }
    return thr, info
    # (“Δέχομαι alerts μόνο αν είμαι ≥90% σίγουρος ότι είναι fraud”)

# ---------------------------
# 3) Plots
# ---------------------------
def plot_pr_curve(y_true, y_prob, title: str, outpath: Optional[str] = None):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title} | AP(PR-AUC)={ap:.4f}")
    plt.grid(True, alpha=0.3)
    if outpath:
        plt.tight_layout()
        plt.savefig(outpath, dpi=160)
    plt.show()
    # Για imbalance, αυτό είναι το πιο τίμιο γράφημα.

# ---------------------------
# 4) Results logging helper
# ---------------------------
def append_result(rows: List[Dict[str, Any]], model_name: str, stage: str, params: Dict[str, Any], metrics: Dict[str, Any], runtime_s: float):
    rows.append({
        "model": model_name,
        "stage": stage,  # e.g. "cv", "valid", "test"
        "runtime_s": float(runtime_s),
        "params_json": json.dumps(_json_safe(params), sort_keys=True),
        **metrics
    })

# ---------------------------
# 5) Random Forest tuning
# ---------------------------
def tune_random_forest(
    X_train, y_train,
    n_iter=30,
    cv_splits=5,
    scoring="average_precision",
    n_jobs=-1,
    random_state=RANDOM_STATE
):
    rf = RandomForestClassifier(
        random_state=random_state,
        n_jobs=n_jobs
    )

    param_distributions = {
    "n_estimators": [200, 400, 600],          
    "max_depth": [None, 8, 12, 16],
    "min_samples_split": [2, 10, 20],
    "min_samples_leaf": [1, 5, 10],
    "max_features": ["sqrt", "log2"],
    "class_weight": ["balanced"]
}

    
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        verbose=2,
        random_state=random_state,
        n_jobs=n_jobs,
        return_train_score=False
    )

    t0 = time.perf_counter()
    search.fit(X_train, y_train)
    runtime = time.perf_counter() - t0

    results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    return search, results, runtime


# ---------------------------
# 6) XGBoost tuning (Option A: sklearn RandomizedSearchCV)
# ---------------------------
def tune_xgb_sklearn_random_search(
    X_train, y_train,
    n_iter=30,
    cv_splits=5,
    scoring="average_precision",
    n_jobs=-1,
    random_state=RANDOM_STATE
):
    if XGBClassifier is None:
        raise RuntimeError("xgboost is not installed / import failed.")

    # base model
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",  # PR-AUC within XGB training
        tree_method="hist",
        random_state=random_state,
        n_jobs=n_jobs
    )

    # class imbalance hint: scale_pos_weight ~ (#neg/#pos) computed on train
    pos = np.sum(y_train == 1)
    neg = np.sum(y_train == 0)
    spw = float(neg / max(pos, 1))

    param_distributions = {
        "n_estimators": np.arange(200, 2001, 100),
        "learning_rate": np.round(np.logspace(np.log10(0.01), np.log10(0.3), 12), 4),
        "max_depth": np.arange(2, 11),
        "min_child_weight": np.arange(1, 11),
        "subsample": np.round(np.linspace(0.5, 1.0, 6), 2),
        "colsample_bytree": np.round(np.linspace(0.5, 1.0, 6), 2),
        "gamma": np.round(np.linspace(0.0, 5.0, 11), 2),
        "reg_alpha": np.round(np.logspace(-4, 1, 8), 6),
        "reg_lambda": np.round(np.logspace(-3, 2, 8), 6),
        "scale_pos_weight": [spw, spw * 0.5, spw * 1.5],
    }

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,  # uses sklearn's average_precision_score
        cv=cv,
        verbose=1,
        random_state=random_state,
        n_jobs=n_jobs,
        return_train_score=False
    )

    t0 = time.perf_counter()
    search.fit(X_train, y_train)
    runtime = time.perf_counter() - t0

    results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    return search, results, runtime


# ---------------------------
# 7) Evaluate a fitted model (probabilities) (helper)
# ---------------------------
def predict_proba_pos(model, X) -> np.ndarray:
    # sklearn style
    prob = model.predict_proba(X)[:, 1]
    return prob
    # απομονώνει το “παίρνω probability για class=1” Aν αύριο αλλάξεις model API, αλλάζεις εδώ

# ---------------------------
# 8) End-to-end: retrain best/top configs and evaluate
# ---------------------------
""" Για κάθε top-k config:
1.Train μόνο στο TRAIN
2.Παίρνεις probabilities στο VALID
3.Διαλέγεις threshold στο VALID
4.Κρατάς το threshold παγωμένο
5.Αξιολογείς στο TEST"""
def top_k_params(cv_results_df: pd.DataFrame, k=3) -> List[Dict[str, Any]]:
    # sklearn cv_results_ has 'params' column
    return list(cv_results_df.sort_values("rank_test_score")["params"].head(k))

# Για κάθε config: fit στο train, pick threshold στο valid, evaluate στο test με αυτό το threshold. Log results σε rows list.
def refit_and_evaluate(
    model_ctor,  # lambda **params: model
    params_list: List[Dict[str, Any]],
    X_train, y_train,
    X_valid, y_valid,
    X_test, y_test,
    model_name: str,
    precision_constraint: Optional[float] = 0.90
):
    rows = []

    for i, params in enumerate(params_list, start=1):
        model = model_ctor(**params)

        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0

        # VALID probs
        valid_prob = predict_proba_pos(model, X_valid)

        # Threshold selection on VALID
        thr_f2, _ = pick_threshold_max_fbeta(y_valid, valid_prob, beta=2.0)
        valid_metrics_f2 = compute_metrics_from_probs(y_valid, valid_prob, threshold=thr_f2)
        append_result(rows, model_name, f"valid_top{i}_thr=maxF2", params, valid_metrics_f2, fit_time)

        if precision_constraint is not None:
            thr_p, info = pick_threshold_for_precision_constraint(y_valid, valid_prob, target_precision=precision_constraint)
            if thr_p is not None:
                valid_metrics_p = compute_metrics_from_probs(y_valid, valid_prob, threshold=thr_p)
                # attach constraint info inside params for traceability
                params2 = dict(params)
                params2["_precision_constraint_info"] = info
                append_result(rows, model_name, f"valid_top{i}_thr=prec>={precision_constraint}", params2, valid_metrics_p, fit_time)

        # TEST evaluation with the SAME threshold(s) picked on valid
        test_prob = predict_proba_pos(model, X_test)

        test_metrics_f2 = compute_metrics_from_probs(y_test, test_prob, threshold=thr_f2)
        append_result(rows, model_name, f"test_top{i}_thr=validMaxF2", params, test_metrics_f2, fit_time)

        if precision_constraint is not None and thr_p is not None:
            test_metrics_p = compute_metrics_from_probs(y_test, test_prob, threshold=thr_p)
            params2 = dict(params)
            params2["_precision_constraint_info"] = info
            append_result(rows, model_name, f"test_top{i}_thr=validPrecConstraint", params2, test_metrics_p, fit_time)

    return pd.DataFrame(rows)


# ---------------------------
# Example MAIN (replace with your own data loading)
# ---------------------------
if __name__ == "__main__":
    # TODO: Replace with your data loading
    # Example assumes you already have X (DataFrame/ndarray) and y (Series/ndarray)
    raise SystemExit("Import this module and call functions from your Week 13 notebook/script.")
