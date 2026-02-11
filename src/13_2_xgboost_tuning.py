# 13_2_xgboost_tuning.py
"""Χρησιμοποιούμε CV στο train set για να βρούμε καλά hyperparameters.
1.Metric: PR-AUC (average_precision) — ίδιο με RF.
2.Παίρνουμε top-3 configs.
3.Κάνουμε refit.
4.Διαλέγουμε threshold στο validation.
5.Αξιολογούμε στο test.
Ίδιο evaluation protocol → καθαρή σύγκριση με RF."""

import pandas as pd
from pathlib import Path

from eval_harness import (
    stratified_train_valid_test_split,
    tune_xgb_sklearn_random_search,
    top_k_params,
    refit_and_evaluate
)

from xgboost import XGBClassifier


# -----------------------------
# 1) Load data
# -----------------------------
DATA_PATH = Path("data/data_raw/creditcard.csv")
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Class"])
y = df["Class"].values

X_train, X_valid, X_test, y_train, y_valid, y_test = \
    stratified_train_valid_test_split(X, y)

print("Fraud rates:",
      y_train.mean(),
      y_valid.mean(),
      y_test.mean())


# -----------------------------
# 2) XGBoost RandomizedSearchCV
# -----------------------------
xgb_search, xgb_cv_results, xgb_runtime = tune_xgb_sklearn_random_search(
    X_train,
    y_train,
    n_iter=15,          # laptop-friendly
    cv_splits=3,        # same as RF
    scoring="average_precision"
)

print("\n===== CV RESULTS =====")
print("Best XGB PR-AUC (CV):", xgb_search.best_score_)
print("Best XGB params:", xgb_search.best_params_)
print("Total CV runtime (sec):", round(xgb_runtime, 2))

# -------------------------
# Save CV results (Week 13)
# -------------------------
OUTPUT_DIR = Path("reports/week13")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

xgb_cv_results.to_csv(
    OUTPUT_DIR / "xgb_cv_results_week13.csv",
    index=False
)

# Print summary table for report
print("\nXGB CV Summary:")
print(
    xgb_cv_results[["mean_test_score", "std_test_score", "params"]]
    .head()
)

# -----------------------------
# 3) Take top-3 configs
# -----------------------------
xgb_top3 = top_k_params(xgb_cv_results, k=3)

print("\nTop-3 parameter sets:")
for i, params in enumerate(xgb_top3, 1):
    print(f"\nTop-{i} params:")
    print(params)


# -----------------------------
# 4) Refit + threshold selection
# -----------------------------
def xgb_ctor(**params):
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        **params
    )


xgb_eval = refit_and_evaluate(
    xgb_ctor,
    xgb_top3,
    X_train, y_train,
    X_valid, y_valid,
    X_test, y_test,
    model_name="XGBoost",
    precision_constraint=0.90
)


# -----------------------------
# 5) Final evaluation table
# -----------------------------
print("\n===== FINAL EVALUATION (sorted) =====")
print(
    xgb_eval
    .sort_values(["stage", "pr_auc"], ascending=[True, False])
    .reset_index(drop=True)
)
