"""η ροή είναι:
1.import splits (ή ξανατρέχεις το split code)
2.τρέχεις tune_random_forest
3.βλέπεις CV αποτελέσματα
4.διαλέγεις top-3
5.refit_and_evaluate"""

import pandas as pd
from pathlib import Path

from eval_harness import (
    stratified_train_valid_test_split,
    tune_random_forest,
    top_k_params,
    refit_and_evaluate
)

from sklearn.ensemble import RandomForestClassifier

# -------------------
# Load data
# -------------------
DATA_PATH = Path("data/data_raw/creditcard.csv")
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Class"])
y = df["Class"].values

X_train, X_valid, X_test, y_train, y_valid, y_test = \
    stratified_train_valid_test_split(X, y)

# -------------------
# RF tuning
# -------------------
search, rf_cv_results, rf_runtime = tune_random_forest(
    X_train, y_train,
    n_iter=10,
    cv_splits=3,
    scoring="average_precision"
)

print("Best RF PR-AUC (CV):", search.best_score_)
print("Best RF params:", search.best_params_)

# -------------------------
# Save CV results (Week 13)
# -------------------------
OUTPUT_DIR = Path("reports/week13")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rf_cv_results.to_csv(
    OUTPUT_DIR / "rf_cv_results_week13.csv",
    index=False
)

# Print summary table for report
print("\nRF CV Summary:")
print(
    rf_cv_results[["mean_test_score", "std_test_score", "params"]]
    .head()
)

# -------------------
# Top-3 configs
# -------------------
rf_top3 = top_k_params(rf_cv_results, k=3)

def rf_ctor(**params):
    return RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        **params
    )

rf_eval = refit_and_evaluate(
    rf_ctor,
    rf_top3,
    X_train, y_train,
    X_valid, y_valid,
    X_test, y_test,
    model_name="RandomForest",
    precision_constraint=0.90
)

print(rf_eval.sort_values(["stage", "pr_auc"], ascending=[True, False]))