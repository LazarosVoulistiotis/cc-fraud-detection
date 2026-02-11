import pandas as pd

# RF
rf = pd.read_csv("reports/week13/rf_cv_results_week13.csv")
xgb = pd.read_csv("reports/week13/xgb_cv_results_week13.csv")

print("Best RF PR-AUC (CV):", rf["mean_test_score"].max())
print("Best XGB PR-AUC (CV):", xgb["mean_test_score"].max())
