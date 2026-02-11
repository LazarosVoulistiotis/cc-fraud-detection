import pandas as pd
import matplotlib
matplotlib.use("Agg")

from pathlib import Path
from eval_harness import stratified_train_valid_test_split, plot_pr_curve
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# -------------------------
# Load data & splits
# -------------------------
DATA_PATH = Path("data/data_raw/creditcard.csv")
df = pd.read_csv(DATA_PATH)
FIG_DIR = Path("reports/figures/week13")
FIG_DIR.mkdir(parents=True, exist_ok=True)

X = df.drop(columns=["Class"])
y = df["Class"].values

X_train, X_valid, X_test, y_train, y_valid, y_test = \
    stratified_train_valid_test_split(X, y)

# -------------------------
# Recreate best RF
# -------------------------
best_rf = RandomForestClassifier(
    n_estimators=200,
    min_samples_split=10,
    min_samples_leaf=1,
    max_features="sqrt",
    max_depth=12,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

best_rf.fit(X_train, y_train)
rf_test_prob = best_rf.predict_proba(X_test)[:, 1]

plot_pr_curve(
    y_test,
    rf_test_prob,
    title="RandomForest (tuned) - TEST PR curve",
    outpath=FIG_DIR / "pr_rf_test.png"
)

# -------------------------
# Recreate best XGB
# -------------------------
best_xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
    subsample=0.5,
    scale_pos_weight=866.4809523809524,
    reg_lambda=0.719686,
    reg_alpha=0.071969,
    n_estimators=900,
    min_child_weight=5,
    max_depth=3,
    learning_rate=0.0871,
    gamma=3.0,
    colsample_bytree=0.6
)

best_xgb.fit(X_train, y_train)
xgb_test_prob = best_xgb.predict_proba(X_test)[:, 1]

plot_pr_curve(
    y_test,
    xgb_test_prob,
    title="XGBoost (tuned) - TEST PR curve",
    outpath=FIG_DIR / "pr_xgb_test.png"
)
