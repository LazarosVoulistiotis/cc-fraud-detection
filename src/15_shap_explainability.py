# src/15_shap_explainability.py `
# python src/15_shap_explainability.py --help (για πλήρη περιγραφή παραμέτρων)
"""
-Run #1 (προτεινόμενο για report): Week12 cost-policy threshold = 0.0884
python src/15_shap_explainability.py \
  --model-path models/xgb_week8.joblib \
  --data-train data/data_interim/splits_week8/train.csv \
  --data-test  data/data_interim/splits_week8/test.csv \
  --target-column Class \
  --figdir reports/figures/week15 \
  --outdir reports/week15_shap \
  --sample-size 10000 \
  --background-size 1000 \
  --threshold 0.0884 \
  --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")          # <--- σημαντικό: headless backend
import matplotlib.pyplot as plt
import xgboost as xgb


# CLI parser
def parse_args():
    p = argparse.ArgumentParser(description="Week 15: SHAP explainability for best model")
    p.add_argument("--model-path", required=True, help="Path to trained model (.joblib). Can be estimator or sklearn Pipeline.")
    p.add_argument("--data-train", required=True, help="Train split CSV (for SHAP background).")
    p.add_argument("--data-test", required=True, help="Test split CSV (for explanations + selecting cases).")
    p.add_argument("--target-column", default="Class")
    p.add_argument("--figdir", required=True, help="Output folder for SHAP figures.")
    p.add_argument("--outdir", required=True, help="Output folder for SHAP metadata (csv/json).")

    p.add_argument("--sample-size", type=int, default=10000, help="Rows to explain globally (runtime control).")
    p.add_argument("--background-size", type=int, default=1000, help="Background rows for TreeExplainer.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=0.5, help="Operating threshold for TP/TN/borderline selection.")
    p.add_argument("--top-dependence", type=int, default=2, help="How many top features to generate dependence plots for.")

    return p.parse_args()


# Utility functions
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def split_xy(df: pd.DataFrame, target: str):
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not in dataframe columns.")
    X = df.drop(columns=[target])
    y = df[target].astype(int).values
    return X, y

def align_to_expected(X: pd.DataFrame, expected: list[str]) -> pd.DataFrame:
    X = X.copy()

    # drop extras
    extra = [c for c in X.columns if c not in expected]
    if extra:
        X = X.drop(columns=extra)

    # add missing
    missing = [c for c in expected if c not in X.columns]
    for c in missing:
        X[c] = 0.0

    # reorder
    return X[expected]


# Unwrap pipeline to get final estimator, transformed X for SHAP, feature names, and predict_proba function
def unwrap_pipeline(model, X: pd.DataFrame):
    """
    Returns (predictor, X_for_shap, feature_names, predict_proba_fn)

    - If model is sklearn Pipeline: shap uses final estimator with transformed X
    - Else: shap uses the model directly on X
    """
    # sklearn Pipeline has attribute "steps"
    if hasattr(model, "steps") and isinstance(model.steps, list) and len(model.steps) >= 2:
        pre = model[:-1]
        est = model[-1]

        X_t = pre.transform(X) # shap needs the transformed features, not the raw ones
        # feature names if available
        if hasattr(pre, "get_feature_names_out"):
            feat_names = list(pre.get_feature_names_out())
        else:
            feat_names = list(X.columns)

        X_t_df = pd.DataFrame(X_t, columns=feat_names) # shap needs a DataFrame with feature names for dependence plots

        def proba_fn(X_raw: pd.DataFrame) -> np.ndarray: # predict_proba needs to run through the full pipeline to be consistent with how SHAP sees the model
            return model.predict_proba(X_raw)[:, 1] # we return the probability of the positive class (index 1) for binary classification

        return est, X_t_df, feat_names, proba_fn # shap explainer should use the final estimator and the transformed features, but for predict_proba we need to run the full pipeline to be consistent with how SHAP sees the model

    # if not a pipeline
    feat_names = list(X.columns)

    def proba_fn(X_raw: pd.DataFrame) -> np.ndarray:
        # Robust prediction for XGBoost: preserve feature names explicitly
        if hasattr(model, "get_booster"):
            booster = model.get_booster()
            expected = booster.feature_names or list(X_raw.columns)
            X_use = align_to_expected(X_raw, list(expected))
            dm = xgb.DMatrix(X_use, feature_names=list(expected))
            return booster.predict(dm)  # probabilities for binary:logistic

        return model.predict_proba(X_raw)[:, 1]

    return model, X.copy(), feat_names, proba_fn

# Select TP, TN, and borderline cases based on predicted probabilities and threshold
def pick_cases(y_true: np.ndarray, proba: np.ndarray, threshold: float):
    pred = (proba >= threshold).astype(int) # Μετατρέπει probabilities σε predictions με βάση το operating threshold.

    tp_idx = np.where((y_true == 1) & (pred == 1))[0]
    tn_idx = np.where((y_true == 0) & (pred == 0))[0]

    tp = int(tp_idx[np.argmax(proba[tp_idx])]) if len(tp_idx) else None
    tn = int(tn_idx[np.argmin(proba[tn_idx])]) if len(tn_idx) else None

    # Borderline = closest probability to threshold (either side)
    border = int(np.argmin(np.abs(proba - threshold)))

    return tp, tn, border

# Ασφαλές saving
def save_fig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

# Main function
def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    figdir = Path(args.figdir)
    outdir = Path(args.outdir)
    ensure_dir(figdir)
    ensure_dir(outdir)

    # Load
    model = joblib.load(args.model_path)

    df_train = pd.read_csv(args.data_train)
    df_test = pd.read_csv(args.data_test)

    X_train_raw, _ = split_xy(df_train, args.target_column)
    X_test_raw, y_test = split_xy(df_test, args.target_column)

    # Align raw features to what XGBoost expects (prevents feature-name mismatch errors)
    is_pipeline = hasattr(model, "steps") and isinstance(model.steps, list) and len(model.steps) >= 2

    if (not is_pipeline) and hasattr(model, "get_booster"):
        expected = model.get_booster().feature_names
        if expected:
            X_train_raw = align_to_expected(X_train_raw, list(expected))
            X_test_raw  = align_to_expected(X_test_raw,  list(expected))

    # Unwrap (keeps preprocessing consistent)
    est, X_train, feat_names, proba_fn = unwrap_pipeline(model, X_train_raw)
    _, X_test, _, _ = unwrap_pipeline(model, X_test_raw)

    # Sampling για global SHAP
    n_test = len(X_test)
    sample_n = min(args.sample_size, n_test)
    sample_idx = rng.choice(n_test, size=sample_n, replace=False)
    X_sample = X_test.iloc[sample_idx].reset_index(drop=True)
    # Background sampling
    bg_n = min(args.background_size, len(X_train))
    bg_idx = rng.choice(len(X_train), size=bg_n, replace=False)
    X_bg = X_train.iloc[bg_idx].reset_index(drop=True)

    # Explainer (fast for tree models)
    explainer = shap.TreeExplainer(est, data=X_bg)

    # ---- GLOBAL SHAP
    shap_values = explainer(X_sample)  # returns shap.Explanation

    # Mean abs shap for ranking
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    imp = pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(outdir / "shap_mean_abs.csv", index=False)

    # Beeswarm = δείχνει distribution SHAP + directionality (high/low values).
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False, max_display=15) # show=False για να μη προσπαθήσει να ανοίξει GUI.
    save_fig(figdir / "shap_summary_beeswarm.png")

    # Bar plot
    plt.figure()
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=15)
    save_fig(figdir / "shap_mean_abs_bar.png")

    # Dependence plots for top-N
    top_feats = imp["feature"].head(args.top_dependence).tolist()
    for f in top_feats:
        plt.figure()
        shap.dependence_plot(f, shap_values.values, X_sample, show=False)
        save_fig(figdir / f"shap_dependence_{f}.png")

    # ---- LOCAL SHAP (TP / TN / Borderline on TEST, based on operating threshold)
    proba_test = proba_fn(X_test_raw)  # use full model for proba selection (pipeline-safe)
    tp_i, tn_i, border_i = pick_cases(y_test, proba_test, args.threshold)

    cases = []
    for name, idx in [("true_positive", tp_i), ("true_negative", tn_i), ("borderline", border_i)]:
        if idx is None:
            continue
        cases.append((name, idx))

    meta = {"threshold": args.threshold, "cases": []}
    # Για κάθε case, εξηγούμε με SHAP και αποθηκεύουμε waterfall plot + metadata. Waterfall plot δείχνει τα top features που οδήγησαν στην πρόβλεψη για αυτό το instance.`
    for name, idx in cases:
        x_row = X_test.iloc[[idx]].reset_index(drop=True) # Παίρνεις μία γραμμή (σαν mini-DataFrame)
        sv = explainer(x_row) # Υπολογίζεις SHAP γι’ αυτήν
        # Ground truth και predicted probability`
        true_label = int(y_test[idx])
        p = float(proba_test[idx])
        # Waterfall plot: δείχνει πώς από base_value πηγαίνεις στο f(x) με contributions
        plt.figure()
        shap.plots.waterfall(sv[0], show=False, max_display=10)
        plt.title(f"{name.replace('_',' ').title()}\nTrue label: {true_label}, Predicted prob: {p:.4f}")
        save_fig(figdir / f"shap_waterfall_{name}.png")
        # metadata
        meta["cases"].append({"name": name, "index": int(idx), "true_label": true_label, "predicted_prob": p})
    # Save JSON
    with open(outdir / "shap_cases.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] Saved figures to: {figdir}")
    print(f"[OK] Saved metadata to: {outdir}")

# Script entrypoint
if __name__ == "__main__":
    main()
