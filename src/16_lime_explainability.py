# src/16_lime_explainability.py
"""
python src/16_lime_explainability.py \
 --model-path models/xgb_week8.joblib \
 --data-train data/data_interim/splits_week8/train.csv \
 --data-test  data/data_interim/splits_week8/test.csv \
 --target-column Class \
 --shap-cases reports/month4/week15_shap/shap_cases.json \
 --figdir reports/figures/week16 \
 --outdir reports/week16_lime \
 --threshold 0.0884 \
 --num-features 10 \
 --num-samples 5000 \
 --seed 42
 """
# This script performs LIME local explainability on selected test instances.
import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import List, Dict, Any, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless-safe (όπως Week 15 fix)
import matplotlib.pyplot as plt

from lime.lime_tabular import LimeTabularExplainer

# We keep it simple and explicit with a dataclass for config. Αντί να κουβαλάς args.model_path, args.data_train κλπ παντού, τα βάζεις σε ένα object (cfg)
@dataclass
class LimeRunConfig:
    model_path: str
    data_train: str
    data_test: str
    target_column: str
    shap_cases_path: Optional[str]
    indices: Optional[List[int]]
    figdir: str
    outdir: str
    threshold: float
    num_features: int
    num_samples: int
    seed: int

# We parse command-line arguments into a structured config object. This keeps our main() cleaner and more maintainable.
def parse_args() -> LimeRunConfig:
    p = argparse.ArgumentParser(description="Week 16 — LIME local explainability (tabular, model-agnostic).")

    p.add_argument("--model-path", required=True, help="Path to trained model (.joblib), e.g. models/xgb_week8.joblib")
    p.add_argument("--data-train", required=True, help="Train CSV (engineered schema), e.g. data/.../splits_week8/train.csv")
    p.add_argument("--data-test", required=True, help="Test CSV (engineered schema), e.g. data/.../splits_week8/test.csv")
    p.add_argument("--target-column", default="Class", help="Target column name (default: Class)")

    p.add_argument("--shap-cases", default=None, help="Path to shap_cases.json (Week 15) to reuse indices")
    p.add_argument("--indices", nargs="*", type=int, default=None,
                   help="Optional explicit test indices to explain. If provided, overrides --shap-cases.")

    p.add_argument("--figdir", required=True, help="Directory to save LIME figures (PNG)")
    p.add_argument("--outdir", required=True, help="Directory to save JSON/CSV outputs")

    p.add_argument("--threshold", type=float, default=0.0884, help="Operating threshold for narrative (default: 0.0884)")
    p.add_argument("--num-features", type=int, default=10, help="How many features to show in LIME explanation")
    p.add_argument("--num-samples", type=int, default=5000, help="LIME perturbation samples per instance")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = p.parse_args()

    return LimeRunConfig(
        model_path=args.model_path,
        data_train=args.data_train,
        data_test=args.data_test,
        target_column=args.target_column,
        shap_cases_path=args.shap_cases,
        indices=args.indices,
        figdir=args.figdir,
        outdir=args.outdir,
        threshold=args.threshold,
        num_features=args.num_features,
        num_samples=args.num_samples,
        seed=args.seed,
    )

# Utility functions
def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)

# This function robustly loads indices from a shap_cases.json file, handling both the expected structure and potential variations. It also returns any metadata found in the file for reference.
def load_cases_from_shap(shap_cases_path: str) -> Tuple[List[int], Dict[str, Any]]:
    with open(shap_cases_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Most common structure in your project:
    # {"threshold": ..., "cases": [ {"name":..., "index":..., ...}, ... ] }
    cases = data.get("cases", data)  # fallback if stored flat
    indices: List[int] = []
    meta: Dict[str, Any] = {}

    # Case A: cases is a dict of dicts (your original expectation)
    if isinstance(cases, dict):
        for _, v in cases.items():
            if isinstance(v, dict):
                # accept a few common index key names
                for key in ("index", "test_index", "idx"):
                    if key in v:
                        indices.append(int(v[key]))
                        break
        meta = data

    # Case B: cases is a list of dicts (your actual file)
    elif isinstance(cases, list):
        for item in cases:
            if isinstance(item, dict):
                for key in ("index", "test_index", "idx"):
                    if key in item:
                        indices.append(int(item[key]))
                        break
        meta = data

    # Case C: flat dict with indices directly (rare but possible)
    elif isinstance(data, dict):
        # try to pull any int-like values under keys that look like indices
        for k, v in data.items():
            if "index" in str(k).lower():
                try:
                    indices.append(int(v))
                except Exception:
                    pass
        meta = data

    else:
        raise ValueError("Unexpected shap_cases.json format.")

    if not indices:
        raise ValueError("No indices extracted from shap_cases.json. Check file structure/keys.")

    # keep stable order but remove duplicates
    seen = set()
    indices_unique = []
    for i in indices:
        if i not in seen:
            seen.add(i)
            indices_unique.append(i)

    return indices_unique, meta

# This function identifies feature columns by excluding the target column. It also checks that at least one feature column remains, otherwise it raises an error. This ensures that we have valid input for LIME.
def get_feature_columns(df: pd.DataFrame, target: str) -> List[str]:
    cols = [c for c in df.columns if c != target]
    if len(cols) == 0:
        raise ValueError("No feature columns found after dropping target.")
    return cols

# This function ensures that both train and test DataFrames have the same set of feature columns in the same order. It checks for missing columns and raises errors if any are found, preventing silent mismatches that could lead to incorrect explanations.
def align_features(train_df: pd.DataFrame, test_df: pd.DataFrame, features: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Ensure both have all columns
    missing_in_test = [c for c in features if c not in test_df.columns]
    missing_in_train = [c for c in features if c not in train_df.columns]
    if missing_in_test:
        raise ValueError(f"Test is missing required feature columns: {missing_in_test}")
    if missing_in_train:
        raise ValueError(f"Train is missing required feature columns: {missing_in_train}")

    return train_df[features], test_df[features]

# This function wraps model.predict_proba to ensure it always returns a 2D array with shape [n, 2], which is what LIME expects for binary classification. It also includes error handling to catch unexpected output shapes, making the code more robust against model issues.
def safe_predict_proba(model, X: np.ndarray) -> np.ndarray:
    """
    LIME expects a function f(X) -> proba for both classes.
    We keep it robust and ensure shape [n, 2].
    """
    proba = model.predict_proba(X)
    proba = np.asarray(proba)

    if proba.ndim != 2 or proba.shape[1] != 2:
        raise ValueError(f"predict_proba returned shape {proba.shape}, expected [n,2].")
    return proba

# Main function``
def main() -> None:
    cfg = parse_args()
    ensure_dirs(cfg.figdir, cfg.outdir)

    # Save config for reproducibility
    with open(os.path.join(cfg.outdir, "lime_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    # Load model + data
    model = joblib.load(cfg.model_path)
    train_df = pd.read_csv(cfg.data_train)
    test_df = pd.read_csv(cfg.data_test)

    if cfg.target_column not in train_df.columns or cfg.target_column not in test_df.columns:
        raise ValueError(f"Target column '{cfg.target_column}' not found in train/test CSVs.")

    # Decide indices
    indices = cfg.indices
    shap_meta = None
    if indices is None:
        if not cfg.shap_cases_path:
            raise ValueError("Provide either --indices or --shap-cases.")
        indices, shap_meta = load_cases_from_shap(cfg.shap_cases_path)

    # Features
    features = get_feature_columns(train_df, cfg.target_column)
    X_train_df, X_test_df = align_features(train_df, test_df, features)
    y_test = test_df[cfg.target_column].astype(int).values

    # LIME explainer
    np.random.seed(cfg.seed)

    explainer = LimeTabularExplainer(
        training_data=X_train_df.to_numpy(), # η LIME θέλει reference distribution για να ξέρει “τι είναι φυσιολογικές τιμές” όταν κάνει perturbation
        feature_names=features,
        class_names=["legit", "fraud"],
        mode="classification",
        discretize_continuous=True, # αυτό βοηθάει στο να έχουμε πιο human-friendly explanations, ειδικά αν τα features είναι numeric. Αν τα αφήσουμε συνεχή, μπορεί να έχουμε rules τύπου "feature_1 <= 0.12345" που δεν είναι τόσο κατανοητά.
        random_state=cfg.seed, # για reproducibility των perturbations. Δεν επηρεάζει το μοντέλο, αλλά κάνει τα explanations σταθερά μεταξύ runs.``
    )

    results: List[Dict[str, Any]] = []
    # We loop over the selected indices, generate LIME explanations, and save both figures and structured results. We also include error handling for out-of-bounds indices and ensure that the predicted probabilities are correctly interpreted according to the specified threshold.``
    for idx in indices:
        if idx < 0 or idx >= len(test_df):
            raise IndexError(f"Index {idx} out of bounds for test set size {len(test_df)}")

        x_row = X_test_df.iloc[idx].to_numpy() # LIME expects a 1D array for a single instance
        x_row_2d = x_row.reshape(1, -1) # reshape to 2D for predict_proba, since it expects [n_samples, n_features]

        proba = safe_predict_proba(model, x_row_2d)[0] # proba[1] = P(fraud) και proba[0] = P(legit)
        pred_label = int(proba[1] >= cfg.threshold) # pred_label γίνεται 1 αν περνάει threshold, αλλιώς 0
        true_label = int(y_test[idx]) 

        # LIME explanation (η καρδιά)
        exp = explainer.explain_instance(
            data_row=x_row,
            predict_fn=lambda X: safe_predict_proba(model, X),
            num_features=cfg.num_features,
            num_samples=cfg.num_samples,
        )
        """
    Τι κάνει conceptually η LIME εδώ:
     `1.Παίρνει το instance x_row.
     `2.Δημιουργεί num_samples “ψεύτικα” κοντινά δείγματα (perturbations) γύρω του.
     `3.Ρωτάει το μοντέλο σου (μέσω predict_fn) τι πιθανότητα fraud δίνει σε καθένα.
     `4.Fit-άρει ένα απλό τοπικό surrogate model (συνήθως sparse linear model) που προσεγγίζει τη συμπεριφορά του black-box τοπικά.
     `5.Επιστρέφει weights που λένε: “αυτά τα features σπρώχνουν προς fraud / προς legit” για αυτό το instance.
    num_samples: όσο μεγαλύτερο, τόσο πιο “σταθερό” αλλά πιο αργό."""

        # Plot
        fig = exp.as_pyplot_figure(label=1)  # label=1 => fraud class explanation
        fig.suptitle(f"LIME — idx={idx} | true={true_label} pred={pred_label} p_fraud={proba[1]:.6f} thr={cfg.threshold}")
        figpath = os.path.join(cfg.figdir, f"lime_idx{idx}.png")
        fig.savefig(figpath, dpi=200, bbox_inches="tight")
        plt.close(fig)

        # Export top features
        top_list = exp.as_list(label=1)  # list of (feature_rule, weight)
        # We also store a "clean" mapping feature->weight when possible.
        # Note: LIME returns "feature <= value" style strings; still useful for report.
        result = {
            "index": idx,
            "true_label": true_label,
            "pred_label": pred_label,
            "p_legit": float(proba[0]),
            "p_fraud": float(proba[1]),
            "threshold": cfg.threshold,
            "num_features": cfg.num_features,
            "num_samples": cfg.num_samples,
            "seed": cfg.seed,
            "lime_top": [{"term": t, "weight": float(w)} for t, w in top_list], # term: συνήθως string τύπου "V14 <= -2.31" ή "Amount_log1p > 0.5", weight: θετικό → σπρώχνει προς fraud (για label=1), αρνητικό → σπρώχνει προς legit (ανάλογα με το orientation του plot)
            "figure_path": figpath.replace("\\", "/"),
        }
        results.append(result)

    # Save JSON
    out_json = {
        "meta": { # γενικά info για το run, χρήσιμο για αναφορά και reproducibility
            "model_path": cfg.model_path,
            "data_train": cfg.data_train,
            "data_test": cfg.data_test,
            "target_column": cfg.target_column,
            "threshold": cfg.threshold,
            "class_names": ["legit", "fraud"],
        },
        "shap_cases_meta": shap_meta, # shap_cases_meta: κρατάς από πού ήρθαν τα indices
        "results": results, # results: για κάθε idx έχεις probs, labels, threshold, lime_top, figure_path
    }
    with open(os.path.join(cfg.outdir, "lime_explanations.json"), "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)

    # Save CSV summary (flat)
    rows = []
    for r in results:
        for rank, item in enumerate(r["lime_top"], start=1):
            rows.append({
                "index": r["index"],
                "true_label": r["true_label"],
                "pred_label": r["pred_label"],
                "p_fraud": r["p_fraud"],
                "threshold": r["threshold"],
                "rank": rank,
                "term": item["term"],
                "weight": item["weight"],
            })
    pd.DataFrame(rows).to_csv(os.path.join(cfg.outdir, "lime_top_features.csv"), index=False)

    print(f"[OK] Saved {len(results)} LIME explanations.")
    print(f"Figures: {cfg.figdir}")
    print(f"Outputs: {cfg.outdir}")


if __name__ == "__main__":
    main()