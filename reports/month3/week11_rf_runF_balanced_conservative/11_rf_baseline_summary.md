# Week 11 — Random Forest (Business-Oriented)

**Setup:** n_estimators=300, max_depth=12, min_samples_split=2,
min_samples_leaf=3, max_features=sqrt, bootstrap=True,
class_weight=balanced, random_state=42,
threshold=0.5, optimize=cost, n_thresholds=101, cost_fp=1.0, cost_fn=20.0

## Μετρικές (Test set — default threshold 0.50)
- Accuracy: **0.999403**
- Precision: **0.8404**
- Recall: **0.8061**
- F1: **0.8229**
- ROC-AUC: **0.9612**
- PR-AUC (AP): **0.8455**
- Recall@Precision≥0.90: **0.6837**
- Precision@Recall≥0.90: **0.1601**

## Best threshold (cost) = **0.1864**

## Artifacts
- Model: `models/rf_runF_balanced.joblib`
- Metrics: `reports/week11_rf_runF_balanced_conservative/metrics.json`
- Report: `reports/week11_rf_runF_balanced_conservative/classification_report.txt`
- Importances: `reports/week11_rf_runF_balanced_conservative/feature_importances.csv`
- Sweep: `reports/week11_rf_runF_balanced_conservative/threshold_sweep.csv`

## Figures (folder)
- `reports/figures/week11/runF`

> Note: In production, threshold tuning should be done on validation/CV; test set remains locked.
