# Week 11 — Random Forest (Business-Oriented)

**Setup:** n_estimators=100, max_depth=None, min_samples_split=2,
min_samples_leaf=1, max_features=sqrt, bootstrap=True,
class_weight=none, random_state=42,
threshold=0.5, optimize=none, n_thresholds=101, cost_fp=1.0, cost_fn=20.0

## Μετρικές (Test set — default threshold 0.50)
- Accuracy: **0.999561**
- Precision: **0.9506**
- Recall: **0.7857**
- F1: **0.8603**
- ROC-AUC: **0.9577**
- PR-AUC (AP): **0.8686**
- Recall@Precision≥0.90: **0.8265**
- Precision@Recall≥0.90: **0.0580**

## Best threshold: (not requested)

## Artifacts
- Model: `models/rf_runA_baseline.joblib`
- Metrics: `reports/week11_rf_runA_baseline/metrics.json`
- Report: `reports/week11_rf_runA_baseline/classification_report.txt`
- Importances: `reports/week11_rf_runA_baseline/feature_importances.csv`
- Sweep: `reports/week11_rf_runA_baseline/threshold_sweep.csv`

## Figures (folder)
- `reports/figures/week11/runA`

> Note: In production, threshold tuning should be done on validation/CV; test set remains locked.
