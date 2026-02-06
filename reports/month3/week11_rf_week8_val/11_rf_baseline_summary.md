# Week 11 — Random Forest (Business-Oriented)

**Setup:** n_estimators=300, max_depth=12, min_samples_split=2,
min_samples_leaf=3, max_features=sqrt, bootstrap=True,
class_weight=balanced, random_state=42,
threshold=0.5, optimize=cost, n_thresholds=101, cost_fp=1.0, cost_fn=20.0

## Μετρικές (Test set — default threshold 0.50)
- Accuracy: **0.999542**
- Precision: **0.8542**
- Recall: **0.8723**
- F1: **0.8632**
- ROC-AUC: **0.9912**
- PR-AUC (AP): **0.8715**
- Recall@Precision≥0.90: **0.8298**
- Precision@Recall≥0.90: **0.0546**

## Best threshold (cost) = **0.2354**

## Artifacts
- Model: `models/rf_week8.joblib`
- Metrics: `reports/week11_rf_week8_val/metrics.json`
- Report: `reports/week11_rf_week8_val/classification_report.txt`
- Importances: `reports/week11_rf_week8_val/feature_importances.csv`
- Sweep: `reports/week11_rf_week8_val/threshold_sweep.csv`

## Figures (folder)
- `reports/figures/week11_rf_week8_val`

> Note: In production, threshold tuning should be done on validation/CV; test set remains locked.
