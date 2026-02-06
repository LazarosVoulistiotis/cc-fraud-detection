# Week 11 — Random Forest (Business-Oriented)

**Setup:** n_estimators=300, max_depth=12, min_samples_split=2,
min_samples_leaf=3, max_features=sqrt, bootstrap=True,
class_weight=balanced, random_state=42,
threshold=0.0198, optimize=none, n_thresholds=101, cost_fp=1.0, cost_fn=20.0

## Μετρικές (Test set — default threshold 0.02)
- Accuracy: **0.968139**
- Precision: **0.0437**
- Recall: **0.8632**
- F1: **0.0832**
- ROC-AUC: **0.9719**
- PR-AUC (AP): **0.8061**
- Recall@Precision≥0.90: **0.7368**
- Precision@Recall≥0.90: **0.0154**

## Best threshold: (not requested)

## Artifacts
- Model: `models/rf_week8.joblib`
- Metrics: `reports/week11_rf_week8_test_recall90/metrics.json`
- Report: `reports/week11_rf_week8_test_recall90/classification_report.txt`
- Importances: `reports/week11_rf_week8_test_recall90/feature_importances.csv`
- Sweep: `reports/week11_rf_week8_test_recall90/threshold_sweep.csv`

## Figures (folder)
- `reports/figures/week11_rf_week8_test_recall90`

> Note: In production, threshold tuning should be done on validation/CV; test set remains locked.
