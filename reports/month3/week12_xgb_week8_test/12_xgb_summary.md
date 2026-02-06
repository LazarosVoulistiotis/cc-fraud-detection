# Week 12 — XGBoost (Business-Oriented)

**Setup:** n_estimators=400, max_depth=6, learning_rate=0.05,
subsample=0.8, colsample_bytree=0.8,
scale_pos_weight=auto (resolved=599.0211480362537),
threshold=0.0884, optimize=none, n_thresholds=101,
cost_fp=1.0, cost_fn=20.0

## Metrics (Eval set — default threshold 0.09)
- Accuracy: **0.999330**
- Precision (Fraud): **0.7938**
- Recall (Fraud): **0.8105**
- F1: **0.8021**
- ROC-AUC: **0.9699**
- PR-AUC (AP): **0.8171**
- Recall@Precision≥0.90: **0.7895**
- Precision@Recall≥0.90: **0.0118**

## Best threshold: (not requested)

## Artifacts
- Model: `models/xgb_week8.joblib`
- Metrics: `reports/week12_xgb_week8_test/metrics.json`
- Config: `reports/week12_xgb_week8_test/config.json`
- Report: `reports/week12_xgb_week8_test/classification_report.txt`
- Sweep: `reports/week12_xgb_week8_test/threshold_sweep.csv`

## Figures (folder)
- `reports/figures/week12_xgb_week8_test`

> Note: Threshold tuning should be done on validation/CV. The test set remains locked for the final evaluation.
