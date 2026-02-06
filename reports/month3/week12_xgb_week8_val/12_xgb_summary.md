# Week 12 — XGBoost (Business-Oriented)

**Setup:** n_estimators=400, max_depth=6, learning_rate=0.05,
subsample=0.8, colsample_bytree=0.8,
scale_pos_weight=auto (resolved=599.0211480362537),
threshold=0.5, optimize=cost, n_thresholds=101,
cost_fp=1.0, cost_fn=20.0

## Metrics (Eval set — default threshold 0.50)
- Accuracy: **0.999718**
- Precision (Fraud): **0.9535**
- Recall (Fraud): **0.8723**
- F1: **0.9111**
- ROC-AUC: **0.9957**
- PR-AUC (AP): **0.8966**
- Recall@Precision≥0.90: **0.8723**
- Precision@Recall≥0.90: **0.1982**

## Best threshold (cost) = **0.0884**

## Artifacts
- Model: `models/xgb_week8.joblib`
- Metrics: `reports/week12_xgb_week8_val/metrics.json`
- Config: `reports/week12_xgb_week8_val/config.json`
- Report: `reports/week12_xgb_week8_val/classification_report.txt`
- Sweep: `reports/week12_xgb_week8_val/threshold_sweep.csv`

## Figures (folder)
- `reports/figures/week12_xgb_week8_val`

> Note: Threshold tuning should be done on validation/CV. The test set remains locked for the final evaluation.
