# Week 21 — Freeze Check Summary

## Frozen artifacts
- Model artifact: `models/xgb_final.joblib`
- Threshold config: `configs/threshold.json`
- Feature schema: `configs/feature_schema.json`

## Locked serving policy
- Policy name: `precision_constraint_p80`
- Policy version: `week16-locked`
- Selected on: `validation`
- Threshold: `0.1279`

## Frozen engineered features
- `Hour`
- `hour_sin`
- `hour_cos`
- `Amount_log1p`

## Final serving note
Week 21 evaluation will use the frozen model, frozen schema, and frozen threshold config as-is.
No re-training and no threshold re-selection will be performed on the hold-out test set.
