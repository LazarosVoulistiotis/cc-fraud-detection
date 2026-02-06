# Week 10 — Decision Tree Baseline (Business-Oriented)

**Setup:** criterion=gini, max_depth=12, min_samples_split=2,
min_samples_leaf=1, class_weight=balanced, random_state=42,
threshold=0.99, optimize=none, n_thresholds=101, cost_fp=1.0, cost_fn=20.0

## Μετρικές (Test set — default threshold 0.99)
- Accuracy: **0.998343**
- Precision: **0.5036**
- Recall: **0.7263**
- F1: **0.5948**
- ROC-AUC: **0.8678**
- PR-AUC (AP): **0.4944**
- Recall@Precision≥0.90: **0.0000**
- Precision@Recall≥0.90: **0.0017**

## Βέλτιστο threshold: (δεν ζητήθηκε)

## Εικόνες
- ROC: `reports/figures/week10/week10_dt_roc.png`
- PR: `reports/figures/week10/week10_dt_pr.png`
- Confusion (default): `reports/figures/week10/week10_dt_confusion_default.png`
- Confusion (best): `reports/figures/week10/week10_dt_confusion_best.png`
- Precision/Recall vs Threshold: `reports/figures/week10/week10_dt_prec_recall_vs_threshold.png`
- Cost vs Threshold: `reports/figures/week10/week10_dt_cost_vs_threshold.png`

## Αρχεία
- Μοντέλο: `models/dt_baseline.joblib`
- Μετρικές: `reports/week10_dt_baseline/metrics.json`
- Classification report: `reports/week10_dt_baseline/classification_report.txt`
- Feature importances: `reports/week10_dt_baseline/feature_importances.csv`
- Threshold Sweep: `reports/week10_dt_baseline/threshold_sweep.csv`
- Config (CLI args): `reports/week10_dt_baseline/config.json`
- Log: `run_20260205-204609.log`

> Σημείωση: Το threshold tuning εδώ είναι για **αναλυτική αναφορά** πάνω στο test.
> Σε παραγωγική ροή, ορίζουμε validation/CV για επιλογή threshold και κρατάμε το test “κλειστό”.
