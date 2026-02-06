# Week 9 — Baseline Logistic Regression (Business-Oriented)

**Setup:** scaler=standard, class_weight=balanced, threshold=0.5,
test_size=0.2, seed=42, optimize=none,
n_thresholds=101, cost_fp=1.0, cost_fn=20.0

## Μετρικές (Test set — default threshold 0.50)
- Precision: **0.0610**
- Recall: **0.9184**
- F1: **0.1144**
- ROC-AUC: **0.9721**
- PR-AUC (AP): **0.7190**
- Recall@Precision≥0.90: **0.0000**
- Precision@Recall≥0.90: **0.1415**

## Βέλτιστο threshold: (δεν ζητήθηκε)

## Εικόνες
- ROC: `reports/figures/week9/week9_logreg_roc.png`
- PR: `reports/figures/week9/week9_logreg_pr.png`
- Confusion (default): `reports/figures/week9/week9_logreg_confusion_default.png`
- Confusion (best): `reports/figures/week9/week9_logreg_confusion_best.png`
- Precision/Recall vs Threshold: `reports/figures/week9/week9_prec_recall_vs_threshold.png`
- Cost vs Threshold: `reports/figures/week9/week9_cost_vs_threshold.png`

## Αρχεία
- Μοντέλο: `models/logreg_baseline.joblib`
- Μετρικές: `reports/week9_baseline/metrics.json`
- Classification report: `reports/week9_baseline/classification_report.txt`
- Συντελεστές: `reports/week9_baseline/coefficients_sorted.csv`
- Threshold Sweep: `reports/week9_baseline/threshold_sweep.csv`
- Config (CLI args): `reports/week9_baseline/config.json`
- Log: `run_20251103-052430.log`

> Σημείωση: Το threshold tuning εδώ είναι για **αναλυτική αναφορά** πάνω στο test.
> Σε παραγωγική ροή, ορίζουμε validation/CV set για επιλογή threshold και κρατάμε το test “κλειστό”.
