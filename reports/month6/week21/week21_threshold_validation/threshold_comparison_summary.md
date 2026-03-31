# Week 21 — Threshold Sensitivity Analysis (Post-hoc, No Leakage)

This comparison is reported as sensitivity analysis only. 
No new threshold is selected from the test set.

| Policy | Threshold | Precision | Recall | F1 | TP | FP | FN | Alerts/10k | Cost/tx |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| final_locked_precision_constraint_p80 | 0.127900 | 0.8280 | 0.8105 | 0.8191 | 77 | 16 | 18 | 16.39 | 0.006626 |
| historical_cost_based_reference | 0.088400 | 0.7938 | 0.8105 | 0.8021 | 77 | 20 | 18 | 17.09 | 0.006697 |
| posthoc_reference_recall_ge_090 | 0.000054 | 0.0118 | 0.9053 | 0.0233 | 86 | 7186 | 9 | 1281.50 | 0.129807 |