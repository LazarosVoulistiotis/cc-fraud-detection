# Table — Week 21 Final Locked-Test Evaluation

**Table X. Final locked-test evaluation of the frozen XGBoost champion model (Week 21)**

| Metric | Value |
|---|---:|
| ROC-AUC | 0.96995 |
| PR-AUC | 0.81713 |
| Precision | 0.82796 |
| Recall | 0.81053 |
| F1-score | 0.81915 |
| TN | 56,635 |
| FP | 16 |
| FN | 18 |
| TP | 77 |
| Alerts per 10,000 transactions | 16.39 |
| Cost per transaction (FP=1, FN=20) | 0.006626 |

**Interpretation:**  
The frozen Week 21 evaluation reproduced the expected locked-test result exactly. The final serving configuration maintained strong precision and controlled false positives while preserving good fraud recall under severe class imbalance.
