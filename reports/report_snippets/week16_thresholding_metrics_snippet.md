# Week 16 — Final Thresholding & Metrics (Report Snippet)

## Goal
After completing explainability (SHAP + LIME), we finalize an **operational decision threshold** that aligns with business constraints, and we report **threshold-dependent metrics** on a locked test set.

**Key principle:** the threshold is selected **only on the validation set**; the test set remains locked for final reporting.

---

## 1) Precision–Recall curve and threshold selection (Validation set)

We use the model’s predicted probabilities on the **validation set** to compute the Precision–Recall curve:

- `precision, recall, thresholds = precision_recall_curve(y_true, y_score)`

We consider three practical policies for selecting the final threshold:

1. **Max F1**  
   Select the threshold that maximizes:  
   \[
   F1 = \frac{2 \cdot (Precision \cdot Recall)}{Precision + Recall}
   \]

2. **Max F2 (recall-weighted)**  
   In fraud detection, missing a fraud (FN) is often more expensive than flagging a legitimate transaction (FP).  
   F2 weights recall higher than precision:
   \[
   F_\beta = \frac{(1+\beta^2) \cdot (Precision \cdot Recall)}{\beta^2 \cdot Precision + Recall}, \quad \beta=2
   \]

3. **Precision constraint (operational policy)**  
   Impose `precision ≥ p` (e.g., `p=0.80`) and select the **smallest** threshold that satisfies it.  
   This reflects analyst capacity / customer friction constraints.

We present the PR curve and mark the selected threshold point.

---

## 2) Cost-based policy (existing)

As an existing business-aligned policy, we also evaluate a **cost-based threshold** selected on validation:

- Assume costs: `cost_fp = 1`, `cost_fn = 20`
- Minimize:  
  \[
  Cost(threshold) = cost_{fp} \cdot FP + cost_{fn} \cdot FN
  \]
- The chosen validation threshold in this project is `thr = 0.0884`.

This policy explicitly prioritizes reducing missed frauds (FN), while keeping false alarms (FP) operationally manageable.

---

## 3) Confusion matrix and MCC (Test set, locked)

Using the selected threshold, we evaluate on the **locked test set** and report the confusion matrix:

- TP, FP, TN, FN

We also report **MCC (Matthews Correlation Coefficient)**, a strong single-number metric under severe class imbalance:

\[
MCC = \frac{TP\cdot TN - FP\cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}
\]

Finally, we report:
- Precision, Recall, F1, F2
- PR-AUC, ROC-AUC
- MCC
- Confusion matrix counts (TP/FP/TN/FN)

All metrics are computed on the test set using the validation-selected threshold (no tuning on test).
