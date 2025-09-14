---
title: Week 3 — Binary Classification, Metrics & Overfitting
author: Lazaros Voulistiotis
repo: cc-fraud-detection
---

## 🎯 Στόχος εβδομάδας
- Κατανόηση **binary classification**, βασικών **μετρικών**.
- Πρακτική σε **Logistic Regression** (baseline + regularization).
- Εφαρμογή **threshold tuning** με στόχο Recall ≥ 0.90.

---

## 📌 Ημέρα 1 — Binary Classification & Metrics ✅

### Βασικές έννοιες
- **Sigmoid / Υπόθεση**
  $$
  \sigma(z)=\frac{1}{1+e^{-z}}, \quad h_\theta(x)=\sigma(\theta^\top x)
  $$
- **Decision boundary**
  $$
  \theta^\top x = 0 \;\; \Rightarrow \;\; h_\theta(x)=0.5
  $$
- **Cost με L2 regularization**
  $$
  J(\theta) = -\frac{1}{m}\sum_{i=1}^{m}[y^{(i)}\log h_\theta(x^{(i)})+(1-y^{(i)})\log(1-h_\theta(x^{(i)}))]+\frac{\lambda}{2m}\sum_{j=1}^{n}\theta_j^2
  $$

### Confusion Matrix
- TP: σωστά fraud  
- FP: false alarms  
- FN: χαμένες απάτες (κρίσιμο)  
- TN: σωστά normal

### Metrics
- Accuracy: $\frac{TP+TN}{TP+FP+FN+TN}$  
- Precision: $\frac{TP}{TP+FP}$  
- Recall: $\frac{TP}{TP+FN}$  
- F1: $\frac{2PR}{P+R}$

⚠️ Σε **class imbalance** προτιμώ F1 + PR-AUC.

---

## 📌 Ημέρα 2 — Overfitting, Regularization & Threshold Tuning ✅

### Κεντρικές σημειώσεις
- **Bias/Variance trade-off**
- **Regularization (L2):** μικρό C ⇒ ισχυρότερη regularization.
- **ROC vs PR:** PR πιο κατατοπιστική σε fraud detection.

### Πειράματα
- Grid σε C = {0.01, 0.1, 1, 10}.
- Μετρικές: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC.
- Επιλογή καλύτερου C βάσει PR-AUC.
- Threshold tuning για Recall ≥ 0.90.

### Αποτελέσματα
- Καλύτερο **C**: `0.01` (PR-AUC = 0.7114)
- Επιλεγμένο threshold: `0.2259`

| Metric     | Value   |
|------------|---------|
| Accuracy   | 0.9302  |
| Precision  | 0.0220  |
| Recall     | 0.9054 ✅ |
| F1         | 0.0430  |
| ROC-AUC    | 0.9689  |
| PR-AUC     | 0.7114  |

**Confusion Matrix @ thr=0.226**

|            | Pred 0 | Pred 1 |
|------------|--------|--------|
| **True 0** | 79349  | 5946   |
| **True 1** | 14     | 134    |

- TN = 79,349 | FP = 5,946  
- FN = 14 | TP = 134  

**Trade-offs:**  
- ✅ Εντοπίζουμε το 90.5% των fraud.  
- ❌ Πολλά FP (χαμηλή precision). Αποδεκτό γιατί προτεραιότητα είναι Recall.

---