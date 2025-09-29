---
title: Week 3 — Binary Classification, Metrics & Overfitting
author: Lazaros Voulistiotis
repo: cc-fraud-detection
---

## 🎯 Στόχος Εβδομάδας
- Κατανόηση **binary classification** και βασικών **μετρικών απόδοσης**.
- Πρακτική σε **Logistic Regression** (baseline + regularization).
- Εφαρμογή **threshold tuning** με στόχο Recall ≥ 0.90.
- Καταγραφή trade-offs (Precision vs Recall) σε συνθήκες **class imbalance**.

---

## 📌 Ημέρα 1 — Binary Classification & Metrics ✅

### Βασικές Έννοιες
- **Sigmoid / Υπόθεση**
  $$
  \sigma(z) = \frac{1}{1 + e^{-z}}, \quad h_\theta(x) = \sigma(\theta^\top x)
  $$

- **Decision boundary**
  $$
  \theta^\top x = 0 \;\; \Rightarrow \;\; h_\theta(x) = 0.5
  $$

- **Cost με L2 Regularization**
  $$
  J(\theta) = -\frac{1}{m}\sum_{i=1}^{m}\Big[y^{(i)}\log h_\theta(x^{(i)})+(1-y^{(i)})\log(1-h_\theta(x^{(i)}))\Big] + \frac{\lambda}{2m}\sum_{j=1}^{n}\theta_j^2
  $$

---

### Confusion Matrix
- **TP**: σωστά fraud  
- **FP**: false alarms  
- **FN**: χαμένες απάτες (κρίσιμο κόστος)  
- **TN**: σωστά normal

---

### Metrics
- **Accuracy:** $\frac{TP+TN}{TP+FP+FN+TN}$  
- **Precision:** $\frac{TP}{TP+FP}$  
- **Recall (Sensitivity):** $\frac{TP}{TP+FN}$  
- **F1:** $\frac{2 \cdot Precision \cdot Recall}{Precision + Recall}$  

⚠️ Σε datasets με **class imbalance** (π.χ. fraud detection), το Accuracy είναι παραπλανητικό.  
👉 Δίνουμε έμφαση σε **Recall**, **F1** και **PR-AUC**.

---

## 📌 Ημέρα 2 — Overfitting, Regularization & Threshold Tuning ✅

### Κεντρικές Σημειώσεις
- **Bias/Variance trade-off**:  
  - Χαμηλό bias → μεγαλύτερος κίνδυνος overfitting.  
  - Υψηλό variance → φτωχή γενίκευση.

- **Regularization (L2):**  
  - Μικρό `C` → πιο ισχυρή regularization.  
  - Μεγάλο `C` → πιο “χαλαρή” regularization.

- **ROC vs PR:**  
  - ROC καλή για balanced datasets.  
  - PR πιο χρήσιμη σε **extreme imbalance** (fraud).

---

### Πειράματα
- Grid search σε `C ∈ {0.01, 0.1, 1, 10}`.
- Αξιολόγηση με: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC.
- Επιλογή **καλύτερου C** βάσει PR-AUC.
- Threshold tuning → στόχος **Recall ≥ 0.90**.

---

### Αποτελέσματα
- **Βέλτιστο C:** `0.01`  
- **Επιλεγμένο threshold:** `0.2259`

| Metric     | Value   |
|------------|---------|
| Accuracy   | 0.9302  |
| Precision  | 0.0220  |
| Recall     | 0.9054 ✅ |
| F1         | 0.0430  |
| ROC-AUC    | 0.9689  |
| PR-AUC     | 0.7114  |

---

### Confusion Matrix (thr=0.226)

|            | Pred 0 | Pred 1 |
|------------|--------|--------|
| **True 0** | 79,349 | 5,946  |
| **True 1** | 14     | 134    |

- TN = 79,349  
- FP = 5,946  
- FN = 14  
- TP = 134  

---

### Trade-offs
- ✅ Πολύ υψηλό **Recall (90.5%)** → εντοπίζουμε σχεδόν όλες τις απάτες.  
- ❌ Χαμηλή Precision (~2.2%) → πολλά false positives (αποδεκτό trade-off σε fraud detection).  
- ℹ️ Επόμενα βήματα:  
  - Δοκιμή άλλων μοντέλων (Trees/Ensembles).  
  - Cost-sensitive learning.  
  - Explainability (SHAP, feature importance).  

---
