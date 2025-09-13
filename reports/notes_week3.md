---
title: Week 3 — Binary Classification, Metrics & Overfitting
author: Lazaros Voulistiotis
repo: cc-fraud-detection
---

## 🎯 Στόχος εβδομάδας
- Να κατανοήσω σε βάθος **binary classification**, βασικές **μετρικές**, και πώς να αποφεύγω **overfitting** με **regularization**.
- Να έχω baseline **Logistic Regression** + καμπύλες **ROC/PR** και να τεκμηριώσω **επιλογή threshold**.

---

## 📌 Ημέρα 1 — Binary Classification & Metrics ✅

### 📺 Παρακολούθηση
- Andrew Ng — *Logistic Regression for Classification*.

**Βασικές έννοιες**
- **Sigmoid / Υπόθεση**
  \[
  \sigma(z)=\frac{1}{1+e^{-z}}, \quad h_\theta(x)=\sigma(\theta^\top x)
  \]
- **Decision boundary**
  \[
  \theta^\top x = 0 \;\; \Rightarrow \;\; h_\theta(x)=0.5
  \]
  (το threshold δεν χρειάζεται να είναι 0.5 στην πράξη).
- **Cost (Log Loss) με L2 regularization**
  \[
  J(\theta)= -\frac{1}{m}\sum_{i=1}^{m}\Big[y^{(i)}\log h_\theta(x^{(i)}) + (1-y^{(i)})\log(1-h_\theta(x^{(i)}))\Big] \;+\; \frac{\lambda}{2m}\sum_{j=1}^{n}\theta_j^2
  \]
  όπου η L2 ποινή μειώνει το overfitting.

---

### 📖 Logistic Regression συνοπτικά
- Δυαδική ταξινόμηση, προβλέπει **πιθανότητα** θετικής κλάσης:
  \[
  f_{w,b}(x) \approx P(y=1 \mid x; w,b)
  \]
- Γιατί όχι Linear Regression; → μπορεί να δώσει τιμές <0 ή >1. Το logistic δίνει **S-σχήματος** καμπύλη (sigmoid) και ορίζει πιθανότητα.

---

### 🧮 Confusion Matrix (binary)
- **TP**: σωστά προβλεπόμενα fraud  
- **FP**: false alarms (κανονικές που δηλώθηκαν fraud)  
- **FN**: χαμένες απάτες (κρίσιμες)  
- **TN**: σωστά προβλεπόμενες κανονικές

---

### 📏 Metrics
- **Accuracy**
  \[
  \frac{TP+TN}{TP+FP+FN+TN}
  \]
  ⚠️ Παραπλανητικό σε **class imbalance**.
- **Precision (P)**
  \[
  \frac{TP}{TP+FP}
  \]
  Από όσα «σήμανα ως fraud», πόσα ήταν όντως fraud;
- **Recall (R)**
  \[
  \frac{TP}{TP+FN}
  \]
  Από όλα τα πραγματικά fraud, πόσα βρήκα;
- **F1**
  \[
  F1=\frac{2PR}{P+R}
  \]
  Εξισορροπεί Precision & Recall → προτιμάται σε ανισορροπία κλάσεων.

**Του καμπύλες**
- **TPR (Recall):** \(\frac{TP}{TP+FN}\)  
- **FPR:** \(\frac{FP}{FP+TN}\)

---

### 📈 ROC-AUC vs PR-AUC
- **ROC-AUC:** TPR vs FPR. Γενική εικόνα, αλλά σε πολύ άνισες κλάσεις μπορεί να φαίνεται «υψηλό» λόγω πολλών TN.
- **PR-AUC (Average Precision):** Precision–Recall πάνω στη **θετική** κλάση → πιο κατατοπιστικό όταν το fraud είναι σπάνιο (<1%).

**Κανόνας:** Σε **imbalance** αναφέρω **F1 + PR-AUC**, μαζί με ROC-AUC.

---

### 🖥️ Mini Hands-on (70/30, stratify + Pipeline)
- **Split:** Train/Test 70/30, `stratify=y`
- **Pipeline:**
  ```python
  from sklearn.model_selection import train_test_split
  from sklearn.pipeline import make_pipeline
  from sklearn.preprocessing import StandardScaler
  from sklearn.linear_model import LogisticRegression
  from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                               roc_auc_score, average_precision_score, ConfusionMatrixDisplay,
                               RocCurveDisplay, PrecisionRecallDisplay)
  import pandas as pd
  import matplotlib.pyplot as plt
  from pathlib import Path

  # X, y: προετοιμασμένα features/target
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.30, random_state=42, stratify=y
  )

  model = make_pipeline(
      StandardScaler(),
      LogisticRegression(class_weight="balanced", max_iter=1000)
  )
  model.fit(X_train, y_train)

  proba = model.predict_proba(X_test)[:, 1]
  y_pred = (proba >= 0.5).astype(int)

  metrics = {
      "accuracy": accuracy_score(y_test, y_pred),
      "precision": precision_score(y_test, y_pred, zero_division=0),
      "recall": recall_score(y_test, y_pred),
      "f1": f1_score(y_test, y_pred),
      "roc_auc": roc_auc_score(y_test, proba),
      "pr_auc": average_precision_score(y_test, proba),
  }
  Path("reports").mkdir(parents=True, exist_ok=True)
  pd.DataFrame([metrics]).to_csv("reports/week3_day1_metrics.csv", index=False)

  # Plots
  Path("images/week3").mkdir(parents=True, exist_ok=True)
  RocCurveDisplay.from_predictions(y_test, proba)
  plt.title("ROC Curve (Baseline)")
  plt.savefig("images/week3/roc_curve.png", dpi=150, bbox_inches="tight"); plt.close()

  PrecisionRecallDisplay.from_predictions(y_test, proba)
  plt.title("Precision–Recall Curve (Baseline)")
  plt.savefig("images/week3/pr_curve.png", dpi=150, bbox_inches="tight"); plt.close()

  ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
  plt.title("Confusion Matrix (Threshold = 0.5)")
  plt.savefig("images/week3/confusion_matrix.png", dpi=150, bbox_inches="tight"); plt.close()
                                                     
ΗΜΕΡΑ 2
Τι σημαίνει bias/variance, τι δείχνουν τα learning curves, πώς η regularization μειώνει το overfitting (με ελαφρά αύξηση bias)

## TODO / Επόμενα βήματα
- [ ] Δοκιμή άλλων αλγορίθμων (e.g., RandomForest, XGBoost)  
- [ ] Cross-validation & learning curves  
- [ ] Calibration (προβλέψεις ως πιθανότητες)


