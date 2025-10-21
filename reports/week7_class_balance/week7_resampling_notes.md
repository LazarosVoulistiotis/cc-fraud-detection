---
title: "Week 7 — Resampling Notes (Oversampling / Undersampling / SMOTE)"
date: 2025-10-21
---

## Πρόβλημα
Ισχυρή ανισορροπία θετικής κλάσης (~0.17% fraud). Θέλουμε να αυξήσουμε **Recall** χωρίς να εκτοξευτεί το **False Positive Rate** (άρα **Precision** παραμένει διαχειρίσιμη για την ομάδα ελέγχου).

> Business framing: στο σημερινό base rate αυτό μεταφράζεται σε ~1 απάτη ανά ~600 συναλλαγές. Οποιοδήποτε μοντέλο πρέπει να στοχεύει σε υψηλή ανάκτηση απάτης με **περιορισμένο αριθμό alerts/ημέρα** (budget).

## Τεχνικές
- **Oversampling (RandomOverSampler)**  
  Αντιγράφει δείγματα της μειοψηφίας μέχρι να ισορροπήσει η αναλογία. **Πλεονέκτημα:** γρήγορο, απλό. **Κίνδυνος:** overfitting (ειδικά με γραμμικά μοντέλα), δεν «δημιουργεί» νέα πληροφορία.
- **Undersampling (RandomUnderSampler, NearMiss)**  
  Πετάει τυχαία (ή επιλεκτικά) δείγματα της πλειοψηφίας. **Πλεονέκτημα:** μειώνει χρόνο/μνήμη, κάνει το πρόβλημα πιο «εύκολο». **Κίνδυνος:** χάνεται πληροφορία από την πλειοψηφία.
- **SMOTE (Synthetic Minority Over-sampling Technique)**  
  Δημιουργεί **συνθετικά** δείγματα της μειοψηφίας με παρεμβολή μεταξύ κοντινών γειτόνων. Συνήθως καλύτερο από «απλή» αντιγραφή.
  - Παραλλαγές: **Borderline-SMOTE**, **SMOTE-Tomek**, **SMOTEENN**, **ADASYN**.
- **Class weights**  
  Σε μοντέλα που το υποστηρίζουν (π.χ. `LogisticRegression`, `LinearSVC`, `RandomForest`, `XGBoost (scale_pos_weight)`), τοποθετούμε **μεγαλύτερο βάρος** στα σπάνια παραδείγματα. Συχνά επαρκεί, ειδικά για δενδροειδή.
- **Υβριδικές προσεγγίσεις**  
  SMOTE **και μετά** undersampling της πλειοψηφίας (π.χ. `SMOTEENN`, `SMOTETomek`).

## Αρχές αποφυγής data leakage
- Κάθε μορφή **resampling εφαρμόζεται μόνο στο training** και **μέσα στα folds** του cross‑validation.
- Δεν αγγίζουμε **validation/test** σύνολα.
- Χρησιμοποιούμε **`imblearn.pipeline.Pipeline`** ώστε το `fit_resample` να εκτελείται **πριν** το μοντέλο και **μέσα** στο CV.

## Μετρικές αξιολόγησης
- **PR‑AUC (Average Precision)**: πιο κατάλληλη για rare positives από ROC‑AUC.
- **Recall, Precision, F1** και **Recall@k** (π.χ. top‑0.5% των συναλλαγών ως alerts).
- **Confusion Matrix** σε συγκεκριμένο threshold.
- **Κοστολόγηση**: False Negative ≫ False Positive. Ρυθμίζουμε threshold σύμφωνα με **capacity/κόστος**.

## Σχέδιο πειραμάτων (Month 3)
1. **Baseline**: Logistic Regression χωρίς/με `class_weight="balanced"` (χωρίς resampling).
2. **SMOTE + LogReg** (pipeline).
3. **RandomUnderSampler + LogReg**.
4. **SMOTE + RandomForest / XGBoost** (light baseline για δενδροειδή).
5. **Καμπύλες Precision‑Recall** και επιλογή threshold ώστε **Recall ≥ 0.90** με διαχειρίσιμο Precision/alerts.

## Εγκατάσταση
```bash
pip install imbalanced-learn
```
Προσθήκη στο `requirements.txt`:
```
imbalanced-learn>=0.12
```

---

## Code templates (business‑ready, χωρίς leakage)

### 1) Cross‑validation με Pipeline + SMOTE
```python
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, average_precision_score, f1_score, recall_score, precision_score
import numpy as np

X = df.drop(columns=['Class'])
y = df['Class'].astype(int)

pipe = Pipeline([
    ('smote', SMOTE(k_neighbors=5, random_state=42)),
    ('clf', LogisticRegression(max_iter=2000, solver='lbfgs'))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scorers = {
    'pr_auc': make_scorer(average_precision_score, needs_proba=True),
    'recall': make_scorer(recall_score),
    'precision': make_scorer(precision_score),
    'f1': make_scorer(f1_score)
}

res = cross_validate(pipe, X, y, scoring=scorers, cv=cv, n_jobs=-1, return_train_score=False)
print({k: np.mean(v) for k, v in res.items() if k.startswith('test_')})
```

### 2) Under‑sampling στο training μόνο
```python
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ('rus', RandomUnderSampler(random_state=42)),
    ('clf', LogisticRegression(max_iter=2000))
])
# ίδια διαδικασία με το παραπάνω για CV & scorers
```

### 3) Μόνο class weights (χωρίς resampling)
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

clf = LogisticRegression(max_iter=2000, class_weight='balanced')
# CV όπως πριν (χωρίς imblearn.Pipeline)
```

> **Σημείωση για δενδροειδή**: RandomForest/XGBoost συχνά αποδίδουν καλά με `class_weight`/`scale_pos_weight` χωρίς SMOTE. Δοκίμασέ τα ως baseline πριν από βαριές υβριδικές λύσεις.

---

## Επιλογή threshold με βάση business capacity
1. Παίρνουμε **probabilities** στο validation.
2. Υπολογίζουμε **precision‑recall curve**.
3. Διαλέγουμε threshold ώστε τα **καθημερινά alerts** να μην ξεπερνούν το διαθέσιμο capacity (π.χ. 0.5% της ροής) **και** η ανάκτηση να μην πέφτει κάτω από τον στόχο (π.χ. 90%).

### Ψευδο‑κώδικας
```python
proba = clf.predict_proba(X_val)[:,1]
thr = np.quantile(proba, 1 - alert_budget)  # π.χ. alert_budget=0.005 για top 0.5%
pred = (proba >= thr).astype(int)
# report precision, recall, confusion
```

---

## Κίνδυνοι & καλές πρακτικές
- **Data leakage** από SMOTE/oversampling εκτός CV → **απαγορευτικό**.
- **Overfitting** με oversampling → ελέγχουμε PR‑AUC/Recall σε **hold‑out**.
- **Drift** του base rate → παρακολούθηση PR‑AUC, predicted positives/day, recall on labelled feedback.
- **Explainability**: κρατάμε αναφορά των pipelines και random seeds στα reports.

---

## TL;DR – Συστάσεις
- Ξεκίνα με **class weights** και **LogReg/Tree** ως baseline.
- Δοκίμασε **SMOTE + LogReg** και **UnderSampling + LogReg** με 5‑fold **Stratified CV**.
- Τελική επιλογή με βάση **PR‑AUC** και **Recall@k** στα **alerts** που μπορεί να χειριστεί η ομάδα.
- Ό,τι και να διαλέξεις, **ενσωμάτωσε** το σε **`imblearn.pipeline.Pipeline`** και κάνε **fit μόνο στο `X_train`**.
