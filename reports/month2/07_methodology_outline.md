---
title: "Week 7 — Methodology Outline (Fraud Detection)"
date: 2025-10-21
---

## Στόχος
Ανίχνευση απάτης ως **δυαδική ταξινόμηση** με **εξαιρετικά μη ισορροπημένες κλάσεις** (~0.17% fraud). Στόχος: **υψηλό Recall** (να «πιάνουμε» τις απάτες) με **αποδεκτό Precision** (να μην «χτυπάμε» άδικα πολλές νόμιμες συναλλαγές).

## Προεπεξεργασία
- **Scaling `Amount`**: `log1p → z-score` (ή σκέτο z-score) όπως τεκμηριώθηκε στο *Week 6*· ο μετασχηματισμός γίνεται **μέσα σε Pipeline** για αποφυγή leakage.
- **Split**: `train/test` με `stratify=y` και σταθερό `random_state`.
- **Feature scaling**: με `StandardScaler` **fit μόνο σε X_train** (μέσα στο pipeline).

## Αντιμετώπιση ανισορροπίας
- **Class weights**: όπου υποστηρίζεται, δοκιμή `class_weight="balanced"`.
- **Resampling ΜΟΝΟ στο training**: `SMOTE`, `RandomUnderSampler`, υβριδικά όπως `SMOTE-Tomek` / `SMOTEENN`.
- Εφαρμογή **εντός cross‑validation folds** με `imblearn.pipeline.Pipeline` (ώστε το `fit_resample` να εκτελείται σωστά σε κάθε fold).

## Αλγόριθμοι που θα δοκιμαστούν
- **Logistic Regression** (baseline, και tuned `C`).
- **Tree-based**: Decision Tree, Random Forest, προαιρετικά **XGBoost/LightGBM** ως ισχυρά baseline.
- (Μετά) **Linear SVM** ή **Calibrated** μοντέλα για ποιοτικές πιθανότητες.
- Για κάθε αλγόριθμο: εκδοχή με/χωρίς `class_weight`, και με **SMOTE** ή/και **undersampling**.

## Αξιολόγηση
- **PR-AUC** (Average Precision) ως κύρια συνολική μετρική για rare positives.
- **Recall**, **Precision**, **F1**, **Recall@k** (π.χ. top‑k alerts που μπορεί να ελέγχει η ομάδα).
- **Threshold tuning** βάσει επιχειρησιακού κόστους (False Negatives ≫ False Positives).
- **StratifiedKFold CV** (π.χ. 5-fold) + τελικό **hold‑out test** μόνο για τελικό report.

## Παραδοτέα (Week 7 → Month 3)
- Κώδικας pipelines (Python/Notebook).
- Αναφορές μετρικών (CSV/JSON) ανά πείραμα και plots (PR/ROC/Confusion).
- Τεκμηρίωση thresholds και trade‑offs (π.χ. Recall≥0.90 με ελεγχόμενο Precision).

---

## Mini‑Snippet: Pipeline με SMOTE (για χρήση σε Notebook / Πειράματα)
> Χρήση **χωρίς leakage**: scaling + SMOTE μέσα στο `Pipeline`, με **StratifiedKFold**.

```python
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# Φόρτωση dataset (προσαρμόστε μονοπάτι αν χρειάζεται)
df = pd.read_csv("data/data_raw/creditcard.csv")
X = df.drop(columns=["Class"]).values
y = df["Class"].values

pipe = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42, sampling_strategy=0.05)),  # ~5% minority vs majority
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
])


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
ap_scorer = make_scorer(average_precision_score, needs_proba=True)

scores = cross_val_score(pipe, X, y, scoring="average_precision", cv=cv, n_jobs=-1)
print(f"PR-AUC (CV mean ± std): {scores.mean():.4f} ± {scores.std():.4f}")
```

### Εγκατάσταση
```bash
pip install -U imbalanced-learn scikit-learn
```
Προσθήκη στο `requirements.txt`:
```txt
imbalanced-learn>=0.14
scikit-learn>=1.7
```

---

## README Snippet (Usage)
```markdown
### Week 7 — Class Imbalance
Generate class balance plot + JSON:

```bash
python src/07_class_imbalance.py \
  --data data/data_raw/creditcard.csv \
  --target Class \
  --reports reports \
  --images reports/figures/week7 \
  --positive-label 1 \
  --normalize \s
  --log-y \
  --title "Class Balance (Non-Fraud vs Fraud)" \
  --log-level INFO
  ```

Outputs:
- `reports/figures/week7/class_balance_bar.png`
- `reports/figures/week7/class_balance_pct.png`
- `reports/week7_class_balance/week7_class_balance.json`
```
---

## Kanban / Suggested Commits
- [ ] **Visualize class imbalance (bar plot + json).**
  ```bash
  git add src/07_class_imbalance.py reports/week7_class_balance/week7_class_balance.json reports/figures/week7/class_balance_bar.png reports/figures/week7/class_balance_pct.png
  git commit -m "week7: class imbalance (bar plots + json)"
  ```
- [ ] **Research resampling (notes).**
  ```bash
  git add reports/week7_resampling_notes.md requirements.txt
  git commit -m "week7: resampling notes (SMOTE/over/under) + imbalanced-learn req"
  ```
- [ ] **Add methodology outline.**
  ```bash
  git add reports/week7_methodology_outline.md notebooks/week7/smote_demo.ipynb
  git commit -m "week7: methodology outline + SMOTE pipeline demo notebook"
  ```
- [ ] **Push**
  ```bash
  git push origin main
  ```
