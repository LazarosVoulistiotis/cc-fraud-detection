# Week 3 · Day 5 — Scope & Goals

## In-scope (τρέχουσα φάση)
- **Supervised binary classification** (fraud vs non-fraud) με το dataset Kaggle `creditcard.csv`.
- **Offline training & evaluation** σε train/validation/test splits.
- **Μετρικές αξιολόγησης**: Precision, Recall, F1, **ROC-AUC & PR-AUC**.
- **Threshold tuning** με στόχο υψηλό Recall (να μη χάνονται απάτες) με ελεγχόμενο Precision (να μην αυξηθούν υπερβολικά τα false positives).
- **Baseline μοντέλο**: Logistic Regression (με scaling & χρήση `class_weight` λόγω ανισορροπίας).

## Out-of-scope (προσωρινά εκτός)
- **Real-time streaming / low-latency inference**.
- **Αντιμετώπιση concept drift** και online/continual learning.
- **Feature store / feature registry**.
- **Deep hyperparameter tuning** (π.χ. Optuna/Bayesian search).

## Κριτήρια επιτυχίας
- **Recall (Fraud)** ≥ **0.90** με **Precision** ≥ **0.10** (αρχικός στόχος ισορροπίας FN/FP).
- **PR-AUC** σημαντικά καλύτερο από baseline/random.
- Επιλογή threshold βασισμένη σε **PR curves** και ανάλυση κόστους.
- **Αναπαραγωγιμότητα**: scripts/notebooks που τρέχουν end-to-end χωρίς “manual hacks”.

## Παραδοτέα
- **Κώδικας**: `src/` scripts (data loading, baseline, evaluation, plots).
- **Report**: τεχνική αναφορά στο `reports/` (EDA, μεθοδολογία, αποτελέσματα, συμπεράσματα).
- **Notebook(s)**: για EDA και πειραματισμούς (στο `notebooks/`).
- **REST API mock** (μήνας **3–4**): endpoint που φορτώνει αποθηκευμένο μοντέλο και κάνει predict σε εισόδους (proof-of-concept).
- **Demo UI (προαιρετικό)**: απλό interface για δοκιμή του API.

## Σημειώσεις
- Ιδιαίτερη έμφαση σε **ερμηνευσιμότητα μοντέλου** (feature importances/SHAP).
- Εξισορρόπηση σφαλμάτων **FN vs FP**: FN κοστίζουν οικονομικά, FP κοστίζουν σε εμπειρία πελάτη.
- Οι στόχοι θα αναθεωρούνται μετά τα πρώτα αποτελέσματα validation.
