# Week 3 · Day 5 — Scope & Goals

## In-scope (γίνονται τώρα)
- **Supervised binary classification** (fraud vs non-fraud) με το Kaggle `creditcard.csv`.
- **Offline training & evaluation** σε train/validation/test splits.
- **Μετρικές**: Precision, Recall, F1, **ROC-AUC & PR-AUC**.
- **Threshold tuning** (optimize για υψηλό Recall με ελεγχόμενο Precision).
- **Baseline μοντέλο**: Logistic Regression (με τυπικό scaling & class_weight όπου χρειάζεται).

## Out-of-scope (προς το παρόν)
- **Real-time streaming / low-latency inference**.
- **Concept drift handling** & online/continual learning.
- **Feature store / feature registry**.
- **Deep hyperparameter tuning** (προχωρημένο search, π.χ. Optuna/Bayes).

## Κριτήρια επιτυχίας (μετρήσιμα)
- **Recall (Fraud)** ≥ **0.90** με **Precision** ≥ **0.10** (αρχικό target για ισορροπία FN/FP).
- **PR-AUC** > baseline (βελτίωση έναντι απλής κατωφλιοποίησης/τυχαίου).
- Τεκμηριωμένη επιλογή threshold βάσει **PR curve** & **cost trade-offs**.
- Αναπαραγωγιμότητα: scripts/notebooks που τρέχουν end-to-end.

## Παραδοτέα
- **Κώδικας**: `src/` scripts (φόρτωση, baseline, αξιολόγηση, plots).
- **Report**: τεχνική αναφορά στο `reports/` (EDA, μέθοδοι, μετρικές, συμπεράσματα).
- **Notebook(s)**: για EDA/plots/πειραματισμούς (στο `notebooks/`).
- **REST API mock** (μήνας **3–4**): απλό endpoint που φορτώνει αποθηκευμένο μοντέλο και κάνει predict σε single/mini-batch (χωρίς real-time infra).
- **Demo UI (προαιρετικά)**: μικρό frontend για δοκιμή του API.

## Σημειώσεις
- Εστίαση στην **ερμηνευσιμότητα** και στην **ισορροπία σφαλμάτων** (FN vs FP).
- Οι στόχοι/όρια μπορεί να αναθεωρηθούν μετά τα πρώτα πειράματα validation.
