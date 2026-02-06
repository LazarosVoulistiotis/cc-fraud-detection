# Week 12 — Business-Oriented Model Selection (Scorecard → Threshold Policy → Final Model)

## 0) Why this week matters (1 paragraph)
Αν οι προηγούμενες εβδομάδες απάντησαν στο «ποια μοντέλα δουλεύουν τεχνικά», αυτή η εβδομάδα απαντά στο «ποιο μοντέλο έχει επιχειρηματική αξία». Μετατρέπουμε metrics → decision policy (threshold) → επιχειρησιακό αφήγημα. Το threshold δεν είναι “0.5 by default”, είναι risk policy.

---

## 1) Fair comparison (“common ground”)
Για δίκαιη σύγκριση:
- Σταθερά splits train/val/test (stratified, seed=42).
- Locked test set: **δεν κάνουμε tuning** πάνω στο test.
- Threshold policy επιλέγεται στο **validation** και εφαρμόζεται “as-is” στο test.

---

## 2) Evidence artifacts (ώστε να μη γράφουμε στον αέρα)
Για κάθε μοντέλο έχουμε/παράγουμε:
- metrics.json
- threshold_sweep.csv (ή *_val / *_test)
- classification_report*.txt
- Figures: PR/ROC curves, confusion matrices, threshold/cost plots

Runs / folders:
- LogReg: reports/week9_logreg_week8/
- DT: reports/week10_dt_week8_val/ + reports/week10_dt_week8_test/
- RF: reports/week11_rf_week8_val/ + reports/week11_rf_week8_test/ + reports/week11_rf_week8_test_recall90/
- XGB: reports/week12_xgb_week8_val/ + reports/week12_xgb_week8_test/

---

## 3) Step 1 — Unified Scorecard (locked TEST, threshold policy from VAL)
Focus metrics: Precision/Recall (Fraud), F1, ROC-AUC, PR-AUC, TP/FP/FN, Cost/Tx.

### 3.1 Logistic Regression (TEST)
- Default thr=0.50 → P=0.0541, R=0.8737, F1=0.1019
- Business thr=0.99 → P=0.6466, R=0.7895, F1=0.7109
Takeaway: baseline useful for interpretability, αλλά χρειάζεται policy threshold για να γίνει επιχειρησιακά βιώσιμο.

### 3.2 Decision Tree (TEST @ thr=0.99 από VAL)
- P=0.5036, R=0.7263, F1=0.5948
- ROC-AUC=0.8678, PR-AUC=0.4944
Takeaway: explainable αλλά υπολείπεται από ensembles στο locked test.

### 3.3 Random Forest (TEST @ thr=0.2354 από VAL cost-opt)
- P=0.7549, R=0.8105, F1=0.7817
- TP=77, FP=25, FN=18, Cost/Tx=0.006785

### 3.4 Random Forest (Fraud-first stress test, TEST @ thr=0.0198)
- P=0.0437, R=0.8632, F1=0.0832
- TP=82, FP=1795, FN=13, Cost/Tx=0.036214
Takeaway: μικρό κέρδος σε FN, τεράστιο operational overload σε FP → μη βιώσιμο.

### 3.5 XGBoost (TEST @ thr=0.0884 από VAL cost-opt)
- P=0.7938, R=0.8105, F1=0.8021
- ROC-AUC=0.9699, PR-AUC=0.8171
- TP=77, FP=20, FN=18, Cost/Tx=0.006697

---

## 4) Step 4 — Threshold tuning with business objective (no “0.5”)
Policy A: Cost-optimal (FP=1, FN=20) — select threshold on VAL (min cost_per_tx), apply to TEST.
- RF cost-policy: TP=77 FP=25 FN=18
- XGB cost-policy: TP=77 FP=20 FN=18
Takeaway: ίδιο fraud protection, XGB έχει λιγότερα false alarms.

Policy B: Fraud-first (target recall) — stress test για operational feasibility.
- RF fraud-first: TP=82 FP=1795 FN=13
Takeaway: alerts explosion → operationally painful.

---

## 5) Step 5 — Confusion Matrix as business tool
Locked TEST size: N=56,746
Μετάφραση:
- TP = “Frauds stopped”
- FN = “Frauds missed” (fraud leakage)
- FP = “False alarms” (customer friction + manual review workload)

Operational view (ανά 10,000 tx):
- XGB (cost-policy): Alerts/10k ≈ 17.09, Missed/10k ≈ 3.17
- RF (cost-policy): Alerts/10k ≈ 17.97, Missed/10k ≈ 3.17
- RF (fraud-first): Alerts/10k ≈ 330.77, Missed/10k ≈ 2.29

---

## 6) Step 6 — Final model selection (decision memo)
Shortlist: Random Forest vs XGBoost.

Decision rule: επιλέγουμε μοντέλο που παρέχει:
1) υψηλό recall (fraud prevention),
2) επιχειρησιακά αποδεκτό precision / manageable FP,
3) σταθερότητα στο locked test,
4) cost-sensitive learning & production-ready pipeline.

Final choice: **XGBoost**
- ίδιο TP/FN με RF στο cost-policy,
- λιγότερα false alarms (FP 20 vs 25),
- ισχυρή επίδοση σε PR-AUC/F1.

Thesis-ready sentence:
«Το XGBoost επιλέγεται ως τελικό μοντέλο, καθώς προσφέρει τον καλύτερο συμβιβασμό μεταξύ υψηλού recall και επιχειρησιακά αποδεκτού precision, ενώ υποστηρίζει cost-sensitive learning μέσω class weighting και threshold policy, και μπορεί να επεκταθεί εύκολα σε παραγωγικό περιβάλλον.»

---

## 7) What goes into the report (Week 12)
- Modeling Strategy
- Model Comparison & Evaluation (scorecard + fairness)
- Handling Class Imbalance (PR-AUC, weighting, threshold policy)
- Business Trade-offs in Fraud Detection (TP/FP/FN narrative)
- Final Model Selection (decision memo + thesis sentence)

---

## 8) Deliverables (End of Week 12)
- Final model artifact: models/xgb_week8.joblib
- Final threshold policy: thr=0.0884 (VAL-selected), cost_fp=1, cost_fn=20
- Scorecard + business narrative “γιατί αυτό”
- Ready figures/tables: PR/ROC, confusion matrices, threshold sweeps
- Bridge to Month 4: interpretability + monitoring + deployment
