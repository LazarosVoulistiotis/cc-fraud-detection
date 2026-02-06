# Month 3 Milestone — Weeks 9–12 (Modeling → Business Selection)

## Month 3 purpose (1 paragraph)
Ο Μήνας 3 μετατρέπει το project από “EDA & baseline” σε **modeling pipeline με επιχειρησιακή αξιολόγηση**. Ο στόχος δεν είναι να μαζευτούν μοντέλα, αλλά να τεκμηριωθεί επιλογή τελικού μοντέλου με κοινό evaluation protocol και business trade-offs.

---

## Week 9 — Logistic Regression Baseline
- Baseline model + threshold sweep + business-style metrics.
- Key learning: σε extreme imbalance, το threshold καθορίζει operational viability (false alarms vs fraud leakage).

Artifacts: metrics.json, threshold sweeps, classification reports, PR/ROC, confusion matrices.

---

## Week 10 — Decision Tree
- Non-linear baseline, explainability-first.
- Key learning: εύκολα εξηγήσιμο, αλλά υπολείπεται σε generalization και μπορεί να είναι πιο “νευρικό” στο test.

Artifacts: metrics.json, sweep, confusion/plots, saved model.

---

## Week 11 — Random Forest
- Ensemble baseline, robust performance.
- Cost-based thresholding (FP=1, FN=20) ως “risk policy”.
- Fraud-first stress test δείχνει γιατί “max recall” μόνο του δεν είναι production-friendly (alerts overload).

Artifacts: metrics.json, sweep, cost-vs-threshold plots, confusion matrices, saved model.

---

## Week 12 — XGBoost + Business-Oriented Selection
- XGBoost με imbalance handling (scale_pos_weight auto).
- Threshold selected on VAL and applied to locked TEST.
- Confusion matrix μεταφρασμένη σε business terms (TP/FP/FN + alerts per 10k).
- Final decision: XGBoost ως τελικό μοντέλο (ίδιο fraud protection με RF cost-policy, λιγότερα false alarms).

Artifacts: XGB metrics/sweeps/plots, unified decision narrative, final selection sentence.

---

## Month 3 Milestone claim (what you can write)
- Implemented & evaluated multiple models (LogReg, DT, RF, XGB).
- Used consistent evaluation protocol with locked test.
- Treated threshold as business policy (cost-based + operational stress test).
- Produced scorecard + executive decision narrative.
- Selected final candidate ready for Month 4 (interpretability + deployment).
