# Week12 — Model Shortlist, Threshold Policy, Confusion-as-Business, and Final Selection (Business-Oriented)

## Ρόλος της 12ης εβδομάδας (1 παράγραφος)
Οι προηγούμενες εβδομάδες απάντησαν στο «ποια μοντέλα δουλεύουν τεχνικά». Η 12η εβδομάδα απαντά στο «ποιο μοντέλο έχει επιχειρηματική αξία». Εδώ μετατρέπουμε **metrics → decision policy (threshold) → επιχειρησιακό αφήγημα**, με κριτήρια όπως fraud leakage (FN), customer friction/ops load (FP), και σταθερότητα (VAL→TEST).

> Reproducibility / commands / artifacts: δες `docs/week12_runbook_scorecard.md`.

---

## Βήμα 3 — Shortlist υποψηφίων (2 μοντέλα max)

### Business κριτήρια shortlist
- **Υψηλό recall (προτεραιότητα):** περιορίζει fraud leakage (FN).
- **Αποδεκτό precision:** περιορίζει false alarms (FP) → καλύτερο customer experience & ops workload.
- **Σταθερότητα (VAL→TEST):** αποφεύγουμε “model risk” από overfitting / calibration drift.
- **Cost sensitivity / class weighting:** υποστήριξη επιχειρησιακής πολιτικής απόφασης.

### Evidence (TEST set, locked splits Week8)
**Operating point = threshold επιλεγμένο στο VAL και εφαρμοσμένο στο TEST (locked).**

- **Logistic Regression (business thr=0.99):** P=0.6466, R=0.7895, F1=0.7109 | TP=75, FP=41, FN=20  
  Ρόλος: baseline & governance anchor (όχι τελικό production candidate).
- **Decision Tree (thr=0.99):** P=0.5036, R=0.7263, F1=0.5948 | PR-AUC=0.4944  
  Απορρίπτεται: χαμηλότερη απόδοση + ένδειξη αστάθειας (VAL→TEST drop).
- **Random Forest (thr=0.2354, cost-opt από VAL):** P=0.7549, R=0.8105, F1=0.7817 | PR-AUC=0.8061 | TP=77 FP=25 FN=18
- **XGBoost (thr=0.0884, cost-opt από VAL):** P=0.7938, R=0.8105, F1=0.8021 | PR-AUC=0.8171 | TP=77 FP=20 FN=18

### Shortlist (απόφαση)
**Shortlist = Random Forest + XGBoost.**  
Δεν λέμε «το XGBoost είναι καλύτερο». Λέμε:
> «Τα δύο αυτά μοντέλα ικανοποιούν τις επιχειρηματικές απαιτήσεις ενός fraud detection συστήματος: υψηλό recall, επιχειρησιακά αποδεκτό precision, cost sensitivity και production-feasible συμπεριφορά.»

---

## Βήμα 4 — Threshold tuning με business στόχο (χωρίς “0.5”)

### Γιατί το threshold είναι “policy” (όχι default)
Στο fraud detection, το threshold καθορίζει **πόσα alerts ανοίγεις** (FP workload) και **πόση απάτη ξεφεύγει** (FN leakage). Άρα δεν είναι τεχνική λεπτομέρεια· είναι επιχειρησιακή πολιτική (risk management).

### RF: δύο policies (για να δείξεις “decision-making”, όχι “metric-chasing”)
**Policy A — Cost-optimal (production default):** `cost_fp=1`, `cost_fn=20` (ελαχιστοποίηση cost_per_tx)  
**Policy B — Fraud-first (stress-test):** στόχος recall>=0.90 στο VAL (μεγάλη ανοχή σε false alarms)

#### Policy comparison (TEST)
| Policy | Threshold | Precision | Recall | TP | FP | FN | Alerts/10k tx | Missed/10k tx | Cost/Tx |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RF cost-optimal | 0.2354 | 0.7549 | 0.8105 | 77 | 25 | 18 | 17.97 | 3.17 | 0.006785 |
| RF fraud-first | 0.0198 | 0.0437 | 0.8632 | 82 | 1795 | 13 | 330.77 | 2.29 | 0.036214 |

**Business reading:**  
Το fraud-first policy κερδίζει +5 TP και -5 FN, αλλά δημιουργεί +1770 FP. Αυτό είναι “operationally non-viable” για day-to-day παραγωγή (review queue explosion).

### XGB: cost-policy threshold (VAL→TEST)
Το XGBoost υποστηρίζει cost-aware training (με `scale_pos_weight=auto`) *και* cost-based threshold επιλογή στο VAL.  
Operating point στο TEST:

| Model | Threshold | Precision | Recall | TP | FP | FN | Alerts/10k tx | Missed/10k tx | Cost/Tx |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| XGB cost-opt (VAL→TEST) | 0.0884 | 0.7938 | 0.8105 | 77 | 20 | 18 | 17.09 | 3.17 | 0.006697 |

---

## Βήμα 5 — Confusion Matrix ως business εργαλείο

### Business μετάφραση
- **TP (True Positives):** “Frauds stopped” (απάτες που σταμάτησαν)
- **FN (False Negatives):** “Frauds missed / leakage” (απάτες που χάθηκαν → οικονομικό/ρυθμιστικό ρίσκο)
- **FP (False Positives):** “False alarms” (άδικα alerts → ops workload + customer friction)
- **TN:** “Clean transactions correctly passed”

### Συγκριτικό business snapshot (TEST, locked)
| Model / Policy | TP (stopped) | FN (missed) | FP (false alarms) | Τι σημαίνει |
|---|---:|---:|---:|---|
| **XGB cost-policy** | 77 | 18 | **20** | ίδια fraud protection με λιγότερα false alarms |
| **RF cost-policy** | 77 | 18 | 25 | σχεδόν ίδιο, λίγο πιο “θορυβώδες” |
| **RF fraud-first** | 82 | 13 | **1795** | review queue explosion (stress-test only) |

### Qualitative cost–benefit (χωρίς ευρώ)
Ορίζεις δύο υποθετικές ποσότητες (για αφήγημα):
- `C_fraud` = μέσο κόστος χαμένης απάτης (FN)
- `C_review` = κόστος/επιβάρυνση ενός false alarm (FP)

Και γράφεις:
- **Expected leakage cost ∝ FN × C_fraud**
- **Expected operational cost ∝ FP × C_review**
- Επιλογή threshold = εξισορρόπηση των δύο, υπό περιορισμό “review capacity”.

**Το punchline:**  
Η πολιτική που ανεβάζει λίγο το recall αλλά εκτοξεύει τα false alarms μπορεί να είναι *χειρότερη* επιχειρησιακά, ακόμη κι αν φαίνεται “καλύτερη” σε recall.

---

## Προ-Βήμα 6 — Head-to-head (RF vs XGB) με business γλώσσα

### Παρατήρηση στο TEST (cost-policy operating point)
- RF vs XGB έχουν **ίδιο TP και ίδιο FN** (77 stopped, 18 missed).
- Το XGB έχει **λιγότερα FP** (20 vs 25) και ελαφρώς καλύτερο F1.
- PR-AUC (ranking quality) είναι ελαφρώς υψηλότερο στο XGB (0.8171 vs 0.8061).

**Άρα, με locked threshold policy, το XGB προσφέρει ίδια “fraud protection” με λιγότερη τριβή/φόρτο.**

---

## Βήμα 6 — Τελική επιλογή μοντέλου (thesis-ready)

### Decision rule (report-safe)
Επιλέγεται το μοντέλο που:
1) πετυχαίνει υψηλό recall (μειώνει FN),  
2) κρατά precision σε αποδεκτό επίπεδο (μειώνει FP),  
3) είναι σταθερό στο TEST,  
4) υποστηρίζει cost-sensitive learning / policy-based thresholds,  
5) είναι production-feasible (monitoring, retraining, explainability hooks).

### Proposed final choice (με τα τωρινά evidence)
**Primary (final): XGBoost**  
**Secondary (backup / benchmark): Random Forest**

**Rationale (μία παράγραφος):**  
«Στο locked test set, και τα δύο shortlisted μοντέλα (RF, XGB) πετυχαίνουν ίδιο επίπεδο fraud protection (TP=77, FN=18) υπό cost-based policy. Το XGBoost εμφανίζει λιγότερα false alarms (FP=20 έναντι 25) και ελαφρώς καλύτερο F1/PR-AUC, άρα μειώνει customer friction και operational load χωρίς να θυσιάζει recall. Επιπλέον, υποστηρίζει cost-sensitive training μέσω `scale_pos_weight` και είναι συμβατό με production monitoring (threshold policy, drift checks, feature importance/SHAP).»

### Thesis-ready sentence (copy-paste)
«Το **XGBoost** επιλέγεται ως τελικό μοντέλο, καθώς προσφέρει τον καλύτερο συμβιβασμό μεταξύ υψηλού recall και επιχειρησιακά αποδεκτού precision, ενώ υποστηρίζει cost-sensitive learning και μπορεί να επεκταθεί εύκολα σε παραγωγικό περιβάλλον.»

> Αν σε επόμενη δοκιμή (π.χ. hyperparameter tuning ή stability check) το RF αποδειχθεί πιο σταθερό/φιλικό σε maintenance, μπορείς να κρατήσεις RF ως τελικό και να γράψεις “robustness/maintainability” ως κύριο επιχείρημα.

---

## Βήμα 7 — Τι γράφεται στο report (αυτή την εβδομάδα)

### Ενότητες που κλειδώνουν
- **Modeling Strategy:** baseline → tree → ensembles → boosting
- **Model Comparison & Evaluation:** scorecard + PR-AUC emphasis + policy thresholds
- **Handling Class Imbalance (practical impact):** class_weight / scale_pos_weight / cost policy
- **Business Trade-offs:** FP vs FN σε επιχειρησιακή γλώσσα
- **Final Model Selection:** τελική επιλογή + rationale + readiness for Month 4

### Tables/Figures που “γράφουν” στην πτυχιακή
- 1 πίνακας **Model Scorecard**:  
  `Model | Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC | TP | FP | FN | Cost/Tx`
- 1 σχήμα **PR curve** για RF vs XGB
- 1 σχήμα **Confusion Matrix** για το τελικό μοντέλο στο TEST (στο επιλεγμένο threshold)
- (προαιρετικά) **Cost vs Threshold** plot (δείχνει policy-driven επιλογή)

---

## Deliverables τέλους Week12
- 1 τελικό μοντέλο + τελικό threshold (policy)
- 1 καθαρό narrative «γιατί αυτό»
- έτοιμα tables/figures για την πτυχιακή
- γέφυρα προς Month 4: interpretability (feature importance / SHAP) + deployment (batch/scoring)
