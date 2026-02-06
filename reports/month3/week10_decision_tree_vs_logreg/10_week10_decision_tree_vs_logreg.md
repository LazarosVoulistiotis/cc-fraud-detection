# Εβδομάδα 10 — Decision Tree vs Logistic Regression

## 1. Εισαγωγή

Στο πλαίσιο της Εβδομάδας 10 εξετάζεται ένα μη-γραμμικό, interpretable μοντέλο 
(**Decision Tree Classifier**) και συγκρίνεται με το baseline μοντέλο **Logistic Regression** 
της Εβδομάδας 9 για το πρόβλημα ανίχνευσης απάτης σε συναλλαγές πιστωτικών καρτών.

Κύριο business ερώτημα:

- Μπορεί ένα Decision Tree να αυξήσει το **recall στα fraud cases (Class = 1)** χωρίς να εκτοξεύσει 
  υπερβολικά τα false positives;

---

## 2. Περιγραφή Μοντέλου — Decision Tree

Ένα Decision Tree είναι ένα **δενδροειδές, rule-based μοντέλο** που χωρίζει επαναληπτικά τον χώρο 
χαρακτηριστικών σε περιοχές με σχετικά ομοιογενείς κλάσεις.

Βασικά χαρακτηριστικά:

- Κάθε εσωτερικός κόμβος ελέγχει μια συνθήκη της μορφής `feature_j <= threshold`.
- Τα φύλλα (leaves) αντιστοιχούν σε προβλέψεις κλάσης (π.χ. non-fraud, fraud).
- Προσφέρει **υψηλή ερμηνευσιμότητα**, καθώς κάθε διαδρομή από root σε leaf μπορεί να διαβαστεί σαν 
  ένα ανθρώπινο rule (πολύ χρήσιμο για business / κανονιστικές ομάδες).

Πλεονεκτήματα για fraud detection:

- Επιτρέπει μη-γραμμικές σχέσεις μεταξύ χαρακτηριστικών.
- Τα rules μπορούν να γίνουν βάση για **χειροκίνητους business κανόνες** (π.χ. triggers για manual review).

---

## 3. Dataset & Πειραματική Ρύθμιση

- Dataset: Kaggle Credit Card Fraud Detection (284.807 συναλλαγές, 492 fraud).
- Target: `Class` (0 = non-fraud, 1 = fraud).
- Splits:
  - Train, Validation, Test όπως ορίστηκαν στην Εβδομάδα 9.
  - Χρησιμοποιούνται τα ίδια αρχεία: `data/data_interim/train.csv`, `data/data_interim/val.csv`, `data/data_interim/test.csv`.
- Προεπεξεργασία:
  - Χρήση των ίδιων features με το Logistic Regression baseline.
  - Χρήση του scaled ποσού (`Amount_scaled` ή αντίστοιχο), όπως ορίστηκε στην Εβδομάδα 6.

### 3.1. Baseline Decision Tree

Baseline ρυθμίσεις:

- `DecisionTreeClassifier(random_state=42)`
- Default hyperparameters (χωρίς tuning):
  - `max_depth=None`
  - `min_samples_split=2`
  - `min_samples_leaf=1`
  - `class_weight=None`

Μετρικές που υπολογίζονται:

- Accuracy (overall)
- Precision (Class = 1)
- Recall (Class = 1)
- F1-score (Class = 1)
- ROC-AUC
- Confusion Matrix

Τα αντίστοιχα plots:

- ROC curve: `reports/figures/week10/dt_roc_curve_baseline.png`
- Precision–Recall curve: `reports/figures/week10/dt_pr_curve_baseline.png`
- Confusion matrix: `reports/figures/week10/dt_baseline_confusion.png`

---

## 4. Tuning Decision Tree & Αντιμετώπιση Imbalance

Για να βελτιωθεί η απόδοση στο fraud class, εφαρμόστηκε **Grid Search** με στόχο την αύξηση του 
**recall για Class = 1**, διατηρώντας σε αποδεκτό επίπεδο την precision.

### 4.1. Hyperparameters που εξετάστηκαν

- `max_depth`: `[3, 5, 7, 9, None]`
- `min_samples_split`: `[2, 10, 50]`
- `min_samples_leaf`: `[1, 5, 10]`
- `class_weight`: `[None, "balanced"]`

### 4.2. Διαδικασία Grid Search

- Χρήση `StratifiedKFold` (3- ή 5-fold).
- Metric για optimization:
  - `recall` στο fraud class (Class = 1), μέσω `make_scorer(recall_score, pos_label=1)`.
- Αποθήκευση αναλυτικών αποτελεσμάτων:
  - `reports/week10_decision_tree_vs_logreg/dt_gridsearch_results.csv`

**Πίνακας 1 — Καλύτερα Hyperparameters Decision Tree (Grid Search)**

- Best params: `max_depth = TODO`, `min_samples_split = TODO`, `min_samples_leaf = TODO`, `class_weight = TODO`
- Best cross-val recall (Class = 1): `TODO`

### 4.3. Τελικό Tuned Decision Tree

Με τα βέλτιστα hyperparameters εκπαιδεύτηκε εκ νέου μοντέλο και αξιολογήθηκε στο **test set**.

**Αρχεία:**

- Metrics JSON: `reports/week10_decision_tree_vs_logreg/dt_tuned_metrics.json`
- Confusion matrix: `reports/figures/week10/dt_tuned_confusion.png`
- ROC curve: `reports/figures/week10/dt_roc_curve_tuned.png`
- PR curve: `reports/figures/week10/dt_pr_curve_tuned.png`

---

## 5. Οπτικοποίηση Απλού Δέντρου (Interpretable Tree)

Για να παρουσιαστεί ένα interpretability παράδειγμα σε business/manager κοινό, εκπαιδεύτηκε ένα
ρηχό δέντρο με:

- `max_depth = 3` (ή `4`)
- Περιορισμό στα φύλλα (π.χ. `min_samples_leaf = 50`)

Το δέντρο οπτικοποιήθηκε και αποθηκεύτηκε ως:

- `reports/figures/week10/dt_simple_tree.png`

**Παράδειγμα rule (ενδεικτικό για το report):**

> Αν `V14 < -3.5` και `V12 > 1.2`, τότε η πιθανότητα η συναλλαγή να είναι fraud αυξάνεται σημαντικά,  
> οπότε μπορεί να προταθεί manual review.

---

## 6. Σύγκριση Decision Tree vs Logistic Regression

Τα συγκριτικά αποτελέσματα συνοψίζονται στον παρακάτω πίνακα:

| Model                 | Accuracy | Precision (fraud) | Recall (fraud) | F1 (fraud) | ROC-AUC |
|-----------------------|----------|-------------------|----------------|------------|---------|
| Logistic Regression   | TODO     | TODO              | TODO           | TODO       | TODO    |
| Decision Tree (tuned) | TODO     | TODO              | TODO           | TODO       | TODO    |

Τα αντίστοιχα δεδομένα αποθηκεύονται στο:

- `reports/week10_decision_tree_vs_logreg/model_comparison.csv`

Και το comparison plot (bar chart):

- `reports/figures/week10/logreg_vs_dt_metrics.png`

---

## 7. Threshold Analysis (Business “What-if” Scenarios)

Για το tuned Decision Tree υπολογίστηκαν διαφορετικά classification thresholds στη θεωρούμενη πιθανότητα `P(Class=1)`:

- Thresholds: `0.3`, `0.5`, `0.7`

Για κάθε threshold μετρήθηκαν:

- Precision (fraud)
- Recall (fraud)
- F1 (fraud)

Τα αποτελέσματα αποθηκεύτηκαν στο:

- `reports/week10_decision_tree_vs_logreg/dt_threshold_scenarios.csv`

**Ενδεικτικός πίνακας για το report:**

| Threshold | Precision (fraud) | Recall (fraud) | F1 (fraud) |
|-----------|-------------------|----------------|------------|
| 0.3       | TODO              | TODO           | TODO       |
| 0.5       | TODO              | TODO           | TODO       |
| 0.7       | TODO              | TODO           | TODO       |

**Business ερμηνεία (θα συμπληρωθεί με νούμερα):**

- Σε χαμηλότερο threshold (`0.3`) αυξάνεται το **recall** (πιάνουμε περισσότερες απάτες) με κόστος
  περισσότερων **false positives**.
- Σε υψηλότερο threshold (`0.7`) μειώνονται τα **false positives** αλλά ρισκάρουμε να χάνουμε
  πραγματικά fraud (χαμηλότερο **recall**).

---

## 8. Business Ανάλυση Αποτελεσμάτων

Στο σημείο αυτό μπαίνει η ποιοτική ανάλυση:

Αν το Decision Tree:

- αυξάνει το **recall** σε σχέση με το Logistic Regression  
- με μικρή/ανεκτή πτώση στην **precision**

τότε, από business σκοπιά μπορεί να είναι πιο κατάλληλο για fraud detection, ειδικά αν:

- το κόστος ενός **false negative** (χαμένη απάτη) είναι πολύ μεγαλύτερο από το κόστος ενός **false positive**.

**Ενδεικτικά bullets που θα γεμίσεις με νούμερα:**

- «Με Logistic Regression χάνεται περίπου **Χ%** των fraud cases στο test set».  
- «Με το tuned Decision Tree το ποσοστό αυτό μειώνεται σε **Υ%**, εις βάρος μιας αύξησης της τάξης του **Ζ%** στα false alarms».

---

## 9. Συμπεράσματα Εβδομάδας 10

Το Decision Tree λειτουργεί ως πρώτο **non-linear** και **interpretable baseline**.

Προσφέρει:

- Καλύτερη ευαισθησία (**recall**) για fraud cases (αν αυτό επιβεβαιωθεί από τα metrics).
- Δυνατότητα εξαγωγής απλών **rules** που μπορούν να επικοινωνηθούν σε business/κανονιστικές ομάδες.

Η σύγκριση με Logistic Regression δείχνει καθαρά τα trade-offs μεταξύ:

- απλότητας + σταθερότητας (**LogReg**)
- ευαισθησίας + ερμηνευσιμότητας (**Decision Tree**).

**Επόμενο βήμα (Week 11):** μετάβαση σε ensemble μοντέλα (π.χ. `RandomForestClassifier`, `XGBoostClassifier`) για
πιθανή περαιτέρω βελτίωση της απόδοσης.
