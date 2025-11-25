# Εβδομάδα 10 — Decision Tree vs Logistic Regression

---

## 1. Εισαγωγή

Στην Εβδομάδα 9 χτίστηκε ένα ισχυρό baseline με **Logistic Regression** για το πρόβλημα
ανίχνευσης απάτης σε συναλλαγές πιστωτικών καρτών.

Στόχος της Εβδομάδας 10 είναι:

- να εισάγει ένα πρώτο **μη-γραμμικό, interpretable** μοντέλο (**Decision Tree**),
- να το βελτιστοποιήσει ως προς το **recall στο fraud class (Class = 1)**,
- και να το συγκρίνει καθαρά, σε **business επίπεδο**, με το Logistic Regression baseline.

Κύριο επιχειρηματικό ερώτημα:

> Μπορεί ένα Decision Tree να προσφέρει συγκρίσιμη απόδοση με το Logistic Regression,  
> ενώ παράλληλα δίνει διαφανή rules που μπορούν να χρησιμοποιηθούν από business / risk
> teams;

---

## 2. Περιγραφή Μοντέλου — Decision Tree

Ένα **Decision Tree** είναι ένα δενδροειδές, rule-based μοντέλο που χωρίζει επαναληπτικά
τον χώρο χαρακτηριστικών σε περιοχές με σχετικά ομοιογενείς κλάσεις.

### 2.1 Βασικά χαρακτηριστικά

- Κάθε εσωτερικός κόμβος ελέγχει μια συνθήκη τύπου  
  `feature_j <= threshold`.
- Τα φύλλα (*leaves*) αντιστοιχούν σε τελικές προβλέψεις κλάσης  
  (non-fraud ή fraud).
- Κάθε διαδρομή από το root σε ένα leaf μπορεί να διαβαστεί σαν ανθρώπινο
  **if–then rule**, κάτι κρίσιμο για κανονιστικά περιβάλλοντα και εσωτερικούς ελέγχους.

### 2.2 Πλεονεκτήματα για fraud detection

- Μοντελοποιεί **μη-γραμμικές σχέσεις** μεταξύ χαρακτηριστικών  
  (π.χ. συνδυασμοί V-μεταβλητών).
- Τα rules του μπορούν να γίνουν βάση για **χειροκίνητους business κανόνες**  
  (π.χ. triggers για manual review, ειδικούς limits ανά segment).

---

## 3. Dataset & Πειραματική Ρύθμιση

- **Dataset:** Kaggle *Credit Card Fraud Detection*  
  (284.807 συναλλαγές, 492 fraud συνολικά).
- **Target:** `Class` (0 = non-fraud, 1 = fraud).

### 3.1 Χωρισμός δεδομένων

- Train / Validation / Test όπως ορίστηκαν στην Εβδομάδα 9.
- Δημιουγούνται και χρησιμοποιούνται τα ίδια αρχεία:

  - `data/data_interim/train.csv`
  - `data/data_interim/val.csv`
  - `data/data_interim/test.csv`

### 3.2 Pre-processing

- Ίδια features με το Logistic Regression baseline.
- Χρήση του scaled ποσού `Amount_scaled`, όπως ορίστηκε στην Εβδομάδα 6.

Στο test set χρησιμοποιήθηκαν περίπου **56.962 συναλλαγές** με fraud rate ~**0,17%**
(98 fraud / ~56.9K συνολικά).

---

## 3.1. Baseline Decision Tree

### 3.1.1 Ρυθμίσεις μοντέλου

```python
DecisionTreeClassifier(
    criterion="gini",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight=None,
    random_state=42,
)
```
### 3.1.2 Μετρικές (test set, threshold = 0.50)

- **Accuracy:** 0.9990  
- **Precision (fraud):** 0.7263  
- **Recall (fraud):** 0.7041  
- **F1 (fraud):** 0.7150  
- **ROC-AUC:** 0.8518  
- **PR-AUC (AP):** 0.5119  

**Confusion matrix** στο test set (`dt_baseline_confusion.png`):

- **TN = 56.838** non-fraud σωστά ως non-fraud  
- **FP = 26** non-fraud λανθασμένα ως fraud  
- **FN = 29** fraud που χάθηκαν  
- **TP = 69** fraud που εντοπίστηκαν  

Οπτικοποίηση στο GitHub:

![Confusion Matrix — DT baseline](../figures/week10/dt_baseline_confusion.png)

---

### 3.1.3 Business σχόλιο για το baseline DT

Παρά την εντυπωσιακή συνολική ακρίβεια (**99,9%**), το baseline δέντρο:

- χάνει **29 από τα 98 fraud** (~30% των πραγματικών απάτων),
- διατηρεί σχετικά υψηλή **precision ~72%**, αφού κάνει λίγα false alarms.

Για ένα περιβάλλον όπου το κόστος χαμένης απάτης είναι πολύ υψηλό, αυτό το baseline:

- κρίνεται **ανεπαρκές ως κύριο παραγωγικό μοντέλο**,  
- αλλά είναι πολύ χρήσιμο ως **σημείο εκκίνησης** και ως μελλοντικός **base learner για ensembles**.

---

## 4. Tuning Decision Tree & Αντιμετώπιση Imbalance

Στόχος του tuning είναι να **ενισχύσει το recall στο Class = 1**, ακόμη και αν χρειαστεί
να θυσιάσουμε μέρος της precision, με την προϋπόθεση ότι το συνολικό κόστος
(συνδυασμός False Positives και False Negatives) παραμένει επιχειρηματικά αποδεκτό.

---

### 4.1. Hyperparameters που εξετάστηκαν

Το Grid Search πραγματοποιήθηκε πάνω στο ακόλουθο πλέγμα παραμέτρων:

- `max_depth`: `[3, 5, 7, 9, None]`
- `min_samples_split`: `[2, 10, 50]`
- `min_samples_leaf`: `[1, 5, 10]`
- `class_weight`: `[None, "balanced"]`

---

### 4.2. Διαδικασία Grid Search

- Χρησιμοποιήθηκε `StratifiedKFold` με **3 folds**.
- Metric βελτιστοποίησης:
  - **Recall στο fraud class (Class = 1)** μέσω  
    `make_scorer(recall_score, pos_label=1)`.
- Αποθήκευση πλήρους αποτελέσματος στο:
  - `reports/week10_decision_tree_vs_logreg/dt_gridsearch_results.csv`.

#### Πίνακας 1 — Καλύτερα Hyperparameters Decision Tree (Grid Search)

Με βάση το `GridSearchCV`, τα hyperparameters που επελέγησαν για το τελικό μοντέλο ήταν:

- `max_depth = 3`
- `min_samples_split = 50`
- `min_samples_leaf = 1`
- `class_weight = "balanced"`

Η επιλογή μικρού `max_depth` και αυξημένου `min_samples_split`:

- περιορίζει την πολύπλοκη δομή του δέντρου,
- μειώνει τον κίνδυνο **overfitting**,

ενώ το `class_weight="balanced"` «δίνει βάρος» στα σπάνια fraud samples.  
(Η ακριβής τιμή του μέσου cross-val recall βρίσκεται στο `dt_gridsearch_results.csv`
και αναλύεται στο τεχνικό παράρτημα.)

---

### 4.3. Τελικό Tuned Decision Tree

Με τα παραπάνω hyperparameters εκπαιδεύτηκε εκ νέου μοντέλο στο **train + validation**
σύνολο και αξιολογήθηκε στο **test set**.

Αποτελέσματα αποθηκεύονται στα:

- **Metrics JSON:**  
  `reports/week10_decision_tree_vs_logreg/dt_tuned_metrics.json`
- **Confusion matrix:**  
  `reports/figures/week10/dt_tuned_confusion.png`
- **ROC curve:**  
  `reports/figures/week10/dt_roc_curve_tuned.png`
- **PR curve:**  
  `reports/figures/week10/dt_pr_curve_tuned.png`

Οπτικά στο GitHub:

![ROC Curve — DT tuned](../figures/week10/dt_roc_curve_tuned.png)  
![PR Curve — DT tuned](../figures/week10/dt_pr_curve_tuned.png)  
![Confusion Matrix — DT tuned](../figures/week10/dt_tuned_confusion.png)

#### 4.3.1 Βασικές μετρικές (test set, threshold = 0.50)

- **Accuracy:** 0.9557  
- **Precision (fraud):** 0.0334  
- **Recall (fraud):** 0.8878  
- **F1 (fraud):** 0.0645  
- **ROC-AUC:** 0.9456  

Confusion matrix στο test set:

- **TN = 54.350** non-fraud σωστά ως non-fraud  
- **FP = 2.514** non-fraud που σημαίνονται λανθασμένα ως fraud  
- **FN = 11** fraud που χάνονται  
- **TP = 87** fraud που εντοπίζονται  

#### 4.3.2 Business σχόλιο για το tuned DT

Σε σχέση με το baseline DT:

- Το **recall** ανεβαίνει από ~70% σε ~89%  
  → σημαντικά λιγότερα χαμένα fraud.
- Η **precision** πέφτει έντονα (από ~72% σε ~3,3%)  
  → τα περισσότερα alerts του δέντρου είναι πλέον **false positives**.
- Η συνολική **accuracy** μειώνεται (95,6%) λόγω των επιπλέον false alarms, αλλά
  παραμένει υψηλή, δεδομένης της ακραίας ανισορροπίας.

Το tuned Decision Tree λειτουργεί ως **επιθετικό μοντέλο ανίχνευσης**:

- πιάνει σχεδόν **9 στις 10 απάτες**,  
- αλλά «χτυπάει» αρκετές νόμιμες συναλλαγές που θα χρειαστούν downstream έλεγχο
  (manual review, SMS verification κ.λπ.).

---

## 5. Οπτικοποίηση Απλού Δέντρου (Interpretable Tree)

Για να υπάρχει ένα πλήρως **interpretable παράδειγμα** για managers, εκπαιδεύτηκε
ένα ρηχό δέντρο με ρυθμίσεις:

- `max_depth = 3`
- `min_samples_leaf = 50`
- `class_weight = "balanced"`

Το δέντρο οπτικοποιήθηκε και αποθηκεύτηκε ως:

- `reports/figures/week10/dt_simple_tree.png`

![Shallow Decision Tree (max_depth=3)](../figures/week10/dt_simple_tree.png)

### 5.1. Ενδεικτικά interpretable rules από το shallow tree

Από το σχήμα παρατηρούνται μερικά χαρακτηριστικά patterns:

**Κανόνας 1 — Περιοχή υψηλού ρίσκου με ακραίες τιμές V14 & V12**

- Αν το **V14** είναι σημαντικά αρνητικό (π.χ. `V14 ≤ -1.8`)  
- και ταυτόχρονα το **V12** είναι επίσης πολύ αρνητικό (π.χ. `V12 ≤ -1.9`),  
- τότε το δέντρο οδηγείται σε φύλλα με κυρίαρχη κλάση **fraud**.

**Κανόνας 2 — Σχετικά ασφαλής ζώνη με «ήπιο» V4**

- Όταν `V14 > -1.8` και το χαρακτηριστικό **V4** μένει σε σχετικά χαμηλές τιμές,  
- το δέντρο ταξινομεί την πλειονότητα των συναλλαγών ως **non-fraud**,  
- κάτι που αντιστοιχεί σε πιο «ήσυχο» προφίλ συναλλαγών.

**Κανόνας 3 — Συνδυασμός V20 και V12 αυξάνει τον κίνδυνο**

- Σε branches όπου το **V4** είναι υψηλό, αλλά ο συνδυασμός  
  `V20` πολύ αρνητικό και `V12` μέτρια αρνητικό,  
- παρατηρούνται φύλλα με αυξημένο ποσοστό **fraud**, άρα πρόκειται για ζώνες
  όπου το μοντέλο «ανάβει κόκκινο» και αξίζει manual review.

Οι κανόνες αυτοί είναι **ενδεικτικοί** και δεν περιγράφουν όλο το Decision Tree,
αλλά δίνουν σε risk & compliance ομάδες μια **διαφανή εικόνα** του πώς ένα ML
μοντέλο βλέπει τις «ζώνες κινδύνου» στο feature space.

---

## 6. Σύγκριση Decision Tree vs Logistic Regression

Χρησιμοποιώντας τα JSON metrics της Εβδομάδας 9 για το **Logistic Regression**
και τα metrics της Εβδομάδας 10 για το **tuned Decision Tree**, δημιουργήθηκε:

- ο πίνακας `reports/week10_decision_tree_vs_logreg/model_comparison.csv`
- το bar chart `reports/figures/week10/logreg_vs_dt_metrics.png`

![Logistic Regression vs Decision Tree (tuned)](../figures/week10/logreg_vs_dt_metrics.png)

### 6.1. Πίνακας σύγκρισης (test set, threshold = 0.50)

| Model                 | Accuracy | Precision_fraud | Recall_fraud | F1_fraud | ROC_AUC |
|-----------------------|----------|-----------------|-------------:|---------:|--------:|
| Logistic Regression   | NaN   | 0.0610          | 0.9184       | 0.1144   | 0.9721  |
| Decision Tree (tuned) | 0.9557   | 0.0334          | 0.8878       | 0.0645   | 0.9456  |

*(Οι τιμές για το Decision Tree προέρχονται από `model_comparison.csv`· για το
Logistic Regression οι μετρικές είναι αυτές της Week 9 baseline.)*

### 6.2. Ποιοτικά συμπεράσματα

Το **Logistic Regression** παραμένει το πιο ισορροπημένο μοντέλο:

- Υψηλότερο **ROC-AUC (~0.97)**.
- Καλύτερο **F1 στο fraud class** (0.114 vs 0.064).
- Λίγο υψηλότερο **recall** (91,8% vs 88,8%) και σχεδόν **διπλάσια precision**
  (6,1% vs 3,3%).

Το **tuned Decision Tree**:

- πλησιάζει την απόδοση σε recall και ROC-AUC,
- αλλά το κάνει με **μεγαλύτερο αριθμό false positives** και χαμηλότερη precision.

Με άλλα λόγια, από καθαρά **predictive performance** σκοπιά,
το Logistic Regression baseline παραμένει **strong winner**, ενώ το Decision Tree
προσθέτει κυρίως **ερμηνευσιμότητα** και λειτουργεί ως stepping stone προς
ενισχυτικά ensemble μοντέλα.

---

## 7. Threshold Analysis (Business “What-if” Scenarios)

Για το **tuned Decision Tree** δοκιμάστηκαν διαφορετικά classification thresholds
στην πιθανότητα `P(Class=1)`:

- Thresholds: `0.3`, `0.5`, `0.7`

Για κάθε threshold μετρήθηκαν:

- Precision (fraud)
- Recall (fraud)
- F1 (fraud)

Τα αναλυτικά αποτελέσματα αποθηκεύονται στο:

- `reports/week10_decision_tree_vs_logreg/dt_threshold_scenarios.csv`

### 7.1. Πίνακας thresholds (tuned DT, test set)

| Threshold | Precision_fraud | Recall_fraud | F1_fraud |
|----------:|----------------:|-------------:|---------:|
| 0.3       | 0.0334          | 0.8878       | 0.0645   |
| 0.5       | 0.0334          | 0.8878       | 0.0645   |
| 0.7       | 0.0470          | 0.8878       | 0.0892   |

**Παρατήρηση:** σε αυτό το μοντέλο οι περισσότερες προβλέψεις είναι πολύ κοντά στο 0
ή στο 1, άρα η αλλαγή threshold **δεν επηρεάζει το recall** μέσα σε αυτό το εύρος·
αλλάζει μόνο ελαφρά η precision και το F1.

### 7.2. Business ερμηνεία

- Για thresholds **0.3–0.5** το μοντέλο είναι πρακτικά ίδιο:  
  πιάνει σχεδόν **89% των fraud**, αλλά μόνο ~**3,3%** των alerts είναι πραγματικές απάτες.
- Στο **0.7** γινόμαστε ελάχιστα πιο «συντηρητικοί» στα alerts:
  - κερδίζουμε λίγη precision (από 3,3% σε ~4,7%),
  - χωρίς να χάνουμε επιπλέον fraud στο συγκεκριμένο test set.

Συνολικά, το tuned Decision Tree παραμένει **επιθετικό** ανεξάρτητα από το
threshold στην περιοχή 0.3–0.7: προσφέρει πολύ υψηλό recall αλλά με **σημαντικό
φόρτο false positives** προς τις ομάδες χειροκίνητου ελέγχου.

---

## 8. Business Ανάλυση Αποτελεσμάτων

Σε επίπεδο αποφάσεων:

- Με **Logistic Regression**, στο default threshold 0.50:
  - χάνονται **8 από τα 98 fraud** (~8,2%),
  - ο ρυθμός false positives είναι περίπου **2,4%** πάνω στο non-fraud class.
- Με το **tuned Decision Tree**, στο ίδιο threshold:
  - χάνονται **11 από τα 98 fraud** (~11,2%),
  - ο ρυθμός false positives ανεβαίνει περίπου στο **4,4%**.

Άρα, στο συγκεκριμένο dataset και ρυθμίσεις:

- Το **Logistic Regression** είναι ταυτόχρονα **πιο ακριβές και πιο “οικονομικό”**
  (λιγότερα χαμένα fraud *και* λιγότερα false alarms).
- Το **Decision Tree** προσφέρει επιπλέον **οπτική ερμηνευσιμότητα**:
  οι κανόνες του μπορούν να χρησιμοποιηθούν:
  - είτε ως manual κανόνες (π.χ. «αν V14 και V12 είναι πολύ αρνητικά → manual review»),
  - είτε ως input σε πιο σύνθετα συστήματα κανόνων.

Ένα πιθανό business setup που προκύπτει:

- Χρησιμοποιούμε το **Logistic Regression** ως κύριο μοντέλο *risk scoring*.
- Χρησιμοποιούμε το **Decision Tree**:
  - για να παράγει interpretable rules που τροφοδοτούν policies,
  - ή ως ένα από τα μοντέλα σε ensemble (π.χ. Random Forest, XGBoost) που θα
    εξεταστεί στην Εβδομάδα 11.

---

## 9. Συμπεράσματα Εβδομάδας 10

Η Εβδομάδα 10 ολοκλήρωσε τον πρώτο κύκλο **non-linear μοντελοποίησης**:

1. Χτίστηκε και αναλύθηκε ένα **Decision Tree baseline**, το οποίο έδειξε ότι
   ένα απλό, βαθύ δέντρο μπορεί να έχει υψηλή συνολική ακρίβεια αλλά όχι ιδανική
   συμπεριφορά στο σπάνιο fraud class.
2. Έγινε **Grid Search tuning** με έμφαση στο recall (Class = 1), οδηγώντας σε ένα
   πιο ρηχό, τακτοποιημένο δέντρο, με βελτιωμένο recall σε σχέση με το baseline,
   αλλά ισχυρή πτώση της precision.
3. Πραγματοποιήθηκε **σύγκριση με Logistic Regression**· το αποτέλεσμα είναι ότι
   το LogReg παραμένει το κύριο production-ready μοντέλο, ενώ το Decision Tree
   λειτουργεί κυρίως ως:
   - interpretable εργαλείο,
   - πηγή για business rules,
   - και δομικό στοιχείο για επόμενα ensemble μοντέλα.
4. Εκτελέστηκε **threshold & cost analysis**, ώστε να συνδεθεί η απόδοση των
   μοντέλων με επιχειρηματικά μεγέθη (false alarms vs χαμένες απάτες).

**Επόμενο βήμα (Week 11):** μετάβαση σε **ensemble μοντέλα**
(`RandomForestClassifier`, `XGBoostClassifier` κ.λπ.), όπου το Decision Tree
χρησιμοποιείται ως base learner και στοχεύουμε σε περαιτέρω βελτίωση τόσο του
recall όσο και των cost-sensitive metrics σε πραγματικά business σενάρια.
