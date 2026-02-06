# 04 — Problem Challenges & Metrics

## 1) Class Imbalance (~0.17%)

Το dataset είναι **εξαιρετικά ανισόρροπο**: ~0.17% των συναλλαγών είναι απάτες (fraud) και ~99.83% νόμιμες.
- Αυτό σημαίνει ότι ένα “χαζό” μοντέλο που προβλέπει πάντα 0 (legit) θα πετυχαίνει ~99.83% **accuracy**, αλλά θα **χάνει όλες** τις απάτες.
- Άρα δεν μας ενδιαφέρει το accuracy — χρειαζόμαστε **metrics που εστιάζουν στην θετική κλάση** (fraud).

### Πρακτικές επιπτώσεις
- **Stratified split**: για να διατηρούμε το ίδιο fraud rate σε train/test, ώστε να είναι αντιπροσωπευτικό (π.χ. `train_test_split(..., stratify=y)` ή `StratifiedKFold`).
- **Κατάλληλα metrics**: **PR-AUC** (Precision-Recall AUC), **Recall**, **Precision**, **F1**, **Confusion Matrix** 
- **Sampling (αργότερα, προσεκτικά)**:
  - **Under-sampling** των 0 (legit) για ταχύτητα/ισορροπία.
  - **Over-sampling** ή **SMOTE** για την 1 (fraud) — *μόνο στο training set και πάντα μέσα στα folds* για αποφυγή data leakage.
- **Calibration** (προαιρετικά): Platt/Isotonic για καλύτερες πιθανότητες → πιο σταθερό **thresholding**.

> Note: Σε ανισόρροπα δεδομένα, το **baseline PR-AUC ≈ prevalence**. Εδώ ~0.17% → baseline PR-AUC ≈ **0.0017**. Αν το μοντέλο δίνει PR-AUC πολύ πάνω από αυτό, όντως μαθαίνει.

---

## 2) Precision vs Recall — Trade-off

Θέλουμε **ψηλό Recall** (να εντοπίζουμε όσο το δυνατόν περισσότερες απάτες) **χωρίς να εκτοξεύεται το κόστος** από πολλά False Positives.

- **Recall (TPR)**: από τις πραγματικές απάτες, πόσες πιάσαμε;
  \[
  \text{Recall} = \frac{TP}{TP + FN}
  \]
- **Precision (PPV)**: από όσες χαρακτηρίσαμε ως απάτες, πόσες ήταν όντως;
  \[
  \text{Precision} = \frac{TP}{TP + FP}
  \]
- **F1** (ισορροπεί Precision & Recall):
  \[
  F1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
  \]

Στο fraud, **Recall** συχνά είναι προτεραιότητα (μην “ξεφύγουν” απάτες), αλλά μεγάλη αύξηση Recall συνήθως ρίχνει Precision (περισσότεροι ψευδείς συναγερμοί).

---

## 3) Γιατί PR Curve (όχι μόνο ROC)

- **ROC-AUC** είναι χρήσιμο, αλλά σε ακραία imbalance μπορεί να δείχνει υπερβολικά καλό score, επειδή ο άξονας FPR επηρεάζεται από το τεράστιο πλήθος TN.
- **PR Curve** εστιάζει στη θετική κλάση (fraud) και απαντά απευθείας στο: “αν στείλω X alerts, πόσα θα είναι πραγματικές απάτες;”.

> Στόχος: **PR-AUC > baseline** (0.0017) και βελτίωση έναντι baseline μοντέλου (π.χ. Logistic Regression).

---

## 4) Confusion Matrix & Thresholding

Η **confusion matrix** στατιστικοποιεί τα αποτελέσματα σε ένα επιλεγμένο **threshold** (π.χ. 0.5, 0.2, tuned…):
- **TP**: σωστές ανιχνεύσεις απάτης
- **FP**: ψευδείς συναγερμοί (κόστος διερεύνησης/ενόχλησης πελατών)
- **FN**: χαμένες απάτες (οικονομική ζημιά)
- **TN**: σωστές νόμιμες

### Cost-sensitive thresholding (ιδέα)
Επιλέγουμε threshold με βάση **αναμενόμενο κόστος**:

- \( C_{\text{FP}} \): κόστος διερεύνησης/ενόχλησης ανά ψευδή συναγερμό  
- \( C_{\text{FN}} \): αναμενόμενη ζημιά από χαμένη απάτη  
- Για κάθε threshold, υπολόγισε FN και FP στο validation set και ελαχιστοποίησε:
  \[
  \text{Expected Cost} = C_{\text{FP}} \cdot FP + C_{\text{FN}} \cdot FN
  \]
- Εναλλακτικά, επίλεξε threshold ώστε **Recall ≥ στόχος** (π.χ. 0.90) με αποδεκτή **Precision** (π.χ. ≥ 0.10).

> Στο project μας (αρχικός στόχος): **Recall ≥ 0.90** με **Precision ≥ 0.10**, και **PR-AUC > baseline**.

---

## 5) Τι μετράμε στην αξιολόγηση (M1 baseline)

- **PR-AUC** (κύριο metric για imbalance)
- **ROC-AUC** (δευτερεύον, για πληρότητα)
- **Confusion Matrix** σε 2 thresholds:
  1) **Default 0.5**
  2) **Tuned** (cost-sensitive ή Recall-targeted)
- **Precision/Recall/F1** στα παραπάνω thresholds
- (προαιρετικά) **Recall@FPR=x** (π.χ. FPR=1%) ή **Precision@top-k** (π.χ. top 0.1% υψηλότερων scores)

**Outputs θα αποθηκεύουμε σε `reports/figures/`:**

---

## 6) Operational (μελλοντικές προεκτάσεις)

### Explainability
- **Feature importances** (π.χ. tree-based) και **Permutation importance**.
- **SHAP** για τοπικές/παγκόσμιες εξηγήσεις, χρήσιμο σε audit & debugging.

### Latency
- **Batch scoring** (π.χ. κάθε ώρα) vs **Real-time** (sub-second).  
- Επιλογή μοντέλων/υποδομής ώστε να πετυχαίνουμε SLAs (π.χ. p95 < 100ms).

### Monitoring
- **Data drift / concept drift** (π.χ. PSI, στατιστικά σύγκρισης κατανομών).
- **Model performance over time**: Recall, Precision, PR-AUC σε πρόσφατα παράθυρα (rolling).
- **Alerting** & κριτήρια retraining.

---

## 7) Good Practices / Προστασία από leakage

- Sampling (SMOTE/oversample) **μόνο στο training** και **μέσα στα folds**.
- **Stratified** CV/split παντού.
- **Scaling/encoding** *μέσα* στο pipeline για να μην “κοιτάει” το test.
- Log/αποθήκευση όλων των metrics + figures με **σταθερά seeds** για αναπαραγωγιμότητα.

---

## 8) Ελάχιστα Deliverables για το Milestone M1 (τέλος Μήνα 1)

- Οριστικοποιημένες **προκλήσεις & metrics** (αυτό το έγγραφο).
- **EDA plots** και **correlation heatmaps** (ήδη σε `reports/figures/`).
- **Baseline αξιολόγηση** (PR/ROC curves, confusion matrices) με CSV και εικόνες

> Επόμενο βήμα (Μήνας 2): baseline μοντέλο (π.χ. Logistic Regression + scaling), threshold tuning με στόχο **Recall ≥ 0.90** και αναφορά αποτελεσμάτων στο README.
