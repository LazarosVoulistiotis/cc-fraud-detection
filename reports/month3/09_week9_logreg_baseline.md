# Week 9 — Baseline Logistic Regression (Business-Oriented)

## 1. Στόχος
Να οριστεί ένα **αναπαραγώγιμο baseline** για ανίχνευση απάτης πιστωτικών συναλλαγών με **Logistic Regression** μέσα σε **Pipeline** (scaler→model), να αξιολογηθεί με **precision, recall, F1, ROC-AUC, PR-AUC**, να παραχθούν βασικά plots και να καταγραφεί επιχειρησιακή ανάγνωση (**threshold & cost**).

---

## 2. Δεδομένα & Ρυθμίσεις Run
- **Dataset:** `data/data_raw/creditcard.csv`  
- **Split:** stratified train/test, `test_size=0.20`, `seed=42`  
- **Pipeline:** `StandardScaler` → `LogisticRegression(max_iter=1000, solver=lbfgs, class_weight=balanced)`  
- **Default threshold:** `0.50`  
- **Threshold sweep:** `0.01..0.99` (101 σημεία), **κόστος:** `cost_fp=1.0`, `cost_fn=20.0`  
- **Run metadata:** `run_id=20251103-052430`, `git_rev=b97b0f8`, Python 3.13 / NumPy 2.1.3 / Pandas 2.3.2 / scikit-learn 1.7.1

---

## 3. Μεθοδολογία
1. **Φόρτωση & έλεγχος** ότι υπάρχει η στήλη `Class`.  
2. **Stratified split** για διατήρηση του fraud rate μεταξύ train/test.  
3. **Pipeline** ώστε ο scaler να “μαθαίνει” μόνο στο train (αποφυγή data leakage).  
4. **Εκπαίδευση** και αποθήκευση μοντέλου (`.joblib`).  
5. **Αξιολόγηση** στο test:  
   - Threshold-free: **ROC-AUC**, **PR-AUC**.  
   - Threshold-based (thr=0.50): **precision**, **recall**, **F1**, **confusion matrix**.  
6. **Threshold sweep** & **Cost per transaction** για επιχειρησιακή πολιτική.

---

## 4. Αποτελέσματα (Test set)

### 4.1 Headline metrics
| Metric | Value |
|---|---:|
| ROC-AUC | **0.9721** |
| PR-AUC (AP) | **0.7190** |
| Precision @ thr=0.50 | **0.0610** |
| Recall @ thr=0.50 | **0.9184** |
| F1 @ thr=0.50 | **0.1144** |
| Recall @ Precision ≥ 0.90 | **0.0000** |
| Precision @ Recall ≥ 0.90 | **0.1415** |

**Σχόλιο:** Το **PR-AUC=0.719** είναι ισχυρό για τόσο χαμηλό prevalence (~0.17% στο test). Το μοντέλο έχει πολύ καλή **διατάξη πιθανοτήτων**. Στο thr=0.50 πετυχαίνει **υψηλό recall** με **χαμηλή precision** — τυπικό με `class_weight=balanced` σε έντονο class imbalance.

### 4.2 Confusion matrix (thr=0.50)
- **TN = 55 478**, **FP = 1 386**, **FN = 8**, **TP = 90**  
- Accuracy: 0.9755 · Specificity: 0.9756 · Macro-avg recall: 0.9470  
- **Prevalence (test):** 98 / 56 962 ≈ **0.172%**

**Κόστος/συναλλαγή @0.50** με `cost_fp=1`, `cost_fn=20`:  
\\[
\\text{cost/tx}=\\frac{FP\\cdot 1 + FN\\cdot 20}{N}=\\frac{1386 + 8\\cdot 20}{56\\,962}\\approx \\mathbf{0.027}
\\]

---

## 5. Διαγράμματα (Figures)
- **ROC Curve:** `reports/figures/week9/week9_logreg_roc.png`  
  Καμπύλη πολύ κοντά στην επάνω-αριστερή γωνία (AUC≈0.972) → ισχυρή διαχωριστική ικανότητα· για extreme imbalance συνοδεύεται πάντα από PR.
- **Precision–Recall Curve:** `reports/figures/week9/week9_logreg_pr.png`  
  AP≈0.719, καθαρό trade-off P↔R καθώς αλλάζει το threshold· baseline precision≈prevalence.
- **Confusion Matrix (thr=0.50):** `reports/figures/week9/week9_logreg_confusion_default.png`  
  Υψηλό **recall** (≈0.918) με **precision** ≈0.061 → ~2.6% των συναλλαγών σημαίνονται για έλεγχο.
- **Precision & Recall vs Threshold:** `reports/figures/week9/week9_prec_recall_vs_threshold.png`  
  Όσο ανεβαίνει το threshold: **precision↑**, **recall↓**· η πτώση του recall είναι σχετικά ήπια μέχρι υψηλά thresholds → καλό ranking.
- **Cost vs Threshold:** `reports/figures/week9/week9_cost_vs_threshold.png`  
  Με FP=1 & FN=20, το **κόστος** πέφτει έντονα με αύξηση threshold και ελαχιστοποιείται ψηλά (≈0.9–1.0), γιατί οι FP καταρρέουν γρηγορότερα από όσο αυξάνουν οι FN.

---

## 6. Threshold & Cost Analysis (business ανάγνωση)
- Στο **thr=0.50** το λειτουργικό φορτίο (FP) είναι υψηλό· **cost/tx ≈ 0.027**.  
- Η καμπύλη **Cost vs Threshold** υποδεικνύει ότι **υψηλότερο threshold** (≈0.9+) πιθανότατα **ελαχιστοποιεί** το κόστος στο test.  
- **Παραγωγική πολιτική:** το threshold **δεν** επιλέγεται στο test. Γίνεται **tuning σε validation/CV** με στόχο **ελαχιστοποίηση κόστους** (ή SLA τύπου precision≥Χ/recall≥Ψ) και το test μένει κλειστό.

---

## 7. Ερμηνευσιμότητα (coefficients)
- Εξήχθησαν συντελεστές Logistic Regression ταξινομημένοι κατά |coef|: `reports/week9_baseline/coefficients_sorted.csv`.  
- Σημείωση: τα **V1–V28** είναι **PCA components** (μη άμεσα ερμηνεύσιμα). Το `Amount` είναι συνήθως πιο “αναγνώσιμο”.

---

## 8. Παραδοτέα (Artifacts)
- **Model:** `models/logreg_baseline.joblib`  
- **Metrics:** `reports/week9_baseline/metrics.json`  
- **Classification report:** `reports/week9_baseline/classification_report.txt`  
- **Coefficients:** `reports/week9_baseline/coefficients_sorted.csv`  
- **Threshold sweep:** `reports/week9_baseline/threshold_sweep.csv`  
- **Figures:**  
  - `reports/figures/week9/week9_logreg_roc.png`  
  - `reports/figures/week9/week9_logreg_pr.png`  
  - `reports/figures/week9/week9_logreg_confusion_default.png`  
  - `reports/figures/week9/week9_prec_recall_vs_threshold.png`  
  - `reports/figures/week9/week9_cost_vs_threshold.png`  
- **Config (CLI args):** `reports/week9_baseline/config.json`  
- **Run log:** `reports/week9_baseline/run_20251103-052430.log`

---

## 9. Αναπαραγωγιμότητα
**CLI (baseline run):**
```bash
python src/09_logreg_baseline.py \
  --data data/data_raw/creditcard.csv \
  --outdir reports/week9_baseline \
  --figdir reports/figures/week9 \
  --model-path models/logreg_baseline.joblib \
  --test-size 0.20 --seed 42 \
  --scaler standard --class-weight balanced --threshold 0.50 \
  --optimize none --n-thresholds 101 --cost-fp 1.0 --cost-fn 20.0
```
**Αποτύπωμα:** `run_id=20251103-052430`, `git_rev=b97b0f8`.

---

## 10. Συμπέρασμα & Επόμενα Βήματα
Το baseline Logistic Regression εμφανίζει **εξαιρετική διαχωριστική ικανότητα** (ROC-AUC≈0.972, PR-AUC≈0.719). Στο thr=0.50 επιτυγχάνει **πολύ υψηλό recall** με **χαμηλή precision** και **cost/tx≈0.027** (FP=1, FN=20).  
Για παραγωγική χρήση προτείνεται:
1. **Threshold tuning** σε validation/CV με **optimize=cost** και αναφορά **default vs cost-optimal** (confusion + cost).  
2. Προαιρετικά: **Calibration** πιθανοτήτων, **RobustScaler** A/B, και **cross-validated PR-AUC (mean±std)** στο train πριν τη σύγκριση με άλλα baselines (π.χ. Decision Tree).

---

**Πρόταση commit message:**
```
Week9: Logistic Regression baseline — metrics, plots, cost analysis, and business-ready report
```
