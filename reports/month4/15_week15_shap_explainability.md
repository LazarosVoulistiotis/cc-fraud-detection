# Week 15 — SHAP Explainability (XGBoost Champion Model)

**Status:** ✅ Completed  
**Date:** 2026-02-22  
**Project:** Credit Card Fraud Detection (`cc-fraud-detection`)  

---

## 0) Executive summary
Τη Week 15 εφαρμόσαμε **SHAP (TreeExplainer)** στο τελικό μοντέλο **XGBoost** ώστε να παρέχουμε:

- **Global explainability:** ποια features οδηγούν γενικά τις προβλέψεις (beeswarm + bar plot).
- **Dependence analysis:** πώς 1–2 κορυφαία features επηρεάζουν μη-γραμμικά την πρόβλεψη (dependence plots).
- **Local explainability:** “γιατί” σε 3 χαρακτηριστικές περιπτώσεις (TP / TN / Borderline) με waterfall plots.

Όλα τα plots είναι έτοιμα για ενσωμάτωση στο τελικό report στον φάκελο `reports/figures/week15/`.

---

## 1) Στόχος εβδομάδας
Να κάνουμε το μοντέλο **υπεύθυνο και εξηγήσιμο** (audit-friendly):

- **Global drivers**: ποια χαρακτηριστικά αυξομειώνουν συστηματικά το fraud score.
- **Case studies**: εξήγηση 3 μεμονωμένων συναλλαγών με “industry-style” local explanations.

---

## 2) Πολιτική threshold (σωστή σειρά project)
Η επιλογή threshold έγινε **στο Validation (Week 12)** με cost-based policy και εφαρμόστηκε **στο locked Test** (χωρίς tuning στο test):

- **Optimize:** `cost`
- **Cost policy:** `cost_fp = 1`, `cost_fn = 20`
- **Selected threshold (VAL):** `0.0884`
- Η ίδια τιμή χρησιμοποιήθηκε στη Week 15 για να επιλεγούν οι local περιπτώσεις (TP/TN/Borderline).

---

## 3) Inputs / Artifacts που χρησιμοποιήθηκαν

### 3.1 Model
- `models/xgb_week8.joblib` (XGBoost champion)

### 3.2 Data splits (με engineered features)
Χρησιμοποιήθηκαν τα Week8 splits που **περιέχουν** τα engineered columns (π.χ. `Hour`, `hour_sin`, `hour_cos`, `Amount_log1p`):

- `data/data_interim/splits_week8/train.csv`
- `data/data_interim/splits_week8/test.csv`

> Σημείωση: Τα `data/data_interim/train.csv` & `data/data_interim/test.csv` (χωρίς engineered columns) **δεν** ταιριάζουν με το συγκεκριμένο trained model και δεν χρησιμοποιούνται.

---

## 4) Υλοποίηση

### 4.1 Script
- `src/15_shap_explainability.py`

### 4.2 SHAP setup (σωστά & οικονομικά)
- Χρήση **`shap.TreeExplainer`** (ιδανικό/γρήγορο για tree models όπως XGBoost).
- Υπολογισμός global SHAP σε **sample** για ελεγχόμενο runtime:
  - `sample_size = 10000` (τυχαίο δείγμα από test)
  - `background_size = 1000` (τυχαίο background από train)
- `seed = 42` για αναπαραγωγιμότητα (ίδιο sampling).
- Local explainability με 3 περιπτώσεις:
  - 1 **True Positive (TP)**
  - 1 **True Negative (TN)**
  - 1 **Borderline** (probability κοντά στο threshold)

### 4.3 Command που εκτελέστηκε (Git Bash)
```bash
python src/15_shap_explainability.py \
  --model-path models/xgb_week8.joblib \
  --data-train data/data_interim/splits_week8/train.csv \
  --data-test  data/data_interim/splits_week8/test.csv \
  --target-column Class \
  --figdir reports/figures/week15 \
  --outdir reports/week15_shap \
  --sample-size 10000 \
  --background-size 1000 \
  --threshold 0.0884 \
  --seed 42
```

---

## 5) Outputs (Deliverables)

### 5.1 Figures (έτοιμα για report)
**Path:** `reports/figures/week15/`

- `shap_summary_beeswarm.png`
- `shap_mean_abs_bar.png`
- `shap_dependence_V4.png`
- `shap_dependence_V14.png`
- `shap_waterfall_true_positive.png`
- `shap_waterfall_true_negative.png`
- `shap_waterfall_borderline.png`

### 5.2 Metadata
**Path:** `reports/week15_shap/`

- `shap_mean_abs.csv` (global ranking)
- `shap_cases.json` (TP/TN/borderline με threshold + probabilities)

---

## 6) Results — Global Explainability

### 6.1 Top features (mean |SHAP|)
Από `reports/week15_shap/shap_mean_abs.csv` (top-10):

| Rank | Feature | mean(|SHAP|) |
|---:|---|---:|
| 1 | V4  | 0.6754 |
| 2 | V14 | 0.5991 |
| 3 | V8  | 0.4950 |
| 4 | V12 | 0.3690 |
| 5 | V15 | 0.3279 |
| 6 | V11 | 0.3200 |
| 7 | V6  | 0.2502 |
| 8 | V3  | 0.2482 |
| 9 | V22 | 0.2306 |
| 10| V18 | 0.2193 |

**Συμπέρασμα:** Το μοντέλο στηρίζεται κυρίως στα PCA components (V-features), κάτι αναμενόμενο σε anonymised fraud dataset. Παρότι υπάρχουν engineered columns, δεν εμφανίζονται στα top-10 drivers του συγκεκριμένου run.

> Σημείωση ερμηνείας: Τα V1–V28 είναι PCA components, άρα η explainability είναι συμπεριφορική (“τι ώθησε την πρόβλεψη”) και όχι άμεσα σημασιολογική (π.χ. merchant/country).

---

## 7) Results — Dependence Analysis (Global, top-2)

### 7.1 `shap_dependence_V4.png` (color: V12)
Το V4 εμφανίζει έντονα **μη-γραμμική** επίδραση: καθώς αυξάνει, το SHAP impact τείνει να γίνεται θετικό/μεγάλο (push προς fraud). Η χρωματική κωδικοποίηση (V12) υποδεικνύει **interaction**: για ίδιες τιμές V4, διαφορετικές τιμές V12 αλλάζουν την ένταση του impact.

### 7.2 `shap_dependence_V14.png` (color: V11)
Το V14 παρουσιάζει “threshold effects” (τυπικό για boosted trees). Η χρωματική κωδικοποίηση (V11) δείχνει αλληλεπίδραση: το V11 επηρεάζει το πώς το V14 μεταφράζεται σε fraud score.

---

## 8) Results — Local Explainability (Case Studies)
Operating threshold: **0.0884** (Week 12 cost-policy).

Από `reports/week15_shap/shap_cases.json` επιλέχθηκαν:

| Case | Index | True label | Predicted prob |
|---|---:|---:|---:|
| True Positive | 18427 | 1 | 0.999998 |
| True Negative | 49260 | 0 | 0.000000083 |
| Borderline | 53293 | 0 | 0.088737 |

### 8.1 True Positive (TP)
- **Πολύ υψηλή πιθανότητα** (≈1.0) και καθαρό fraud pattern.
- Το waterfall plot δείχνει ισχυρές θετικές συνεισφορές (κυρίως V14/V12/V10/V4), που ωθούν την έξοδο του μοντέλου πολύ πάνω από το threshold.

### 8.2 True Negative (TN)
- **Πρακτικά μηδενική πιθανότητα** (≈0).
- Το waterfall plot δείχνει κυρίως αρνητικές συνεισφορές από βασικά features, κρατώντας σταθερά χαμηλό το fraud score.

### 8.3 Borderline (near threshold)
- Probability **πολύ κοντά** στο threshold (0.088737 vs 0.0884).
- Το waterfall plot δείχνει “μάχη” μεταξύ θετικών pushes προς fraud (π.χ. V14/V12/V10/V4) και αρνητικών συνεισφορών (π.χ. V28/V26/hour_sin και λοιπά), οδηγώντας σε οριακή απόφαση.

**Business interpretation:** τέτοιες borderline περιπτώσεις είναι κατάλληλες για διαδικασία **manual review** ή “soft action”, ανάλογα με κόστος/λειτουργική πολιτική.

---

## 9) Engineering notes (τι διορθώθηκε / τι μάθαμε)
- **Matplotlib Tkinter error (Windows):** λύθηκε με `matplotlib.use("Agg")` (headless backend) για αποθήκευση PNG χωρίς GUI/Tk.
- **XGBoost feature-name mismatch:** αντιμετωπίστηκε με ευθυγράμμιση input columns και robust prediction ώστε να ταιριάζουν τα expected feature names.
- **Shell mismatch:** PowerShell vs Git Bash έχουν διαφορετικό line continuation (`\` vs backtick). Το τελικό run έγινε σε Git Bash με `\`.

---

## 10) Πώς ενσωματώνεται στο τελικό report (mini checklist)
- Ενότητα: **Explainability (SHAP)**
- Εικόνες:
  - Global: `shap_summary_beeswarm.png`, `shap_mean_abs_bar.png`
  - Dependence: `shap_dependence_V4.png`, `shap_dependence_V14.png`
  - Local: `shap_waterfall_true_positive.png`, `shap_waterfall_true_negative.png`, `shap_waterfall_borderline.png`
- Κείμενο:
  - 1 παράγραφος “Global drivers” (top-10)
  - 1 παράγραφος “Dependence/Interactions”
  - 1 παράγραφος “Case studies” (TP/TN/Borderline + threshold 0.0884)

---
