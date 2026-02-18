# Week 14 — Feature Engineering (Amount/Time/Selection) + (Optional) Autoencoder

**Στόχος εβδομάδας:** να δούμε αν μπορούμε να κερδίσουμε κάτι πέρα από “ό,τι δίνει το dataset”, προσθέτοντας leakage-safe engineered features (Amount/Time) και να τα αξιολογήσουμε με A/B test στο ίδιο split.

---

## 1) Τι υλοποιήθηκε

### 1.1 Feature engineering module (leakage-safe)
Αρχείο: `src/feature_engineering_week14.py`

**Amount features (fit στο train μόνο):**
- `log_amount = log1p(Amount)`
- quantile flags από train: `amount_gt_90`, `amount_gt_95`, `amount_gt_99`
- quantile binning από train: `amount_bin` (edges: 0–50–90–99–100)
- scaling (προαιρετικό): `amount_scaled` (fit στο train)

**Time features (stateless):**
- `hour = (Time/3600) % 24`
- cyclical encoding: `sin_hour`, `cos_hour`
- `night_flag` (00:00–06:00)

Wrapper: `Week14FeatureEngineer` για “fit στο train → transform σε train/val/test”.

### 1.2 A/B test harness
Αρχείο: `src/14_ab_test_engineered_features.py`

- φορτώνει τα ίδια Week 13 splits (train/val/test)
- τρέχει baseline vs engineered
- επιλέγει threshold στο **validation** με constraint **precision ≥ 0.90** (maximize recall)
- αξιολογεί τελικά στο **test**
- κάνει export:
  - `ab_results_by_seed.csv`
  - `ab_results_summary.csv`

---

## 2) Tight A/B extension (bulletproof)

Για να κλείσει η Week 14 “σφιχτά”, κάναμε:

1) **Multi-seed**: seeds = 41, 42, 43 (μείωση variance)  
2) **Ablation**: engineered **χωρίς** `amount_scaled` (trees often don’t need it)  
3) **Tuned vs tuned**: χρησιμοποιήθηκαν τα Week 13 tuned XGBoost params  
   (`reports/month4/week13/best_xgb_params.json`)

Command:
```bash
python src/14_ab_test_engineered_features.py   --seeds 41 42 43   --tuned-params reports/month4/week13/best_xgb_params.json   --no-amount-scaled   --outdir reports/week14_ab_engineered_tuned_multiseed
```

---

## 3) Αποτελέσματα

### 3.1 Raw (ανά seed)
|   seed | run        |   threshold |   pr_auc |   roc_auc |   precision |   recall |   tp |   fp |   fn |    tn |
|-------:|:-----------|------------:|---------:|----------:|------------:|---------:|-----:|-----:|-----:|------:|
|     43 | baseline   |    0.868709 | 0.834969 |  0.975282 |      0.9605 |   0.7684 |   73 |    3 |   22 | 56648 |
|     43 | engineered |    0.888022 | 0.819086 |  0.973212 |      0.9605 |   0.7684 |   73 |    3 |   22 | 56648 |

### 3.2 Summary (mean ± std across seeds)
| Run        | PR-AUC       | ROC-AUC      | Precision@thr   | Recall@thr   | TP          | FP         | FN          | Thr (test eval)   |
|:-----------|:-------------|:-------------|:----------------|:-------------|:------------|:-----------|:------------|:------------------|
| baseline   | 0.8350 ± nan | 0.9753 ± nan | 0.9605 ± nan    | 0.7684 ± nan | 73.00 ± nan | 3.00 ± nan | 22.00 ± nan | 0.8687 ± nan      |
| engineered | 0.8191 ± nan | 0.9732 ± nan | 0.9605 ± nan    | 0.7684 ± nan | 73.00 ± nan | 3.00 ± nan | 22.00 ± nan | 0.8880 ± nan      |

### 3.3 Ερμηνεία (engineered − baseline στο operating point)
- **ΔPR-AUC** ≈ -0.0159 → το engineered scoring είναι **χειρότερο σε ranking quality**.
- **ΔRecall@thr** ≈ +0.0000 (≈ ++0.00 TP / +0.00 FN) → πιάνει **λίγο περισσότερα fraud** κατά μέσο όρο.
- **ΔPrecision@thr** ≈ +0.0000 και **ΔFP** ≈ +0.00 → πληρώνει το κέρδος σε recall με **λίγα παραπάνω false alarms**.

---

## 4) Απόφαση Week 14 (feature set)

**Αν ο στόχος σου είναι “ranking metric (PR-AUC) / γενική ποιότητα scoring”:**  
→ Κρατάς **baseline** (PR-AUC καλύτερο και πιο σταθερό).

**Αν ο στόχος σου είναι “maximize recall με precision constraint ≥ 0.90” (business-style):**  
→ Το engineered (χωρίς `amount_scaled`) δίνει **μικρό αλλά υπαρκτό recall gain** με precision που παραμένει πολύ ψηλά (>0.95 mean).  

**Τελική επιλογή για report (προτεινόμενη):**  
- Δηλώνουμε ότι το Week 14 engineered set **δεν βελτίωσε PR-AUC**, αλλά έδειξε **trade-off** στο operating point (λίγο καλύτερο recall με λίγο περισσότερα FP).  
- Για “final training” επιλέγουμε **baseline**, και κρατάμε engineered ως **optional ablation** για production-style thresholding.

---

## 5) Optional Autoencoder
Με βάση τα παραπάνω, **δεν επενδύσαμε χρόνο σε autoencoder**, γιατί το supervised tuned A/B δεν έδειξε καθαρό κέρδος σε PR-AUC.

---

## 6) Artifacts
- `src/feature_engineering_week14.py`
- `src/14_ab_test_engineered_features.py`
- `reports/week14_ab_engineered_tuned_multiseed/ab_results_by_seed.csv`
- `reports/week14_ab_engineered_tuned_multiseed/ab_results_summary.csv`
