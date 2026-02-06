# Week 11 — Random Forest (Ensemble Learning + Business KPIs)

## Στόχος εβδομάδας
Να βελτιώσω ουσιαστικά την ανίχνευση απάτης με **Random Forest (ensemble/bagging)** και να τεκμηριώσω το trade-off **FP vs FN** με **business KPIs** (metrics + threshold policy + expected cost).

---

## Σύνδεση με Week 9–10 (για πτυχιακή)
- **Week 9 — Logistic Regression:** explainable baseline, αλλά σε imbalanced fraud data μπορεί να γίνει business-wise προβληματικό (ιδίως με default threshold).
- **Week 10 — Decision Tree:** πιάνει non-linear patterns αλλά έχει υψηλό variance / overfitting risk.
- **Week 11 — Random Forest:** κλιμάκωση ενός tree σε πιο **σταθερό** και **αξιόπιστο** μοντέλο (variance reduction) με καλύτερο balance precision/recall.

---

## Γιατί Random Forest (business framing)
Fraud detection = πρόβλημα υψηλού κόστους σφάλματος.
- Ένα single tree είναι “ασταθές” (variance).
- Το Random Forest εκπαιδεύει πολλά trees (bagging) και κάνει aggregation → **μείωση variance** → πιο **σταθερές** αποφάσεις.
- Επιπλέον, επειδή παράγει probabilities, μπορούμε να εφαρμόσουμε **threshold policy** (π.χ. cost-based) αντί να μένουμε στο default 0.50.

---

## Δεδομένα / Splits
- Train: `data/data_interim/train.csv`
- Test:  `data/data_interim/test.csv`
- Target: `Class`
- Test fraud rate (observed): ~0.00172 (έντονο class imbalance)

---

## Scripts που χρησιμοποιήθηκαν

### 1) `src/11_random_forest.py`
- Εκπαίδευση + αξιολόγηση Random Forest
- Παράγει artifacts:
  - `metrics.json`, `config.json`
  - `classification_report.txt`
  - `feature_importances.csv`
  - `threshold_sweep.csv`
  - Figures: ROC / PR / Confusion Matrix / Precision–Recall vs Threshold / Cost vs Threshold
  - Μοντέλο: `.joblib`

### 2) `src/12_compare_models.py`
- Σύγκριση **LogReg vs DecisionTree vs RandomForest** με κοινή “decision policy”:
  - **Policy A:** threshold = 0.50 (default)
  - **Policy B:** threshold που ελαχιστοποιεί **expected cost per transaction**
- Export:
  - `reports/week11_model_comparison/model_comparison.csv`
  - sweeps ανά μοντέλο

**Σημείωση (feature schema drift):**
Τα Week 9/10 models είχαν εκπαιδευτεί με διαφορετικό feature schema σε σχέση με το τρέχον `test.csv`. Για να τρέξει δίκαια η σύγκριση, έγινε **feature alignment** στο compare script (drop unseen engineered features για το αντίστοιχο model).

---

## Run A — Baseline Random Forest (χωρίς imbalance handling)

### Command
```bash
python src/11_random_forest.py \
  --input-train data/data_interim/train.csv \
  --input-test  data/data_interim/test.csv \
  --target-column Class \
  --outdir reports/week11_rf_runA_baseline \
  --figdir reports/figures/week11/runA \
  --model-path models/rf_runA_baseline.joblib \
  --n-estimators 100 --max-depth None --min-samples-leaf 1 \
  --class-weight none \
  --threshold 0.50 --optimize none --n-thresholds 101 \
  --cost-fp 1.0 --cost-fn 20.0
```

### Results (Test, thr=0.50)
- Accuracy: **0.999561**
- Precision (fraud): **0.9506**
- Recall (fraud): **0.7857**
- F1 (fraud): **0.8603**
- ROC-AUC: **0.9577**
- PR-AUC (AP): **0.8686**

### Confusion Matrix (thr=0.50)
- TN=56860, FP=4, FN=21, TP=77

Business ανάγνωση: πολύ λίγα false alarms (FP=4) αλλά 21 frauds περνάνε (FN).

---

## Run F — Balanced + Conservative Random Forest

### Command
```bash
python src/11_random_forest.py \
  --input-train data/data_interim/train.csv \
  --input-test  data/data_interim/test.csv \
  --target-column Class \
  --outdir reports/week11_rf_runF_balanced_conservative \
  --figdir reports/figures/week11/runF \
  --model-path models/rf_runF_balanced.joblib \
  --n-estimators 300 --max-depth 12 --min-samples-leaf 3 \
  --class-weight balanced \
  --threshold 0.50 --optimize cost --n-thresholds 101 \
  --cost-fp 1.0 --cost-fn 20.0
```

### Results (Test, thr=0.50)
- Accuracy: **0.999403**
- Precision (fraud): **0.8404**
- Recall (fraud): **0.8061**
- F1 (fraud): **0.8229**
- ROC-AUC: **0.9612**
- PR-AUC (AP): **0.8455**

### Confusion Matrix (thr=0.50)
- TN=56849, FP=15, FN=19, TP=79

---

## Threshold sweep & cost-based policy (risk management)
Ορίστηκε cost ratio:
- `cost_fp = 1`
- `cost_fn = 20` (FN πολύ πιο ακριβό από FP)

**Επιλογή threshold:** ελαχιστοποίηση expected cost per transaction.

### Cost-optimal threshold (Run F)
- Best threshold: **0.1864**
- Precision: **0.6370**
- Recall: **0.8776**
- F1: **0.7382**
- Cost/tx: **0.005074**
- Confusion (thr≈0.19): TN=56815, FP=49, FN=12, TP=86

Business ανάγνωση: μειώνω FN (frauds που περνάνε) αποδεχόμενος αύξηση FP (operational friction), γιατί με cost_fn >> cost_fp το συνολικό expected cost μειώνεται.

> Σημείωση: σε production, threshold tuning γίνεται σε validation/CV και κρατάμε το test “κλειστό”.

---

## Βήμα 5 — Σύγκριση με LogReg & Decision Tree (ίδια policy)

Export: `reports/week11_model_comparison/model_comparison.csv`

### Policy A: threshold = 0.50
| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | TP | FP | FN | Cost/tx |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RandomForest | 0.961248 | 0.845487 | 0.840426 | 0.806122 | 0.822917 | 79 | 15 | 19 | 0.006934 |
| LogReg | 0.972083 | 0.718971 | 0.060976 | 0.918367 | 0.114358 | 90 | 1386 | 8 | 0.027141 |
| DecisionTree | 0.851812 | 0.511895 | 0.726316 | 0.704082 | 0.715026 | 69 | 26 | 29 | 0.010639 |

### Policy B: cost-optimal threshold (min cost/tx)
| Model | thr* | Precision | Recall | F1 | TP | FP | FN | Cost/tx |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **RandomForest** | **0.1864** | 0.637037 | 0.877551 | 0.738197 | 86 | 49 | 12 | **0.005074** |
| LogReg | 0.9704 | 0.478022 | 0.887755 | 0.621429 | 87 | 95 | 11 | 0.005530 |
| DecisionTree | 0.0100 | 0.726316 | 0.704082 | 0.715026 | 69 | 26 | 29 | 0.010639 |

**Συμπέρασμα σύγκρισης (what the examiner cares about):**
- Δεν μας νοιάζει αν το accuracy πέσει ελάχιστα. Μας νοιάζει **TP/FN/FP** και το **expected cost**.
- Με cost-based policy, το **Random Forest** δίνει το χαμηλότερο cost/tx και καλύτερο operational trade-off (λιγότερα FP από LogReg για παρόμοιο recall).

---

## Βήμα 6 — Confusion Matrix + “κόστος” ανάγνωση (report-ready)
Για το επιλεγμένο candidate (RF, thr≈0.19):
- **TP:** απάτες που σταμάτησαν (86)
- **FN:** απάτες που πέρασαν (12) → άμεση απώλεια
- **FP:** false alarms (49) → operational cost / customer friction
- **TN:** “ήσυχες” συναλλαγές (56815)

Επιλέχθηκε threshold που ελαχιστοποιεί expected cost per transaction με cost ratio FN:FP = 20:1.

---

## Feature Importances (Random Forest)
Από `feature_importances.csv` (impurity-based importances):

- Τα **Top-5** features εξηγούν ~**57.2%** της συνολικής σημασίας.
- Τα **Top-10** features εξηγούν ~**80.2%** της συνολικής σημασίας.

### Top-10 features
| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `V14` | 0.160782 |
| 2 | `V10` | 0.114788 |
| 3 | `V12` | 0.099477 |
| 4 | `V17` | 0.099458 |
| 5 | `V4` | 0.097164 |
| 6 | `V11` | 0.073418 |
| 7 | `V3` | 0.057161 |
| 8 | `V16` | 0.049352 |
| 9 | `V7` | 0.025464 |
| 10 | `V2` | 0.025088 |

**Σύντομη ερμηνεία:**
- Τα ισχυρότερα σήματα εμφανίζονται στα **V14, V10, V12, V17, V4, V11** (κλασικοί “fraud drivers” στο συγκεκριμένο dataset).
- Features χρόνου/ώρας (`Time`, `Hour`, `hour_sin`, `hour_cos`) έχουν μικρότερη σημασία σε αυτό το RF run.

> Σημείωση: οι impurity-based importances μπορεί να έχουν bias (και να μπερδεύονται σε correlated features). Σε επόμενη εβδομάδα μπορούμε να προσθέσουμε **Permutation Importance** ή/και **SHAP** για πιο αξιόπιστη ερμηνεία.

---

## Deliverables Week 11 (τι έμεινε στο repo)

### Models
- `models/rf_runA_baseline.joblib`
- `models/rf_runF_balanced.joblib`
- (for comparison) `models/logreg_baseline.joblib`, `models/dt_baseline.joblib`

### Reports (per run)
- `reports/week11_rf_runA_baseline/` (metrics.json, config.json, report, sweep, importances)
- `reports/week11_rf_runF_balanced_conservative/` (metrics.json, config.json, report, sweep, importances)

### Figures (Week 11)
- ROC / PR / Confusion matrices (default & best) / Precision-Recall vs threshold / Cost vs threshold

### Model comparison
- `reports/week11_model_comparison/model_comparison.csv`
- `reports/week11_model_comparison/*_threshold_sweep.csv`

---

## Key takeaways (1 paragraph για πτυχιακή)
Το Random Forest επιλέχθηκε ως primary candidate model επειδή παρέχει καλύτερη ισορροπία precision/recall από τα baseline μοντέλα και, μέσω cost-based thresholding, ελαχιστοποιεί το αναμενόμενο κόστος ανά συναλλαγή (FN >> FP), κάτι που ταιριάζει άμεσα με real-world fraud detection requirements.

---

## Next steps (Week 12 preview)
- Gradient Boosting / XGBoost
- πιο “σοβαρό” imbalance handling (sampling/weights strategies)
- interpretability: Permutation importance / SHAP
- deployment thinking: monitoring, drift checks, threshold policy documentation
