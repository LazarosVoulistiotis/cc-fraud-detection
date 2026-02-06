# Week12 — Model Scorecard Runbook (Commands + Evidence)

> **Purpose:** Να μπορείς να ξανατρέξεις 100% τα experiments και να έχεις “audit trail” για Week12 scorecard.  
> Αυτό το αρχείο είναι **runbook** (ops/reproducibility). Το τελικό `week12_...md` θα έχει **executive story** + link εδώ.

---

## Table of Contents
1. [Common Ground (Fair Comparison)](#common-ground-fair-comparison)
2. [0) Locked Splits (Week8 Features)](#0-locked-splits-week8-features)
3. [1) Logistic Regression (Locked Splits + VAL threshold tuning)](#1-logistic-regression-locked-splits--val-threshold-tuning)
4. [2) Decision Tree (VAL tuning → TEST evaluation)](#2-decision-tree-val-tuning--test-evaluation)
5. [3) Random Forest (VAL tuning → TEST evaluation)](#3-random-forest-val-tuning--test-evaluation)
6. [4) XGBoost (VAL tuning → TEST evaluation)](#4-xgboost-val-tuning--test-evaluation)
7. [5) Threshold tuning policies (RF: cost-opt vs fraud-first)](#5-threshold-tuning-policies-rf-cost-opt-vs-fraud-first)
8. [Quick Checks / Useful Extractors](#quick-checks--useful-extractors)
9. [Artifacts Checklist (what must exist)](#artifacts-checklist-what-must-exist)

---

## Common Ground (Fair Comparison)

### Canonical dataset
- **Dataset:** `data/data_interim/creditcard_features_week8.csv`
- **Locked splits:** `data/data_interim/splits_week8/{train,val,test}.csv`
- **Target column:** `Class`
- **Seed:** `42` (fixed)
- **Rule:** Threshold selection γίνεται στο **VAL**, τελική αξιολόγηση στο **TEST** (locked).

### Business threshold policy (cost-based)
- `cost_fp = 1`  (false alarm / customer friction / review workload)
- `cost_fn = 20` (missed fraud / fraud leakage)

> Στόχος: να συγκρίνουμε όλα τα μοντέλα **με ίδια δεδομένα** και **ίδια policy**.

---

## 0) Locked splits (Week8 features)

### Create splits
```bash
python src/08_2_make_splits.py   --data data/data_interim/creditcard_features_week8.csv   --outdir data/data_interim/splits_week8   --target Class   --test-size 0.20   --val-size 0.10   --seed 42   --drop-duplicates
```

### Sanity check fraud rates (train/val/test)
```bash
python - <<'PY'
import pandas as pd
paths=[
  'data/data_interim/splits_week8/train.csv',
  'data/data_interim/splits_week8/val.csv',
  'data/data_interim/splits_week8/test.csv'
]
for p in paths:
    df=pd.read_csv(p)
    print(p, df.shape, 'fraud_rate=', df['Class'].mean())
PY
```

---

## 1) Logistic Regression (Locked Splits + VAL threshold tuning)

### Train on TRAIN, tune threshold on VAL, evaluate on TEST
```bash
python src/09_1_logreg_from_splits.py   --train-csv data/data_interim/splits_week8/train.csv   --val-csv data/data_interim/splits_week8/val.csv   --test-csv data/data_interim/splits_week8/test.csv   --target Class   --outdir reports/week9_logreg_week8   --figdir reports/figures/week9_logreg_week8   --model-path models/logreg_week8.joblib   --scaler standard   --class-weight balanced   --threshold 0.50   --optimize cost   --n-thresholds 101   --cost-fp 1   --cost-fn 20
```

### Extract key TEST results (default + business threshold)
```bash
python - <<'PY'
import json
m=json.load(open('reports/week9_logreg_week8/metrics.json','r',encoding='utf-8'))
print("VAL-selected business threshold:", m["threshold_selected_on_val"])

d=m["test_eval_default"]
b=m["test_eval_business"]

print("
TEST @ default thr:", d["threshold"])
print("P,R,F1:", d["precision"], d["recall"], d["f1"])
print("Confusion:", d["confusion"])

print("
TEST @ business thr:", b["threshold"])
print("P,R,F1:", b["precision"], b["recall"], b["f1"])
print("Confusion:", b["confusion"])
PY
```

### Observed results (from your run)
- **VAL-selected business threshold:** `0.99`
- **TEST @ thr=0.50:** P=0.0541 R=0.8737 F1=0.1019 | TN=55200 FP=1451 FN=12 TP=83
- **TEST @ thr=0.99:** P=0.6466 R=0.7895 F1=0.7109 | TN=56610 FP=41   FN=20 TP=75

---

## 2) Decision Tree (VAL tuning → TEST evaluation)

### 2A) VAL run (choose threshold on VAL)
```bash
python src/10_decision_tree.py   --input-train data/data_interim/splits_week8/train.csv   --input-test  data/data_interim/splits_week8/val.csv   --target-column Class   --outdir reports/week10_dt_week8_val   --figdir reports/figures/week10_dt_week8_val   --model-path models/dt_week8.joblib   --max-depth 12   --class-weight balanced   --random-state 42   --threshold 0.50 --optimize cost --n-thresholds 101   --cost-fp 1 --cost-fn 20
```

### Read chosen threshold from VAL
```bash
python - <<'PY'
import json
m=json.load(open('reports/week10_dt_week8_val/metrics.json','r',encoding='utf-8'))
print("DT VAL best_threshold =", m["best_threshold"])
print("DT VAL best_threshold_metrics =", m["best_threshold_metrics"])
PY
```

### Observed VAL results (from your run)
- `best_threshold = 0.99`
- `best_threshold_metrics (VAL)`:
  - P=0.6212 R=0.8723 F1=0.7257 | TP=41 FP=25 TN=28301 FN=6 | cost_per_tx=0.00511

### 2B) TEST run (evaluate at thr=0.99 — IMPORTANT: put the number, no placeholders)
```bash
python src/10_decision_tree.py   --input-train data/data_interim/splits_week8/train.csv   --input-test  data/data_interim/splits_week8/test.csv   --target-column Class   --outdir reports/week10_dt_week8_test   --figdir reports/figures/week10_dt_week8_test   --model-path models/dt_week8.joblib   --max-depth 12   --class-weight balanced   --random-state 42   --threshold 0.99 --optimize none --n-thresholds 101   --cost-fp 1 --cost-fn 20
```

### Observed TEST results (from your run @ thr=0.99)
- P=0.5036 R=0.7263 F1=0.5948 ROC-AUC=0.8678 PR-AUC=0.4944

### Get TP/FP/FN for thr from threshold_sweep.csv (closest match)
```bash
python - <<'PY'
import pandas as pd
thr=0.99
df=pd.read_csv('reports/week10_dt_week8_test/threshold_sweep.csv')
row=df.iloc[(df['threshold']-thr).abs().argmin()]
print(row.to_dict())
PY
```

---

## 3) Random Forest (VAL tuning → TEST evaluation)

### 3A) VAL run (choose threshold on VAL)
```bash
python src/11_random_forest.py   --input-train data/data_interim/splits_week8/train.csv   --input-test  data/data_interim/splits_week8/val.csv   --target-column Class   --outdir reports/week11_rf_week8_val   --figdir reports/figures/week11_rf_week8_val   --model-path models/rf_week8.joblib   --n-estimators 300 --max-depth 12 --min-samples-leaf 3   --class-weight balanced   --random-state 42   --threshold 0.50 --optimize cost --n-thresholds 101   --cost-fp 1 --cost-fn 20
```

### Read chosen threshold from VAL
```bash
python - <<'PY'
import json
m=json.load(open('reports/week11_rf_week8_val/metrics.json','r',encoding='utf-8'))
print("RF VAL best_threshold =", m["best_threshold"])
print("RF VAL best_threshold_metrics =", m["best_threshold_metrics"])
PY
```

### Observed VAL results (from your run)
- `best_threshold = 0.2354`
- `best_threshold_metrics (VAL)`:
  - P=0.7368 R=0.8936 F1=0.8077 | TP=42 FP=15 TN=28311 FN=5 | cost_per_tx=0.00405

### 3B) TEST run (evaluate at thr=0.2354)
```bash
python src/11_random_forest.py   --input-train data/data_interim/splits_week8/train.csv   --input-test  data/data_interim/splits_week8/test.csv   --target-column Class   --outdir reports/week11_rf_week8_test   --figdir reports/figures/week11_rf_week8_test   --model-path models/rf_week8.joblib   --n-estimators 300 --max-depth 12 --min-samples-leaf 3   --class-weight balanced   --random-state 42   --threshold 0.2354 --optimize none --n-thresholds 101   --cost-fp 1 --cost-fn 20
```

### Observed TEST results (from your run @ thr≈0.24)
- P=0.7549 R=0.8105 F1=0.7817 ROC-AUC=0.9719 PR-AUC=0.8061

### Get TP/FP/FN from threshold_sweep.csv (closest match)
```bash
python - <<'PY'
import pandas as pd
thr=0.2354
df=pd.read_csv('reports/week11_rf_week8_test/threshold_sweep.csv')
row=df.iloc[(df['threshold']-thr).abs().argmin()]
print(row.to_dict())
PY
```

---

## 4) XGBoost (VAL tuning → TEST evaluation)

> Notes:
> - Χρησιμοποιούμε `eval_metric=aucpr` για να δώσουμε βάρος στο PR-AUC (σωστό για imbalanced).
> - Χρησιμοποιούμε `scale_pos_weight=auto` ώστε να ενσωματώσουμε class imbalance με τρόπο “cost-aware”.
> - Threshold tuning γίνεται στο VAL, και μετά εφαρμόζεται locked στο TEST.

### 4A) VAL run (choose threshold on VAL via cost policy)
```bash
python src/12_xgboost.py   --input-train data/data_interim/splits_week8/train.csv   --input-test  data/data_interim/splits_week8/val.csv   --target-column Class   --outdir reports/week12_xgb_week8_val   --figdir reports/figures/week12_xgb_week8_val   --model-path models/xgb_week8.joblib   --n-estimators 400 --max-depth 6 --learning-rate 0.05   --subsample 0.8 --colsample-bytree 0.8   --scale-pos-weight auto   --threshold 0.50 --optimize cost --n-thresholds 101   --cost-fp 1 --cost-fn 20
```

### Read chosen threshold from VAL
```bash
python - <<'PY'
import json
m=json.load(open('reports/week12_xgb_week8_val/metrics.json','r',encoding='utf-8'))
print("XGB VAL best_threshold =", m["best_threshold"])
print("XGB VAL best_threshold_metrics =", m["best_threshold_metrics"])
PY
```

### Observed VAL results (from your run)
- Default thr=0.50: P=0.9535 R=0.8723 F1=0.9111 ROC-AUC=0.9957 PR-AUC=0.8966
- `best_threshold (cost) = 0.0884`
- `best_threshold_metrics (VAL)`:
  - P=0.7636 R=0.8936 F1=0.8235 | TP=42 FP=13 TN=28313 FN=5 | cost_per_tx=0.00398

### 4B) TEST run (evaluate at VAL-selected thr=0.0884)
```bash
python src/12_xgboost.py   --input-train data/data_interim/splits_week8/train.csv   --input-test  data/data_interim/splits_week8/test.csv   --target-column Class   --outdir reports/week12_xgb_week8_test   --figdir reports/figures/week12_xgb_week8_test   --model-path models/xgb_week8.joblib   --n-estimators 400 --max-depth 6 --learning-rate 0.05   --subsample 0.8 --colsample-bytree 0.8   --scale-pos-weight auto   --threshold 0.0884 --optimize none --n-thresholds 101   --cost-fp 1 --cost-fn 20
```

### Observed TEST results (from your run @ thr=0.0884)
- P=0.7938 R=0.8105 F1=0.8021 ROC-AUC=0.9699 PR-AUC=0.8171
- TP=77 FP=20 FN=18 | cost_per_tx=0.00670

### Extract TP/FP/FN row (closest match from sweep)
```bash
python - <<'PY'
import pandas as pd
thr=0.0884
df=pd.read_csv('reports/week12_xgb_week8_test/threshold_sweep.csv')
row=df.iloc[(df['threshold']-thr).abs().argmin()]
print("XGB TEST @thr=", float(row.threshold))
print(row[['precision','recall','f1','tp','fp','fn','cost_per_tx']].to_dict())
PY
```

---

## 5) Threshold tuning policies (RF: cost-opt vs fraud-first)

> Στόχος: να δείξουμε ότι το threshold είναι **policy** (risk management), όχι “0.5”.

### 5A) Derive thresholds from VAL sweep (two policies)
Παράγουμε 2 operating points από το **VAL** (`reports/week11_rf_week8_val/threshold_sweep.csv`):
- **Policy A — Cost-optimal (production default):** ελαχιστοποιεί `cost_per_tx` με `cost_fp=1`, `cost_fn=20`
- **Policy B — Fraud-first / High-recall (stress-test):** threshold που δίνει `recall >= 0.90` στο VAL (best precision υπό αυτόν τον περιορισμό)

```bash
python - <<'PY'
import pandas as pd

path = 'reports/week11_rf_week8_val/threshold_sweep.csv'
df = pd.read_csv(path)

target_recall = 0.90
delta = 0.02

def nearest_row(thr):
    return df.iloc[(df['threshold']-thr).abs().argmin()]

row_cost = df.sort_values('cost_per_tx', ascending=True).iloc[0]

cand = df[df['recall'] >= target_recall].copy()
if len(cand) == 0:
    row_r = df.iloc[(df['recall']-target_recall).abs().argmin()]
    mode_note = f'No thr reaches recall>= {target_recall:.2f}. Using closest.'
else:
    row_r = cand.sort_values('precision', ascending=False).iloc[0]
    mode_note = f'Best precision under recall>= {target_recall:.2f}.'

def stability(row):
    t = float(row['threshold'])
    left  = nearest_row(max(0.01, t - delta))
    right = nearest_row(min(0.99, t + delta))
    return {
        "thr": t,
        "P": float(row['precision']), "R": float(row['recall']),
        "P_left": float(left['precision']), "R_left": float(left['recall']),
        "P_right": float(right['precision']), "R_right": float(right['recall']),
        "FP": int(row['fp']), "FN": int(row['fn']), "TP": int(row['tp']),
        "FP_left": int(left['fp']), "FN_left": int(left['fn']),
        "FP_right": int(right['fp']), "FN_right": int(right['fn']),
    }

print("=== RF / VAL threshold policies ===")
print("\nPolicy A (cost-optimal):")
print(row_cost[['threshold','precision','recall','f1','cost_per_tx','tp','fp','fn']].to_dict())
print("Stability ±0.02:", stability(row_cost))

print("\nPolicy B (recall-target):", mode_note)
print(row_r[['threshold','precision','recall','f1','cost_per_tx','tp','fp','fn']].to_dict())
print("Stability ±0.02:", stability(row_r))
PY
```

### Observed VAL results (from your run)
- **Policy A (cost-opt):** `thr=0.2354` | P=0.7368 R=0.8936 | TP=42 FP=15 FN=5 | cost/tx=0.00405  
  Stability ±0.02: μικρές αλλαγές ⇒ **σταθερό**.
- **Policy B (recall≥0.90 on VAL):** `thr=0.0198` | P=0.0467 R=0.9149 | TP=43 FP=877 FN=4 | cost/tx=0.03373  
  Stability ±0.02: FP αλλάζει έντονα ⇒ **νευρικό / operational pain**.

### 5B) Evaluate Policy B on TEST (same model, VAL-selected threshold)
```bash
python src/11_random_forest.py   --input-train data/data_interim/splits_week8/train.csv   --input-test  data/data_interim/splits_week8/test.csv   --target-column Class   --outdir reports/week11_rf_week8_test_recall90   --figdir reports/figures/week11_rf_week8_test_recall90   --model-path models/rf_week8.joblib   --n-estimators 300 --max-depth 12 --min-samples-leaf 3   --class-weight balanced   --random-state 42   --threshold 0.0198 --optimize none --n-thresholds 101   --cost-fp 1 --cost-fn 20
```

---

## Quick Checks / Useful Extractors

### List artifacts for a run (example: XGB TEST)
```bash
ls reports/week12_xgb_week8_test
ls reports/figures/week12_xgb_week8_test
```

### Extract a single model’s “business-ready” summary from metrics.json (XGB TEST example)
```bash
python - <<'PY'
import json
m=json.load(open('reports/week12_xgb_week8_test/metrics.json','r',encoding='utf-8'))
print({
  "model": m["model"],
  "threshold": m["threshold_default"],
  "precision": m["metrics"]["precision"],
  "recall": m["metrics"]["recall"],
  "f1": m["metrics"]["f1"],
  "roc_auc": m["metrics"]["roc_auc"],
  "pr_auc": m["metrics"]["pr_auc"],
})
PY
```

---

## Artifacts Checklist (what must exist)

### LogReg (week9_logreg_week8)
- `reports/week9_logreg_week8/metrics.json`
- `reports/week9_logreg_week8/threshold_sweep_val.csv`
- `reports/week9_logreg_week8/threshold_sweep_test.csv`
- `reports/week9_logreg_week8/classification_report_default.txt`
- `reports/week9_logreg_week8/classification_report_business.txt`
- figures under `reports/figures/week9_logreg_week8/`

### Decision Tree (week10_dt_week8_val + week10_dt_week8_test)
- `reports/week10_dt_week8_val/metrics.json`
- `reports/week10_dt_week8_test/metrics.json`
- `reports/week10_dt_week8_test/threshold_sweep.csv`
- figures under `reports/figures/week10_dt_week8_*`

### Random Forest (week11_rf_week8_val + week11_rf_week8_test + recall90)
- `reports/week11_rf_week8_val/metrics.json`
- `reports/week11_rf_week8_test/metrics.json`
- `reports/week11_rf_week8_test/threshold_sweep.csv`
- `reports/week11_rf_week8_test_recall90/metrics.json`
- figures under `reports/figures/week11_rf_week8_*`

### XGBoost (week12_xgb_week8_val + week12_xgb_week8_test)
- `reports/week12_xgb_week8_val/metrics.json`
- `reports/week12_xgb_week8_val/threshold_sweep.csv`
- `reports/week12_xgb_week8_test/metrics.json`
- `reports/week12_xgb_week8_test/threshold_sweep.csv`
- `reports/week12_xgb_week8_test/feature_importances.csv`
- figures under `reports/figures/week12_xgb_week8_*`

---
