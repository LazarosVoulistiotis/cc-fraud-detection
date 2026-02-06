# 💳 Credit Card Fraud Detection (Business-Oriented ML)

![Python](https://img.shields.io/badge/python-3.13-blue)
![ML](https://img.shields.io/badge/ml-sklearn%20%7C%20xgboost-brightgreen)
![Status](https://img.shields.io/badge/status-active-yellow)

A production-inspired **credit card fraud detection** project built with a **business-first evaluation mindset**:
metrics → decisions → operational impact.

This repo is designed like a “risk / fraud analytics” deliverable: not just models, but **threshold policies**, **cost trade-offs**, and **auditable artifacts**.

---

## 🧠 Business framing (why this matters)

Fraud detection is a rare-event classification problem where **accuracy is misleading**.
What matters is the trade-off between:

- **Recall (Fraud)** → how many frauds we stop (reduce fraud leakage)
- **Precision (Fraud)** → how many legitimate customers we disturb / how much analyst workload we generate (false alarms)

This project treats the classification threshold as an **operational policy**, not as “0.5 by default”.

---

## 📊 Dataset

- Kaggle Credit Card Fraud Detection (MLG-ULB style dataset)
- 284,807 transactions
- 492 frauds (~0.17% fraud rate)
- Features: `V1..V28` (PCA), `Time`, `Amount`
- Target: `Class` (0 = legitimate, 1 = fraud)

> The raw dataset is kept locally under `data/data_raw/` and is **not committed** to Git.

---

## ✅ Current Snapshot (Month 3: Modeling + Business Selection)

Evaluation uses:
- **Locked train/val/test splits** (stratified)
- **Threshold selected on validation** and applied to test
- Metrics: Precision/Recall/F1 + ROC-AUC + **PR-AUC**
- Cost-based policy example: `cost_fp=1`, `cost_fn=20` (fraud miss is 20× worse than a false alarm)

### Shortlist (business-ready): Random Forest vs XGBoost (locked test)

| Model | Threshold policy | Threshold | Precision (Fraud) | Recall (Fraud) | F1 | ROC-AUC | PR-AUC | TP | FP | FN |
|------|-------------------|----------:|------------------:|---------------:|----:|--------:|-------:|---:|---:|---:|
| Random Forest | cost-optimal (val-selected) | 0.2354 | 0.7549 | 0.8105 | 0.7817 | 0.9719 | 0.8061 | 77 | 25 | 18 |
| XGBoost | cost-optimal (val-selected) | 0.0884 | 0.7938 | 0.8105 | 0.8021 | 0.9699 | 0.8171 | 77 | 20 | 18 |

**Business interpretation:**
- Both models stop the same number of frauds (TP=77) and miss the same (FN=18),
- XGBoost produces fewer false alarms (FP=20 vs 25) → **lower operational cost + lower customer friction**.

### Why “Recall-first” can break operations (example)

A recall-target policy on RF (threshold ≈ 0.0198) achieved:
- TP=82, FN=13 (slightly better fraud capture)
- but **FP=1795** → massive analyst workload + customer friction

This is why the system is framed around **policy thresholds**, not only “maximize recall”.

---

## 🧰 Tech Stack

- Python 3.13
- pandas, numpy, scipy
- scikit-learn (LogReg, DT, RF)
- xgboost
- matplotlib

Artifacts saved per run:
- `metrics.json`
- `threshold_sweep*.csv`
- `classification_report*.txt`
- plots: ROC, PR, cost-vs-threshold, confusion matrices

---

## ⚡ Reproducible runs (Quickstart)

### 1) Setup
```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt
```

### 2) Put dataset locally
```bash
data/data_raw/creditcard.csv
```

### 3) Create stratified splits (train/val/test)
```bash
python src/08_2_make_splits.py \
  --data data/data_raw/creditcard.csv \
  --outdir data/data_interim \
  --target Class \
  --test-size 0.20 \
  --val-size 0.10 \
  --seed 42 \
  --drop-duplicates
```

### 4) Train + evaluate models (examples)

**Logistic Regression (baseline + threshold policy):**
```bash
python src/09_1_logreg_from_splits.py \
  --train-csv data/data_interim/splits_week8/train.csv \
  --val-csv   data/data_interim/splits_week8/val.csv \
  --test-csv  data/data_interim/splits_week8/test.csv \
  --target Class \
  --outdir reports/week9_logreg_week8 \
  --figdir reports/figures/week9_logreg_week8 \
  --model-path models/logreg_week8.joblib \
  --scaler standard \
  --class-weight balanced \
  --threshold 0.50 \
  --optimize cost \
  --n-thresholds 101 \
  --cost-fp 1 \
  --cost-fn 20
```

**Decision Tree:**
```bash
python src/10_decision_tree.py \
  --input-train data/data_interim/splits_week8/train.csv \
  --input-test  data/data_interim/splits_week8/val.csv \
  --target-column Class \
  --outdir reports/week10_dt_week8_val \
  --figdir reports/figures/week10_dt_week8_val \
  --model-path models/dt_week8.joblib \
  --max-depth 12 \
  --class-weight balanced \
  --threshold 0.50 \
  --optimize cost \
  --n-thresholds 101 \
  --cost-fp 1 \
  --cost-fn 20
```

**Random Forest:**
```bash
python src/11_random_forest.py \
  --input-train data/data_interim/splits_week8/train.csv \
  --input-test  data/data_interim/splits_week8/val.csv \
  --target-column Class \
  --outdir reports/week11_rf_week8_val \
  --figdir reports/figures/week11_rf_week8_val \
  --model-path models/rf_week8.joblib \
  --n-estimators 300 --max-depth 12 --min-samples-leaf 3 \
  --class-weight balanced \
  --threshold 0.50 \
  --optimize cost \
  --n-thresholds 101 \
  --cost-fp 1 \
  --cost-fn 20
```

**XGBoost (Week 12):**
```bash
python src/12_xgboost.py \
  --input-train data/data_interim/splits_week8/train.csv \
  --input-test  data/data_interim/splits_week8/val.csv \
  --target-column Class \
  --outdir reports/week12_xgb_week8_val \
  --figdir reports/figures/week12_xgb_week8_val \
  --model-path models/xgb_week8.joblib \
  --n-estimators 400 --max-depth 6 --learning-rate 0.05 \
  --subsample 0.8 --colsample-bytree 0.8 \
  --scale-pos-weight auto \
  --threshold 0.50 \
  --optimize cost \
  --n-thresholds 101 \
  --cost-fp 1 \
  --cost-fn 20
```

---

## 📁 Repository structure (report-friendly)

```bash
reports/
├── month1/                 # intro + dataset overview + early notes
├── month2/                 # EDA + data quality + scaling + imbalance + features
├── month3/                 # modeling + scorecard + business selection (Weeks 9–12)
├── figures/                # report figures (ROC/PR/CM/threshold policy plots)
├── report_snippets/        # report-ready text/snippets (copy–paste)
└── README.md               # explains the reports structure
```

---

## 🧾 Thesis / Report integration

Week 12 is the “business decision” week:
- unified scorecard
- conscious elimination of weaker approaches
- shortlist (RF vs XGB)
- threshold tuning as risk policy
- confusion matrix translated to operational impact
- final selection narrative for the report

Report-ready text lives under:
- `reports/month3/12_week12_business_model_selection.md`
- `reports/month3/month3_milestone_weeks9-12.md`
- `reports/report_snippets/`

---

## 🚀 Roadmap (Month 4)

- Interpretability (feature importance + SHAP-style explanations)
- Deployment-ready pipeline (preprocessing + model bundle)
- REST API endpoint + lightweight dashboard
- Optional Docker packaging

---

## 👤 Author

**Lazaros Voulistiotis**  
BSc Computer Science (Final Year) — Thesis Project  
Aspiring Machine Learning Engineer
