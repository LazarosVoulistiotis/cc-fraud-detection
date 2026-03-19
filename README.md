# 💳 Credit Card Fraud Detection System

A production-inspired machine learning project for detecting fraudulent credit card transactions under extreme class imbalance.

This repository documents the full lifecycle of the project: data understanding, modelling, business-driven threshold selection, explainability, and a working FastAPI inference service for deployment-style demonstration.

---

## 🚀 Project Highlights

- Built and evaluated multiple fraud detection models, including Logistic Regression, Decision Tree, Random Forest, and XGBoost
- Selected **XGBoost** as the final champion model based on locked test-set performance and business suitability
- Treated the classification threshold as an **operational policy**, not a fixed default
- Finalised a **precision-constrained threshold policy** to reduce false positives while maintaining strong fraud capture
- Added **SHAP** and **LIME** explainability for global and local model interpretation
- Implemented a working **FastAPI** service with:
  - `GET /health`
  - `GET /metadata`
  - `POST /predict`
  - `POST /predict_by_id`
- Structured the repository to support both **academic reporting** and **portfolio presentation**

---

## 📌 Problem Statement

Credit card fraud detection is a high-impact binary classification problem with severe class imbalance. In the public benchmark dataset used here, fraud represents only a tiny fraction of all transactions, which makes raw accuracy misleading.

The real challenge is not to maximise accuracy, but to balance:

- **Recall** — catching as many fraudulent transactions as possible
- **Precision** — limiting false alarms that create customer friction and analyst overhead
- **Operational cost** — recognising that missed fraud is usually much more expensive than an unnecessary review

This project approaches fraud detection as a **business-aware ML system**, not just a leaderboard exercise.

---

## 🧠 Final Model Snapshot

### Champion Model
- **Model:** XGBoost Classifier
- **Frozen artifact:** `models/xgb_final.joblib`
- **Serving threshold policy:** `precision_constraint_p80`
- **Threshold:** `0.1279`

### Final Locked Test Performance

| Model | Threshold Policy | Selected On | Threshold | Precision | Recall | F1 | F2 | MCC | ROC-AUC | PR-AUC | TP | FP | FN |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | cost-optimal | validation | 0.2354 | 0.7549 | 0.8105 | 0.7817 | 0.7987 | 0.7815 | 0.9719 | 0.8061 | 77 | 25 | 18 |
| **XGBoost (Champion)** | **precision ≥ 0.80** | **validation** | **0.1279** | **0.8280** | **0.8105** | **0.8191** | **0.8140** | **0.8189** | **0.9699** | **0.8171** | **77** | **16** | **18** |

### Business Interpretation

The final XGBoost model catches **77 frauds**, misses **18**, and keeps false positives low at **16** on the locked test set. This makes it more operationally attractive than alternatives that achieve similar recall but generate more false alarms.

---

## 🗂️ Dataset

This project uses the well-known Kaggle Credit Card Fraud Detection dataset.

### Raw Dataset Characteristics
- **Rows:** 284,807 transactions
- **Fraud cases:** 492
- **Fraud rate:** ~0.17%
- **Features:** `Time`, `V1`–`V28`, `Amount`, `Class`

### Notes
- `V1`–`V28` are anonymised PCA-style features
- `Time` is measured in seconds from the first recorded transaction
- `Amount` is the transaction value
- `Class = 1` indicates fraud
- The raw dataset is stored locally under `data/data_raw/` and is **not committed** to the repository

---

## ⚙️ Serving Design

The deployed inference contract uses a frozen serving schema.

### Input to `POST /predict`
The API accepts **raw canonical input**:
- `Time`
- `V1` to `V28`
- `Amount`

### Engineered Inside the API
The API derives the following features at serving time:
- `Hour`
- `hour_sin`
- `hour_cos`
- `Amount_log1p`

### Demo Endpoint
The `POST /predict_by_id` endpoint accepts a frozen `row_id` and reconstructs the raw payload from:

- `data/data_interim/splits_week8/test_with_row_id.csv`

This is intended for demonstration and report evidence, not for real production use.

---

## 🔍 Explainability

Model explainability is a core part of the system.

### SHAP
Used for:
- global feature importance
- beeswarm analysis
- dependence plots
- local case studies

### LIME
Used for:
- local explanation of individual predictions
- complementary model-agnostic interpretation
- case-level reporting for TP / TN / borderline examples

This helps make the final model more transparent and more suitable for fraud/risk use cases where interpretability matters.

---

## 🌐 API Endpoints

### `GET /health`
Simple liveness check.

### `GET /metadata`
Returns:
- model version
- threshold policy
- threshold used
- raw input features
- engineered features
- final model feature order

### `POST /predict`
Scores a raw transaction payload and returns:
- fraud probability
- predicted label
- threshold used

### `POST /predict_by_id`
Scores a frozen demo row and returns:
- fraud probability
- predicted label
- `row_id`
- `true_label`

---

## 🛠️ Tech Stack

- **Language:** Python
- **ML / Data:** pandas, numpy, scikit-learn, XGBoost
- **Explainability:** SHAP, LIME
- **API:** FastAPI, Pydantic, Uvicorn
- **Serialization:** joblib
- **Visualisation:** matplotlib
- **Environment:** VS Code, Jupyter, Git, GitHub

---

## ▶️ Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/LazarosVoulistiotis/cc-fraud-detection.git
cd cc-fraud-detection
```

### 2. Create and activate a virtual environment

#### Windows PowerShell
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### Git Bash
```bash
python -m venv .venv
source .venv/Scripts/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the API

```bash
uvicorn src.api.main:app --reload
```

### 5. Open the docs

```text
http://127.0.0.1:8000/docs
```

---

## 📁 Repository Structure

```text
src/                          # training, tuning, explainability, API code
src/api/                      # FastAPI inference service
models/                       # saved model artifacts
configs/                      # frozen threshold + feature schema
data/                         # raw/interim/working datasets
reports/                      # monthly reports, snippets, figures, evidence
README.md                     # project overview
README_deployment.md          # deployment-focused usage guide
```

---

## 📚 Key Artifacts

- `models/xgb_final.joblib` — frozen final champion model
- `configs/threshold.json` — final threshold policy
- `configs/feature_schema.json` — frozen serving schema
- `reports/month5/model_card.md` — model card
- `README_deployment.md` — local deployment guide

---

## ✅ Current Status

**Completed**
- Data exploration and preprocessing
- Baseline modelling
- Business-aware model selection
- Threshold optimisation
- SHAP explainability
- LIME explainability
- Frozen model artifact and serving schema
- Working FastAPI inference API

**Next steps**
- Dockerisation
- optional Streamlit/demo UI
- deployment hardening
- monitoring and future MLOps extensions

---

## 🎯 What This Project Demonstrates

This project showcases hands-on experience in:

- applied machine learning under class imbalance
- model evaluation beyond accuracy
- threshold optimisation for business goals
- explainable AI for risk-sensitive use cases
- turning an ML model into a deployable inference service
- structuring a project for both thesis documentation and portfolio presentation

---

## 👤 Author

**Lazaros Voulistiotis**  
Final-year BSc Computer Science student  
Aspiring Machine Learning Engineer

---

## 📄 License / Dataset Notice

This repository contains the code, report artifacts, and deployment skeleton for the project. The original raw dataset is not redistributed in this repository. Please obtain the Kaggle dataset separately and place it under `data/data_raw/` if you want to reproduce the full pipeline.
