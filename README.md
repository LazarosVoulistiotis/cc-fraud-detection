# 💳 Credit Card Fraud Detection System

A production-inspired machine learning system for detecting fraudulent credit card transactions under extreme class imbalance.

This repository documents the full lifecycle of the project: data understanding, modelling, business-aware threshold selection, explainability, API serving, operational hardening, and Docker-based packaging for reproducible local deployment.

---

## 🚀 Project Highlights

- Built and evaluated multiple fraud detection models, including Logistic Regression, Decision Tree, Random Forest, and XGBoost
- Selected **XGBoost** as the final champion model based on locked test-set performance and business suitability
- Treated the classification threshold as an **operational policy**, not a default probability cutoff
- Finalised a **precision-constrained threshold policy** to reduce false positives while maintaining strong fraud capture
- Added **SHAP** and **LIME** explainability for global and local model interpretation
- Implemented a working **FastAPI** inference service with:
  - `GET /health`
  - `GET /metadata`
  - `POST /predict`
  - `POST /predict_by_id`
- Hardened the serving layer with:
  - reproducible preprocessing
  - strict feature alignment
  - structured JSON logging
  - centralized error handling
  - automated tests with `pytest`
  - GitHub Actions CI on push / pull request
- Added **Docker packaging** so the API can run in a consistent containerized environment
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

The deployed inference contract uses a frozen serving schema and configuration-driven serving policy.

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

### Reproducible Preprocessing

The serving pipeline:

- validates the frozen schema
- computes engineered features deterministically
- aligns features to the exact frozen model feature order
- rejects missing, unexpected, or invalid inputs

This reduces the risk of inference-time feature mismatch and improves deployment reliability.

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

Returns serving metadata such as:

- model version
- model artifact path
- git commit
- train date
- threshold policy and version
- threshold used
- schema version
- raw input features
- engineered features
- final model feature order

### `POST /predict`

Scores a raw transaction payload and returns:

- fraud probability
- predicted label
- threshold used
- threshold policy
- model version

### `POST /predict_by_id`

Scores a frozen demo row and returns:

- fraud probability
- predicted label
- `row_id`
- `true_label`

---

## 📈 Observability and Operational Hardening

The API was hardened to behave less like a fragile local demo and more like a production-inspired service.

### Structured Logging

The service emits structured JSON logs for:

- request completion
- prediction scoring
- validation errors
- HTTP errors
- unexpected exceptions
- metadata requests

Typical logged fields include:

- `request_id`
- `method`
- `path`
- `status_code`
- `latency_ms`
- `prediction_latency_ms`
- `fraud_probability`
- `predicted_label`
- `threshold_used`
- `threshold_policy`
- `model_version`

### Error Handling

The FastAPI app includes centralized error handling for:

- invalid request payloads
- explicit HTTP errors
- unexpected server-side exceptions

### Configuration-Driven Serving

The serving layer relies on:

- `configs/threshold.json`
- `configs/feature_schema.json`
- `configs/model_metadata.json`

This means thresholding, schema alignment, and model provenance are managed through configs rather than hardcoded values.

---

## 🧪 Testing and CI

### Test Coverage

The project includes automated tests for:

- health endpoint behavior
- prediction endpoint behavior
- metadata endpoint behavior
- preprocessing and feature engineering logic
- golden-path demo prediction flow

### Current Test Structure

```text
tests/
├── test_health.py
├── test_predict.py
├── test_preprocess.py
└── test_golden.py
```

### Local Result

```bash
pytest -q
22 passed in 3.46s
```

### Continuous Integration

A GitHub Actions workflow automatically runs the test suite on:

- every push to `main`
- every pull request to `main`

The CI was designed to be artifact-safe by mocking model-dependent API test paths where appropriate.

---

## 🛠️ Tech Stack

- **Language:** Python
- **ML / Data:** pandas, numpy, scikit-learn, XGBoost
- **Explainability:** SHAP, LIME
- **API:** FastAPI, Pydantic, Uvicorn
- **Serialization:** joblib
- **Logging:** structlog
- **Testing:** pytest
- **CI:** GitHub Actions
- **Containerization:** Docker
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

### 4. Run the API locally

```bash
uvicorn src.api.main:app --reload
```

### 5. Open the docs

```text
http://127.0.0.1:8000/docs
```

### 6. Run the test suite

```bash
pytest -q
```

---

## 🐳 Run via Docker

The project includes a Docker-based local deployment path so the API can run in a reproducible environment without depending on the host Python setup.

### Build the image

```bash
docker build -t fraud-api .
```

### Run the container

```bash
docker run --rm -p 8000:8000 fraud-api
```

### Access the API

Once the container starts successfully, the service will be available at:

```text
http://127.0.0.1:8000
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

### Optional shortcut commands with Makefile

If `make` is installed on your system, you can use these shortcuts:

```bash
make run
make test
make docker
make docker-run-quick
```

#### What each target does

- `make run` → starts the API locally with auto-reload
- `make test` → runs the test suite with `pytest -q`
- `make docker` → builds the Docker image tagged as `fraud-api`
- `make docker-run-quick` → runs the built image and exposes port `8000`

> Note: On some Windows Git Bash setups, `make` may not be installed by default. In that case, use the direct commands shown above.

---

## 📁 Repository Structure

```text
src/                          # training, tuning, explainability, API code
src/api/                      # FastAPI inference service
models/                       # saved model artifacts
configs/                      # frozen threshold, schema, model metadata
data/                         # raw/interim/working datasets
tests/                        # API and preprocessing tests
reports/                      # monthly reports, snippets, figures, evidence
.github/workflows/            # CI workflows
Dockerfile                    # container build definition
.dockerignore                 # Docker build context exclusions
Makefile                      # convenience commands for run / test / docker
README.md                     # project overview
README_deployment.md          # deployment-focused usage guide
```

---

## 📚 Key Artifacts

- `models/xgb_final.joblib` — frozen final champion model
- `configs/threshold.json` — final threshold policy
- `configs/feature_schema.json` — frozen serving schema
- `configs/model_metadata.json` — model provenance and serving metadata
- `reports/month5/model_card.md` — model card
- `README_deployment.md` — deployment-focused usage guide

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
- Reproducible preprocessing hardening
- Structured logging and observability
- Automated tests
- GitHub Actions CI
- Dockerisation
- Docker-ready local deployment documentation

**Next steps**

- local smoke testing inside the built container
- optional Streamlit/demo UI
- deployment hardening beyond local serving
- monitoring and future MLOps extensions

---

## 🎯 What This Project Demonstrates

This project showcases hands-on experience in:

- applied machine learning under class imbalance
- model evaluation beyond accuracy
- threshold optimisation for business goals
- explainable AI for risk-sensitive use cases
- turning an ML model into a deployable inference service
- reproducible preprocessing and serving contracts
- configuration-driven deployment logic
- automated testing and CI-backed quality control
- containerising a Python API with Docker
- structuring a project for both thesis documentation and portfolio presentation

---

## 👤 Author

**Lazaros Voulistiotis**  
Final-year BSc Computer Science student  
Aspiring Machine Learning Engineer

---

## 📄 License / Dataset Notice

This repository contains the code, report artifacts, and deployment skeleton for the project. The original raw dataset is not redistributed in this repository. Please obtain the Kaggle dataset separately and place it under `data/data_raw/` if you want to reproduce the full pipeline.
