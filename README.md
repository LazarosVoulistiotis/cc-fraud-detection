# 💳 Credit Card Fraud Detection System

A production-inspired machine learning system for detecting fraudulent credit card transactions under extreme class imbalance.

This repository documents the full lifecycle of the project: data understanding, exploratory analysis, business-aware modelling, threshold policy selection, explainability, API serving, operational hardening, Docker-based packaging, live cloud deployment, and final release-readiness validation.

---

## 🚀 Project Highlights

- Built and evaluated multiple fraud detection models, including **Logistic Regression**, **Decision Tree**, **Random Forest**, and **XGBoost**
- Selected **XGBoost** as the final champion model based on locked evaluation and business suitability
- Treated the classification threshold as an **operational policy**, not a default probability cutoff
- Finalized a **precision-constrained threshold policy** to reduce false positives while preserving strong fraud capture
- Added **SHAP** and **LIME** explainability for both global and local model interpretation
- Implemented a working **FastAPI** inference service with:
  - `GET /health`
  - `GET /metadata`
  - `POST /predict`
  - `POST /predict_by_id`
- Hardened the serving layer with:
  - deterministic preprocessing
  - frozen feature alignment
  - config-driven thresholding
  - structured JSON logging
  - centralized error handling
  - automated tests with `pytest`
  - GitHub Actions CI on push / pull request
- Added **Docker packaging** for reproducible local deployment
- Validated the containerized API through a **local smoke test**
- Deployed the final API as a **live public service on Google Cloud Run**
- Completed a **Week 21 release-readiness pass** with frozen-system validation, threshold sensitivity analysis, and API robustness testing

---

## 📌 Problem Statement

Credit card fraud detection is a high-impact binary classification problem with severe class imbalance. In the public benchmark dataset used here, fraud represents only a tiny fraction of all transactions, which makes raw accuracy misleading.

The real challenge is not to maximize accuracy, but to balance:

- **Recall** — catching as many fraudulent transactions as possible
- **Precision** — limiting false alarms that create customer friction and analyst overhead
- **Operational cost** — recognizing that missed fraud is usually much more expensive than an unnecessary review

This project approaches fraud detection as a **business-aware ML system**, not just a leaderboard exercise.

---

## 🧠 Final Model Snapshot

### Champion Model

- **Model:** XGBoost Classifier
- **Frozen artifact:** `models/xgb_final.joblib`
- **Serving threshold policy:** `precision_constraint_p80`
- **Threshold:** `0.1279`

### Final Locked Validation Snapshot (Week 21)

The final frozen system was re-validated in **Week 21** without retraining the model, re-selecting the threshold, or changing the serving policy.

| Metric | Value |
|---|---:|
| Test size | **56,746** |
| Fraud cases | **95** |
| ROC-AUC | **0.96995** |
| PR-AUC | **0.81713** |
| Precision | **0.82796** |
| Recall | **0.81053** |
| F1-score | **0.81915** |
| TP | **77** |
| FP | **16** |
| FN | **18** |
| TN | **56,635** |
| Alerts / 10k transactions | **16.39** |
| Cost / transaction (`FP=1`, `FN=20`) | **0.006626** |

### Business Interpretation

At the frozen serving threshold `0.1279`, the final model:

- catches **77 fraudulent transactions**
- misses **18 fraud cases**
- triggers only **16 false positives**
- preserves strong precision while keeping alert volume operationally manageable

This makes the final XGBoost serving setup more operationally attractive than lower-threshold alternatives that preserve similar recall but generate more false alarms.

---

## 🗂️ Dataset

This project uses the well-known **Kaggle Credit Card Fraud Detection** dataset.

### Raw Dataset Characteristics

- **Rows:** 284,807 transactions
- **Fraud cases:** 492
- **Fraud rate:** ~0.17%
- **Features:** `Time`, `V1`–`V28`, `Amount`, `Class`

### Notes

- `V1`–`V28` are anonymized PCA-style features
- `Time` is measured in seconds from the first recorded transaction
- `Amount` is the transaction value
- `Class = 1` indicates fraud
- The raw dataset is stored locally under `data/data_raw/` and is **not committed** to the repository

---

## 🧪 Modelling Journey

The project was developed progressively, moving from simpler baselines toward a stronger deployment-ready champion model.

### Models evaluated

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

### Key modelling decisions

- Use **PR-AUC**, **precision**, **recall**, **F1**, and business-facing confusion metrics instead of relying on raw accuracy
- Apply **validation-based threshold selection**
- Keep the **test set locked**
- Treat the threshold as a **risk / operational policy**
- Prefer the model and threshold pair that balances fraud capture with lower false-positive burden

### Final selection

The final selected system was:

- **Champion model:** XGBoost
- **Final threshold policy:** `precision_constraint_p80`
- **Reason for selection:** strong fraud capture with lower analyst workload and lower customer friction than alternative operating points

---

## 🔍 Explainability

Explainability is a core part of the final system.

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

This makes the final model more transparent and more appropriate for fraud and risk use cases where interpretability matters.

---

## ⚙️ Serving Design

The deployed inference contract uses a frozen serving schema and configuration-driven serving policy.

### Input to `POST /predict`

The API accepts **raw canonical input**:

- `Time`
- `V1` to `V28`
- `Amount`

### Engineered inside the API

The API derives the following features at serving time:

- `Hour`
- `hour_sin`
- `hour_cos`
- `Amount_log1p`

### Reproducible preprocessing

The serving pipeline:

- validates the frozen schema
- computes engineered features deterministically
- aligns features to the exact frozen model feature order
- rejects missing, unexpected, or invalid inputs

This reduces the risk of inference-time feature mismatch and improves deployment reliability.

### Demo endpoint

The `POST /predict_by_id` endpoint accepts a frozen `row_id` and reconstructs the raw payload from:

- `data/data_interim/splits_week8/test_with_row_id.csv`

This endpoint is intended for demonstration and report evidence, not for real production use.

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

## ☁️ Live Cloud Deployment

The final serving API was deployed as a live public service on **Google Cloud Run**.

### Public endpoint
```text
https://cc-fraud-api-726136433853.europe-west1.run.app
```

### Interactive API docs
```text
https://cc-fraud-api-726136433853.europe-west1.run.app/docs
```

### Notes

- The root path `/` may return `{"detail":"Not Found"}`. This is expected because the deployment is an API service rather than a website landing page.
- The correct browser entry point for documentation and interactive testing is `/docs`.

### Example live calls
```bash
export SERVICE_URL="https://cc-fraud-api-726136433853.europe-west1.run.app"

curl "$SERVICE_URL/health"
curl "$SERVICE_URL/metadata"

curl -X POST "$SERVICE_URL/predict_by_id" \
  -H "Content-Type: application/json" \
  -d '{"row_id": 0}'
```

---

## 📈 Observability and Operational Hardening

The API was hardened to behave less like a fragile local demo and more like a production-inspired service.

### Structured logging

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

### Error handling

The FastAPI app includes centralized error handling for:

- invalid request payloads
- explicit HTTP errors
- unexpected server-side exceptions

### Configuration-driven serving

The serving layer relies on:

- `configs/threshold.json`
- `configs/feature_schema.json`
- `configs/model_metadata.json`

This means thresholding, schema alignment, and model provenance are managed through configs rather than hardcoded values.

### Monitoring and drift concept

The deployed system is documented with a production-aware monitoring concept around:

- latency and service health
- alert rate
- precision / recall drift
- feature distribution drift
- retraining triggers based on schedule or drift thresholds

---

## 🧪 Testing and CI

### Test coverage

The project includes automated tests for:

- health endpoint behavior
- prediction endpoint behavior
- metadata endpoint behavior
- preprocessing and feature-engineering logic
- golden-path demo prediction flow

### Current test structure

```text
tests/
├── test_health.py
├── test_predict.py
├── test_preprocess.py
└── test_golden.py
```

### Local result

```bash
pytest -q
22 passed in 3.46s
```

### Continuous Integration

A GitHub Actions workflow automatically runs the test suite on:

- every push to `main`
- every pull request to `main`

The CI workflow was designed to be artifact-safe by mocking model-dependent API test paths where appropriate.

---

## ✅ Week 21 Release-Readiness Summary

Week 21 functioned as the final validation and release-readiness pass of the project.

### What Week 21 confirmed

- the frozen model artifact, feature schema, metadata config, and threshold config are aligned
- the locked hold-out metrics reproduce exactly under the deployed serving policy
- threshold sensitivity can be discussed **without leakage** and without redefining serving policy
- the API is robust for valid inputs and for most malformed payloads
- the final system is ready to be presented as a deployment-oriented academic project

### Important transparency note

Week 21 also documented one remaining serving-layer hardening issue:

- **non-finite numeric payloads** (`NaN`, `Infinity`) can still trigger `500 Internal Server Error` instead of a clean validation response

This is documented transparently as a future hardening improvement, not as a model-quality issue.

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
- **Cloud Deployment:** Google Cloud Run, Cloud Build, Artifact Registry
- **Visualization:** matplotlib
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

### Quick smoke test
From a second terminal, you can verify the containerized API with:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/metadata
curl -X POST "http://localhost:8000/predict_by_id" \
  -H "Content-Type: application/json" \
  -d '{"row_id": 0}'
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
- `reports/month6/week21_release_readiness/week21_summary.md` — final validation and release-readiness summary
- `reports/month6/week21_release_readiness/week21_appendix_ready.md` — supporting Week 21 evidence
- `README_deployment.md` — deployment-focused usage guide

---

## ✅ Project Status

### Completed

- Data exploration and preprocessing
- Baseline modelling
- Business-aware model selection
- Threshold optimization
- SHAP explainability
- LIME explainability
- Frozen model artifact and serving schema
- Working FastAPI inference API
- Reproducible preprocessing hardening
- Structured logging and observability
- Automated tests
- GitHub Actions CI
- Dockerization
- Successful local smoke testing of the containerized API
- Live Cloud Run deployment
- Final locked-system validation in Week 21
- Threshold sensitivity analysis without leakage
- API robustness / edge-case testing
- Release-readiness documentation and evidence organization

### Optional future improvements

These are **not** missing core project steps. They are optional enhancements beyond the completed project scope.

- clean validation handling for `NaN` / `Infinity` payloads
- authenticated or private deployment variant
- stronger production monitoring implementation
- retraining / challenger workflow
- optional Streamlit or demo UI layer

---

## 🎯 What This Project Demonstrates

This project showcases hands-on experience in:

- applied machine learning under extreme class imbalance
- model evaluation beyond accuracy
- threshold optimization for business goals
- explainable AI for risk-sensitive use cases
- turning an ML model into a deployable inference service
- reproducible preprocessing and serving contracts
- configuration-driven deployment logic
- automated testing and CI-backed quality control
- containerizing a Python API with Docker
- deploying a public ML inference API on Google Cloud Run
- validating a live cloud-hosted service
- performing final release-readiness checks on a frozen ML system
- structuring a project for both thesis documentation and portfolio presentation

---

## 👤 Author

**Lazaros Voulistiotis**  
Final-year BSc Computer Science student  
Aspiring Machine Learning Engineer

---

## 📄 License / Dataset Notice

This repository contains the code, report artifacts, and deployment skeleton for the project. The original raw dataset is not redistributed in this repository. Please obtain the Kaggle dataset separately and place it under `data/data_raw/` if you want to reproduce the full pipeline.
