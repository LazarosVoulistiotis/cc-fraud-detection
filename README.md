# 💳 Credit Card Fraud Detection System

A production-inspired machine learning project for detecting fraudulent credit card transactions under **extreme class imbalance**, combining business-aware model evaluation, explainable AI, API serving, Docker packaging, automated tests, and live cloud deployment.

This repository supports the CN6000 final-year project:

> **Credit Card Fraud Detection with Explainable Machine Learning and Production-Inspired Implementation**  
> Final-year BSc Computer Science project by **Lazaros Voulistiotis**

---

## Executive Summary

Credit card fraud detection is not a simple accuracy-driven classification task. Fraud cases are rare, false negatives can represent financial loss, and false positives can create customer friction and operational workload.

This project follows an end-to-end applied machine learning workflow:

1. understand the fraud detection problem and the dataset,
2. perform EDA and leakage-safe preprocessing,
3. compare baseline and ensemble models,
4. select a validation-based threshold policy,
5. interpret the final model using SHAP and LIME,
6. expose the frozen model through a FastAPI inference API,
7. package the service with Docker,
8. validate the deployed API on Google Cloud Run.

The final selected solution is an **XGBoost champion model** served through a **FastAPI API** using a frozen schema and a precision-constrained threshold policy.

---

## Final Outcome at a Glance

| Area | Final project outcome |
|---|---|
| Problem type | Binary fraud classification |
| Dataset | Kaggle Credit Card Fraud Detection |
| Fraud rate | ~0.17% in the raw dataset |
| Champion model | XGBoost Classifier |
| Final threshold policy | `precision_constraint_p80` |
| Final threshold | `0.1279` |
| Explainability | SHAP + LIME |
| Serving layer | FastAPI |
| Packaging | Docker |
| CI/testing | pytest + GitHub Actions |
| Cloud deployment | Google Cloud Run |
| Project status | Completed academic proof of concept |

---

## Project Highlights

- Built and evaluated **Logistic Regression**, **Decision Tree**, **Random Forest**, and **XGBoost** models.
- Selected **XGBoost** as the champion model based on locked test evaluation and business suitability.
- Treated the classification threshold as an **operational risk policy**, not as a default `0.50` cutoff.
- Finalized a precision-constrained threshold policy to reduce false positives while preserving strong fraud capture.
- Added **SHAP** for global and local model interpretation.
- Added **LIME** for model-agnostic local explanations of individual predictions.
- Implemented a working **FastAPI inference service** with:
  - `GET /health`
  - `GET /metadata`
  - `POST /predict`
  - `POST /predict_by_id`
- Hardened the serving layer with:
  - deterministic preprocessing,
  - frozen feature alignment,
  - config-driven thresholding,
  - structured JSON logging,
  - centralized error handling,
  - automated tests with `pytest`,
  - GitHub Actions CI.
- Added Docker packaging for reproducible runtime execution.
- Validated the containerized API through local smoke tests.
- Deployed the final API as a live service on **Google Cloud Run**.
- Completed a final release-readiness pass with frozen-system validation and threshold sensitivity analysis.

---

## Problem Statement

Credit card fraud detection is a high-impact and highly imbalanced binary classification problem. In the public benchmark dataset used in this project, fraudulent transactions represent only a very small fraction of all transactions.

This makes **accuracy misleading**. A model could achieve very high accuracy by predicting almost every transaction as legitimate, while still missing the fraud cases that matter most.

The real objective is to balance:

- **Recall** — catching as many fraud cases as possible.
- **Precision** — keeping fraud alerts reliable and avoiding excessive false positives.
- **Operational cost** — recognizing that false negatives and false positives have different business consequences.
- **Explainability** — supporting analyst trust and responsible use in a risk-sensitive domain.

This project therefore approaches fraud detection as a **business-aware ML system**, not simply as a leaderboard exercise.

---

## Dataset

This project uses the well-known **Kaggle Credit Card Fraud Detection** dataset.

### Raw Dataset Characteristics

| Property | Value |
|---|---:|
| Transactions | 284,807 |
| Fraud cases | 492 |
| Fraud rate | ~0.17% |
| Features | `Time`, `V1`–`V28`, `Amount`, `Class` |
| Target | `Class` |
| Positive class | `1 = fraud` |
| Negative class | `0 = legitimate` |

### Dataset Notes

- `V1`–`V28` are anonymized PCA-style features.
- `Time` is measured in seconds from the first transaction in the dataset.
- `Amount` is the transaction amount.
- `Class` is the binary target label.
- The original raw dataset is **not committed** to this repository.
- To reproduce the full pipeline, place `creditcard.csv` under:

```text
data/data_raw/creditcard.csv
```

---

## Methodology

The project was developed progressively through a leakage-safe and reproducible workflow.

### 1. Data Quality and EDA

The initial analysis focused on:

- checking dataset shape and schema,
- verifying missing values and duplicate records,
- measuring class imbalance,
- analyzing `Amount` skewness,
- exploring `Time` as a relative temporal feature,
- studying correlations between PCA components and the target class.

Key EDA findings:

- The dataset is extremely imbalanced.
- `Amount` is heavily right-skewed, motivating `Amount_log1p`.
- `Time` can be converted into a proxy hour feature.
- Several PCA features, such as `V17`, `V14`, `V12`, `V10`, and `V16`, show stronger relationship with `Class`.

### 2. Feature Engineering

The final serving schema uses raw inputs and derives additional features internally:

- `Hour`
- `hour_sin`
- `hour_cos`
- `Amount_log1p`

The API accepts the canonical raw transaction schema and computes engineered features during serving. This reduces the risk of mismatches between training and inference.

### 3. Model Comparison

The following models were implemented and compared:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

The evaluation emphasized metrics that are suitable for imbalanced classification:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion matrix
- False positives and false negatives
- Business-oriented threshold behavior

### 4. Threshold Selection

Threshold selection was treated as a decision policy.

The final policy was selected on the validation set and then applied once to the locked test set. The test set was not used for tuning.

Final policy:

```text
precision_constraint_p80
```

Decision rule:

```text
predict fraud if fraud_probability >= 0.1279
```

---

## Final Model Snapshot

### Champion Model

| Item | Value |
|---|---|
| Model | XGBoost Classifier |
| Frozen artifact | `models/xgb_final.joblib` |
| Serving threshold policy | `precision_constraint_p80` |
| Threshold | `0.1279` |
| Positive class | Fraud |
| Serving API | FastAPI |

### Final Locked Test Snapshot

The final frozen system was re-validated without retraining, threshold re-selection, or serving policy changes.

| Metric | Value |
|---|---:|
| Test size | **56,746** |
| Fraud cases | **95** |
| ROC-AUC | **0.96995** |
| PR-AUC | **0.81713** |
| Precision | **0.82796** |
| Recall | **0.81053** |
| F1-score | **0.81915** |
| True Positives | **77** |
| False Positives | **16** |
| False Negatives | **18** |
| True Negatives | **56,635** |
| Alerts / 10k transactions | **16.39** |
| Cost / transaction (`FP=1`, `FN=20`) | **0.006626** |

### Business Interpretation

At the final frozen threshold `0.1279`, the XGBoost model:

- catches **77 fraud cases**,
- misses **18 fraud cases**,
- triggers only **16 false positives**,
- preserves strong precision,
- keeps alert volume operationally manageable.

This makes the final XGBoost serving setup more practical than lower-threshold alternatives that preserve similar recall but generate more false alarms.

---

## Explainable AI

Explainability is a core part of the project because fraud detection is a risk-sensitive domain.

### SHAP

SHAP was used for:

- global feature importance,
- beeswarm analysis,
- dependence analysis,
- local waterfall explanations,
- case studies for true-positive, true-negative, and borderline predictions.

The strongest global drivers were mainly anonymized PCA features, including:

- `V4`
- `V14`
- `V8`
- `V12`
- `V15`
- `V11`

Because the dataset uses anonymized PCA features, the interpretation is mostly technical/statistical rather than fully business-semantic.

### LIME

LIME was used for:

- local explanation of individual predictions,
- complementary model-agnostic interpretation,
- case-level analysis for fraud analyst style review.

SHAP and LIME were used together to improve confidence in model behavior, especially for true-positive and borderline cases.

---

## Serving Design

The final inference layer is designed around a frozen serving contract.

### Raw Input to `POST /predict`

The API accepts:

- `Time`
- `V1` to `V28`
- `Amount`

### Engineered Inside the API

The API derives:

- `Hour`
- `hour_sin`
- `hour_cos`
- `Amount_log1p`

### Reproducible Serving Pipeline

The serving pipeline:

- validates the incoming request,
- rejects missing or unexpected fields,
- computes engineered features deterministically,
- aligns all features to the frozen model feature order,
- loads the frozen model artifact,
- applies the frozen threshold policy,
- returns a structured JSON response.

This reduces the risk of training/serving skew and improves deployment reliability.

---

## API Endpoints

### `GET /health`

Simple liveness check.

Example response:

```json
{
  "status": "ok"
}
```

### `GET /metadata`

Returns model and serving metadata, including:

- model version,
- model artifact path,
- git commit,
- train date,
- threshold policy,
- threshold used,
- schema version,
- raw input features,
- engineered features,
- final model feature order.

### `POST /predict`

Scores a raw transaction payload and returns:

- fraud probability,
- predicted label,
- threshold used,
- threshold policy,
- model version.

### `POST /predict_by_id`

Scores a frozen demo row by `row_id`.

This endpoint is intended for demonstration and report evidence, not for real production use.

It reconstructs the raw payload from:

```text
data/data_interim/splits_week8/test_with_row_id.csv
```

---

## Live Cloud Deployment

The final FastAPI service was deployed on **Google Cloud Run**.

### Public Service URL

```text
https://cc-fraud-api-726136433853.europe-west1.run.app
```

### Interactive Swagger UI

```text
https://cc-fraud-api-726136433853.europe-west1.run.app/docs
```

### Notes

- The root path `/` may return `{"detail": "Not Found"}`.
- This is expected because the deployment is an API service, not a website.
- Use `/docs` for browser-based testing.

### Example Live Calls

```bash
export SERVICE_URL="https://cc-fraud-api-726136433853.europe-west1.run.app"

curl "$SERVICE_URL/health"
curl "$SERVICE_URL/metadata"

curl -X POST "$SERVICE_URL/predict_by_id" \
  -H "Content-Type: application/json" \
  -d '{"row_id": 0}'
```

---

## Observability and API Hardening

The API includes production-inspired hardening features.

### Structured Logging

The service emits structured JSON logs for:

- request completion,
- prediction scoring,
- validation errors,
- HTTP errors,
- unexpected exceptions,
- metadata requests.

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

The full raw transaction payload is not logged by default, reducing unnecessary exposure of input data.

### Error Handling

The API includes centralized handling for:

- invalid request payloads,
- explicit HTTP exceptions,
- unexpected server-side errors.

### Config-Driven Serving

Serving behavior is controlled by configuration files:

```text
configs/threshold.json
configs/feature_schema.json
configs/model_metadata.json
```

This keeps thresholding, schema alignment, and model provenance outside hardcoded application logic.

---

## Testing and CI

### Test Coverage

The project includes automated tests for:

- `/health` endpoint behavior,
- `/metadata` endpoint behavior,
- `/predict` endpoint behavior,
- preprocessing and feature engineering,
- schema validation,
- golden-path demo prediction flow.

### Test Structure

```text
tests/
├── test_health.py
├── test_predict.py
├── test_preprocess.py
└── test_golden.py
```

### Local Test Result

```bash
pytest -q
22 passed in 3.46s
```

### Continuous Integration

A GitHub Actions workflow runs tests automatically on:

- pushes to `main`,
- pull requests to `main`.

The CI workflow is artifact-safe by mocking model-dependent API paths where appropriate.

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/LazarosVoulistiotis/cc-fraud-detection.git
cd cc-fraud-detection
```

### 2. Create and Activate a Virtual Environment

#### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### Git Bash

```bash
python -m venv .venv
source .venv/Scripts/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the API Locally

```bash
uvicorn src.api.main:app --reload
```

### 5. Open Swagger UI

```text
http://127.0.0.1:8000/docs
```

### 6. Run Tests

```bash
pytest -q
```

---

## Run via Docker

The repository includes a Docker-based local deployment path so the API can run in a reproducible environment.

### Build the Image

```bash
docker build -t fraud-api .
```

### Run the Container

```bash
docker run --rm -p 8000:8000 fraud-api
```

### Open the API

```text
http://127.0.0.1:8000/docs
```

### Docker Smoke Test

From a second terminal:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/metadata

curl -X POST "http://localhost:8000/predict_by_id" \
  -H "Content-Type: application/json" \
  -d '{"row_id": 0}'
```

### Optional Makefile Commands

If `make` is installed:

```bash
make run
make test
make docker
make docker-run-quick
```

What each target does:

| Command | Purpose |
|---|---|
| `make run` | Starts the API locally with auto-reload |
| `make test` | Runs the test suite |
| `make docker` | Builds the Docker image |
| `make docker-run-quick` | Runs the Dockerized API |

On some Windows Git Bash setups, `make` may not be installed by default. In that case, use the direct commands above.

---

## Repository Structure

```text
src/                          # training, tuning, explainability, API code
src/api/                      # FastAPI inference service
models/                       # saved model artifacts
configs/                      # frozen threshold, schema, model metadata
data/                         # raw/interim/working datasets
tests/                        # API and preprocessing tests
reports/                      # reports, snippets, figures, evidence
.github/workflows/            # GitHub Actions CI workflows
Dockerfile                    # container build definition
.dockerignore                 # Docker build context exclusions
Makefile                      # convenience commands
README.md                     # project overview
README_deployment.md          # deployment-focused usage guide
```

---

## Key Runtime Artifacts

| Artifact | Purpose |
|---|---|
| `models/xgb_final.joblib` | Frozen final champion model |
| `configs/threshold.json` | Final threshold policy |
| `configs/feature_schema.json` | Frozen serving schema |
| `configs/model_metadata.json` | Model provenance and serving metadata |
| `data/data_interim/splits_week8/test_with_row_id.csv` | Demo lookup file for `/predict_by_id` |
| `README_deployment.md` | Deployment-focused usage guide |

---

## Project Status

### Completed

- Data exploration and preprocessing
- Leakage-safe feature engineering
- Baseline modelling
- Business-aware model comparison
- Threshold optimization
- SHAP explainability
- LIME explainability
- Frozen model artifact and serving schema
- FastAPI inference API
- Deterministic preprocessing hardening
- Structured logging and metadata exposure
- Automated tests
- GitHub Actions CI
- Dockerization
- Local Docker smoke testing
- Google Cloud Run deployment
- Final locked-system validation
- Release-readiness documentation

### Optional Future Improvements

These are not missing core project steps; they are logical extensions beyond the completed academic proof of concept.

- production-grade authentication and authorization,
- rate limiting and abuse protection,
- encrypted secrets management,
- full audit logging,
- batch or streaming inference,
- drift monitoring implementation,
- scheduled retraining loop,
- champion/challenger model governance,
- probability calibration,
- richer real-world banking features,
- analyst-facing review dashboard.

---

## Limitations and Responsible Use

This project is a technically mature **academic proof of concept**, not a final banking production system.

Key limitations:

- The dataset is public, anonymized, and limited to two days of transactions.
- The PCA-transformed features restrict direct business interpretation.
- The model has not been validated on live banking data.
- No real customer, merchant, device, or geographic features are included.
- No full drift monitoring or retraining loop is implemented.
- The API is designed for demonstration and single-transaction scoring.
- Real deployment would require security, governance, compliance review, human oversight, and operational monitoring.

In a real fraud operations environment, this type of model should support **human-in-the-loop decision making** rather than automatically penalizing customers without review.

---

## What This Project Demonstrates

This project demonstrates practical experience in:

- applied machine learning under extreme class imbalance,
- model evaluation beyond accuracy,
- business-aware threshold optimization,
- explainable AI for risk-sensitive use cases,
- SHAP and LIME interpretability workflows,
- turning an ML model into a deployable API service,
- deterministic preprocessing and frozen serving contracts,
- configuration-driven model serving,
- automated tests and CI-backed quality control,
- Docker-based runtime packaging,
- cloud deployment on Google Cloud Run,
- release-readiness validation for a frozen ML system,
- structuring an academic ML project for portfolio presentation.

---

## Author

**Lazaros Voulistiotis**  
Final-year BSc Computer Science student  
Aspiring Machine Learning Engineer

---

## License and Dataset Notice

This repository contains project code, report artifacts, and deployment material for academic and portfolio purposes.

The original raw dataset is not redistributed in this repository. To reproduce the full training pipeline, obtain the Kaggle Credit Card Fraud Detection dataset separately and place it under:

```text
data/data_raw/creditcard.csv
```
