# Deployment README — Credit Card Fraud Detection API

## Overview
This API serves the **frozen Week 17 XGBoost champion model** for credit card fraud detection.

It exposes four main endpoints:
- `GET /health`
- `GET /metadata`
- `POST /predict`
- `POST /predict_by_id`

The serving layer is built around a **frozen inference contract** so that local testing, demo behaviour, Docker execution, and cloud deployment all rely on the same artifacts, schema, and threshold policy.

---

## Frozen Artifacts

The API depends on the following frozen files:
- `models/xgb_final.joblib` — final XGBoost champion artifact
- `configs/threshold.json` — locked threshold policy (`precision_constraint_p80`, threshold `0.1279`)
- `configs/feature_schema.json` — frozen serving feature schema and feature order
- `configs/model_metadata.json` — model provenance and serving metadata
- `data/data_interim/splits_week8/test_with_row_id.csv` — demo lookup dataset for `/predict_by_id`

---

## Project Structure

```text
src/api/main.py                               # FastAPI entry point and routes
src/api/schemas.py                            # Pydantic request/response schemas
src/api/preprocess.py                         # Serving-time feature engineering and alignment
src/api/model_loader.py                       # Cached loading of configs, model, and demo data
configs/threshold.json                        # Frozen threshold policy
configs/feature_schema.json                   # Frozen raw / engineered / model feature schema
configs/model_metadata.json                   # Frozen serving metadata
models/xgb_final.joblib                       # Frozen champion model artifact
data/data_interim/splits_week8/test_with_row_id.csv   # Demo lookup dataset
reports/month5/model_card.md                  # Model card for the frozen serving model
```

---

## Prerequisites
- Python 3.13+
- Project virtual environment (`.venv` recommended)
- Installed dependencies from `requirements.txt`

---

## Environment Setup

Create and activate a virtual environment, then install dependencies.

### 1. Create the virtual environment
```bash
python -m venv .venv
```

### 2. Activate the environment

#### Windows PowerShell
```powershell
.venv\Scripts\Activate.ps1
```

#### Git Bash
```bash
source .venv/Scripts/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Run the API Locally

From the **project root**, start the FastAPI app with Uvicorn:

```bash
uvicorn src.api.main:app --reload
```

If startup is successful, the API will be available at:

```text
http://127.0.0.1:8000
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

---

## Run via Docker

### Build the image
```bash
docker build -t fraud-api .
```

### Run the container
```bash
docker run --rm -p 8000:8000 fraud-api
```

### Containerized docs URL
```text
http://127.0.0.1:8000/docs
```

### Quick smoke test
```bash
curl http://localhost:8000/health
curl http://localhost:8000/metadata
curl -X POST "http://localhost:8000/predict_by_id" -H "Content-Type: application/json" -d '{"row_id": 0}'
```

---

## Live Cloud Deployment (Google Cloud Run)

The final API was deployed as a public Cloud Run service.

### Public endpoint
```text
https://cc-fraud-api-726136433853.europe-west1.run.app
```

### Interactive docs
```text
https://cc-fraud-api-726136433853.europe-west1.run.app/docs
```

### Notes
- The root path `/` may return `{"detail":"Not Found"}`. This is expected because the service is an API rather than a web landing page.
- The correct public browser entry point is `/docs`.

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

## Serving Contract

### Input contract for `POST /predict`
The endpoint accepts the **raw canonical transaction fields**:
- `Time`
- `V1` to `V28`
- `Amount`

The API derives the following engineered features internally:
- `Hour`
- `hour_sin`
- `hour_cos`
- `Amount_log1p`

After feature engineering, the request is aligned to the exact frozen feature order defined in `configs/feature_schema.json`.

### Output contract for prediction endpoints
Successful predictions return:
- `model_version`
- `threshold_policy`
- `threshold_used`
- `fraud_probability`
- `predicted_label`

The `POST /predict_by_id` endpoint additionally returns:
- `row_id`
- `true_label`

---

## Endpoints

### `GET /health`
Simple liveness check.

#### Example response
```json
{
  "status": "ok"
}
```

### `GET /metadata`
Returns serving metadata, including:
- model version
- model artifact path
- train date
- threshold policy
- threshold used
- raw input feature list
- engineered feature list
- final frozen model feature order

### `POST /predict`
Scores a raw transaction payload.

#### Example request
```json
{
  "Time": 12345.67,
  "V1": -1.359807,
  "V2": -0.072781,
  "V3": 2.536347,
  "V4": 1.378155,
  "V5": -0.338321,
  "V6": 0.462388,
  "V7": 0.239599,
  "V8": 0.098698,
  "V9": 0.363787,
  "V10": 0.090794,
  "V11": -0.5516,
  "V12": -0.617801,
  "V13": -0.99139,
  "V14": -0.311169,
  "V15": 1.468177,
  "V16": -0.470401,
  "V17": 0.207971,
  "V18": 0.025791,
  "V19": 0.403993,
  "V20": 0.251412,
  "V21": -0.018307,
  "V22": 0.277838,
  "V23": -0.110474,
  "V24": 0.066928,
  "V25": 0.128539,
  "V26": -0.189115,
  "V27": 0.133558,
  "V28": -0.021053,
  "Amount": 149.62
}
```

#### Example response
```json
{
  "model_version": "xgb_final",
  "threshold_policy": "precision_constraint_p80",
  "threshold_used": 0.1279,
  "fraud_probability": 0.000014150726201478392,
  "predicted_label": "legit"
}
```

### `POST /predict_by_id`
Demo-only endpoint that looks up a frozen row from `test_with_row_id.csv`, rebuilds the raw canonical payload, and scores it through the same serving pipeline as `POST /predict`.

#### Example request
```json
{
  "row_id": 0
}
```

#### Example response
```json
{
  "model_version": "xgb_final",
  "threshold_policy": "precision_constraint_p80",
  "threshold_used": 0.1279,
  "fraud_probability": 0.00004742308647109894,
  "predicted_label": "legit",
  "row_id": 0,
  "true_label": 0
}
```

---

## Validation and Error Behaviour

The serving API has been validated for the following behaviours:
- valid `POST /predict` request → `200 OK`
- valid `POST /predict_by_id` request → `200 OK`
- invalid `row_id` → `404 Not Found`
- extra unexpected field in `/predict` payload → `422 Unprocessable Content`
- negative `Amount` in `/predict` payload → `422 Unprocessable Content`

### Example invalid `row_id` response
```json
{
  "detail": "row_id 999999999 not found"
}
```

### Example extra field validation response
```json
{
  "detail": [
    {
      "type": "extra_forbidden",
      "loc": ["body", "dummy"],
      "msg": "Extra inputs are not permitted",
      "input": 123
    }
  ]
}
```

---

## Troubleshooting

### `ModuleNotFoundError` or import issues
Run the server from the project root:
```bash
uvicorn src.api.main:app --reload
```

### Feature-name / XGBoost inference issues
The serving path uses explicit feature-aware XGBoost scoring to avoid feature-name mismatch issues between the frozen model and runtime input.

### Dependency problems
Rebuild the environment if needed.

#### Git Bash
```bash
rm -rf .venv
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

#### PowerShell
```powershell
Remove-Item -Recurse -Force .venv
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Docs page does not open
Confirm that:
- the server started successfully
- port `8000` is available
- you are opening `http://127.0.0.1:8000/docs` for local mode
- or `https://cc-fraud-api-726136433853.europe-west1.run.app/docs` for live cloud mode

---

## Versioning Intent

The serving layer establishes a reproducible deployment baseline through:
- frozen model artifact
- frozen threshold policy
- frozen feature schema
- deterministic preprocessing and feature alignment
- shared prediction path for both direct input and demo lookup

This makes packaging, Docker execution, and live cloud deployment much more reliable.
