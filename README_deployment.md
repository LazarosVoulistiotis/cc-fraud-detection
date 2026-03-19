# Deployment README — Credit Card Fraud Detection API

## Overview
This API serves the **frozen Week 17 XGBoost champion model** for credit card fraud detection.

It exposes four endpoints:
- `GET /health`
- `GET /metadata`
- `POST /predict`
- `POST /predict_by_id`

The serving layer is built around a **frozen inference contract** so that local testing, demo behavior, and later deployment work from the same artifacts and schema.

## Frozen Artifacts
The API depends on the following frozen files:
- `models/xgb_final.joblib` — final XGBoost champion artifact
- `configs/threshold.json` — locked threshold policy (`precision_constraint_p80`, threshold `0.1279`)
- `configs/feature_schema.json` — frozen serving feature schema and feature order
- `data/data_interim/splits_week8/test_with_row_id.csv` — demo lookup dataset for `/predict_by_id`

## Project Structure

```text
src/api/main.py                               # FastAPI entry point and routes
src/api/schemas.py                            # Pydantic request/response schemas
src/api/preprocess.py                         # Serving-time feature engineering and alignment
src/api/model_loader.py                       # Cached loading of configs, model, and demo data
configs/threshold.json                        # Frozen threshold policy
configs/feature_schema.json                   # Frozen raw / engineered / model feature schema
models/xgb_final.joblib                       # Frozen champion model artifact
data/data_interim/splits_week8/test_with_row_id.csv   # Demo lookup dataset
reports/month5/model_card.md                  # Model card for the frozen serving model
```

## Prerequisites
- Python 3.13+
- Project virtual environment (`.venv` recommended)
- Installed dependencies from `requirements.txt`

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

## Run the API
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

## Validation and Error Behavior
The Week 17 API has been tested for the following behaviors:
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

## Example cURL Commands

### Health check
```bash
curl http://127.0.0.1:8000/health
```

### Metadata
```bash
curl http://127.0.0.1:8000/metadata
```

### Predict from raw fields
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### Predict from demo row ID
```bash
curl -X POST "http://127.0.0.1:8000/predict_by_id" \
  -H "Content-Type: application/json" \
  -d '{"row_id": 0}'
```

## Troubleshooting

### `ModuleNotFoundError` or import issues
Run the server from the project root:

```bash
uvicorn src.api.main:app --reload
```

### Feature-name / XGBoost inference issues
The Week 17 serving path uses explicit feature-aware XGBoost scoring to avoid feature-name mismatch issues between the frozen model and runtime input.

### Dependency problems
Rebuild the environment if needed:

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
- you are opening `http://127.0.0.1:8000/docs`

## Versioning Intent
Week 17 establishes a reproducible deployment baseline:
- frozen model artifact
- frozen threshold policy
- frozen feature schema
- deterministic preprocessing and feature alignment
- shared prediction path for both direct input and demo lookup

This makes later packaging, API hardening, containerization, and cloud deployment much easier.
