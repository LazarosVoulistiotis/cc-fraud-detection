# Model Card — Credit Card Fraud Detection Champion

## 1. Model Summary
- **Model name:** XGBoost Champion
- **Artifact:** `models/xgb_final.joblib`
- **Origin:** frozen from `models/xgb_week8.joblib`
- **Serving layer:** FastAPI inference API (`src/api/`)
- **Task:** binary classification for fraud detection
- **Positive class:** fraud (`1`)
- **Negative class:** legitimate transaction (`0`)

## 2. Intended Use
This model is intended for:
- academic demonstration of a production-inspired fraud detection pipeline
- API-based scoring of individual credit card transactions
- explainable, reproducible serving of a frozen Week 17 champion model
- showcasing model packaging, thresholding, and inference design decisions

This model is **not** intended to be used as a real banking production system without additional work on monitoring, drift detection, retraining, governance, security, and integration.

## 3. Dataset
- **Source:** Kaggle Credit Card Fraud Detection dataset
- **Raw schema:** `Time`, `V1`–`V28`, `Amount`
- **Target:** `Class`
- **Data nature:** highly imbalanced binary classification problem
- **Feature characteristics:** anonymized PCA-style features plus transaction time and amount

## 4. Serving Schema
### Raw canonical input expected by the API
- `Time`
- `V1` to `V28`
- `Amount`

### Engineered features created internally at serving time
- `Hour`
- `hour_sin`
- `hour_cos`
- `Amount_log1p`

### Final serving contract
The API aligns all inputs to the exact frozen feature order stored in:
- `configs/feature_schema.json`

## 5. Threshold Policy
- **Policy name:** `precision_constraint_p80`
- **Selection basis:** locked from validation-based Week 16 model selection
- **Serving threshold:** `0.1279`
- **Decision rule:**
  - predict `fraud` if `fraud_probability >= 0.1279`
  - predict `legit` otherwise

## 6. Prediction Output
The serving API returns:
- `model_version`
- `threshold_policy`
- `threshold_used`
- `fraud_probability`
- `predicted_label`

The demo endpoint `POST /predict_by_id` also returns:
- `row_id`
- `true_label`

## 7. Business Objective
The model was selected to balance fraud detection usefulness with operational practicality.

Primary goals:
- detect as many fraudulent transactions as possible
- maintain precision at a more operationally acceptable level
- reduce unnecessary analyst workload
- reduce avoidable customer friction from excessive false positives

## 8. Validation Evidence from Week 17 API Serving
The frozen API layer was validated for the following behaviors:
- `GET /health` → `200 OK`
- `GET /metadata` → `200 OK`
- valid `POST /predict` → `200 OK`
- valid `POST /predict_by_id` → `200 OK`
- invalid `row_id` → `404 Not Found`
- extra unexpected field in payload → `422 Unprocessable Content`
- negative `Amount` → `422 Unprocessable Content`

This confirms that the model artifact, threshold policy, schema alignment, and request validation are all wired into the Week 17 inference API.

## 9. Strengths
- frozen and reproducible serving setup
- explicit threshold policy outside the codebase
- exact feature alignment through a frozen schema file
- raw-input API contract with internal feature engineering
- demo-friendly endpoint for deterministic testing
- clean separation between schema validation, preprocessing, loading, and routing

## 10. Limitations
- public dataset may not reflect real production fraud behavior
- features are anonymized, which limits business interpretability
- no real transaction ID exists in the source dataset
- no online learning or retraining loop is implemented
- no drift detection, monitoring, or alerting is implemented yet
- no authentication, rate limiting, or production-grade security controls are included yet
- API currently scores single transactions rather than high-throughput batch or streaming workloads

## 11. Ethical and Operational Considerations
A fraud model can affect customer experience and analyst workload. In real deployment, this type of system should be supported by:
- careful threshold governance
- human review processes
- fairness and error analysis
- monitoring for concept drift and degradation
- secure audit logging and access control

## 12. Future Improvements
Planned or logical next steps include:
- containerization with Docker
- environment pinning and deployment hardening
- batch scoring support
- authentication and API security
- drift monitoring and performance tracking
- retraining workflow and model version registry
- richer explainability outputs for live predictions

## 13. Artifact References
- `models/xgb_final.joblib`
- `configs/threshold.json`
- `configs/feature_schema.json`
- `data/data_interim/splits_week8/test_with_row_id.csv`
- `src/api/main.py`
- `src/api/preprocess.py`
- `src/api/model_loader.py`
- `src/api/schemas.py`
