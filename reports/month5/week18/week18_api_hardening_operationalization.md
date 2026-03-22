# Week 18 — API Hardening, Tests, Logging, Metadata, and CI

## Overview
This document summarizes the work completed in **Week 18** of the Credit Card Fraud Detection project.

The objective of this week was to move the API from a fragile local demo toward a more **production-inspired serving layer** by improving:
- reproducible preprocessing
- structured logging and observability
- API and preprocessing tests
- metadata/versioning exposure
- config-driven serving behavior
- CI automation with GitHub Actions

---

## Week 18 Goals
The main goals for this week were:

1. Test the API locally with sample transactions
2. Improve error handling for missing or invalid inputs
3. Make preprocessing deterministic and deployment-safe
4. Add structured JSON logging
5. Add automated tests with `pytest`
6. Expose versioning and serving metadata through `/metadata`
7. Keep thresholding fully config-driven
8. Add GitHub Actions CI for automatic test execution on push / pull request

---

## 1. Reproducible Preprocessing
A hardened preprocessing layer was implemented to ensure that inference-time inputs are transformed exactly as expected by the frozen model.

### What was implemented
- validation of the serving feature schema
- deterministic feature engineering for:
  - `Hour`
  - `hour_sin`
  - `hour_cos`
  - `Amount_log1p`
- strict feature alignment to the frozen model feature order
- numeric coercion and rejection of invalid values
- explicit handling of:
  - missing fields
  - unexpected fields
  - negative `Time`
  - negative `Amount`
  - `NaN` / `inf`
- one-row float DataFrame output for model serving

### Why it matters
This reduces the risk of:
- feature-name mismatch
- inconsistent inference inputs
- silent schema drift
- deployment-time preprocessing bugs

---

## 2. Structured Logging and Observability
The API was upgraded with structured JSON logging for basic observability.

### What was implemented
- `logging_config.py` created for centralized logger configuration
- request-level logging middleware added
- global exception handlers added for:
  - validation errors (`422`)
  - explicit HTTP errors
  - unexpected server exceptions
- prediction-level logging added for inference endpoints
- metadata request logging added for `/metadata`

### Logged event types
- `request_completed`
- `request_failed_before_response`
- `validation_error`
- `http_error`
- `unhandled_exception`
- `prediction_scored`
- `metadata_requested`

### Typical logged fields
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

### Logging design principle
The API does **not** log the full raw transaction payload by default.  
This keeps logs cleaner and avoids unnecessary exposure of input data.

---

## 3. API Hardening
The FastAPI application was improved to support more robust serving behavior.

### Main changes
- request ID added to each request
- `X-Request-ID` added to responses
- centralized JSON error responses
- `/metadata` upgraded to expose richer serving metadata
- prediction logic kept inside request-time execution
- threshold selection remains config-driven
- feature ordering remains schema-driven

---

## 4. Automated Tests with pytest
A test suite was created to validate both preprocessing logic and the API contract.

### Current test structure
```text
tests/
├── test_health.py
├── test_predict.py
├── test_preprocess.py
└── test_golden.py
```

### Test coverage
#### Health endpoint
`tests/test_health.py`
- `/health` returns `200`
- response body matches expected health payload
- `X-Request-ID` header is present

#### Prediction and metadata API tests
`tests/test_predict.py`
- `/metadata` returns versioning and serving metadata fields
- `/predict` returns a valid fraud probability in `[0,1]`
- `/predict` rejects missing required fields with `422`
- API tests are CI-safe through mocking of:
  - model loading
  - model path resolution
  - model metadata

#### Preprocessing tests
`tests/test_preprocess.py`
- schema validation
- duplicate feature rejection
- feature order mismatch rejection
- engineered feature creation
- missing columns handling
- negative amount handling
- negative time handling
- exact feature alignment
- deterministic payload preparation
- rejection of missing / unexpected / invalid inputs

#### Golden test
`tests/test_golden.py`
- optional reference-style test for a known demo prediction path

### Final local test results
```bash
pytest -q
22 passed in 3.46s
```

### CI result
The GitHub Actions workflow completed successfully after:
- making Windows-only dependencies platform-specific
- adding `pytest`
- running tests with `python -m pytest -q tests`
- making API tests independent from the real model artifact via mocking

---

## 5. Versioning Metadata
The `/metadata` endpoint was extended so that it behaves like a lightweight model-registry view.

### Metadata now exposes
- `model_version`
- `model_artifact_path`
- `git_commit`
- `train_date`
- `threshold_policy`
- `threshold_policy_version`
- `threshold_used`
- `schema_version`
- `raw_input_features`
- `engineered_features`
- `model_features`
- `training_data_reference`
- `training_target`
- `framework`
- `task`

### Supporting config file
A dedicated config file was added:

```text
configs/model_metadata.json
```

This stores model provenance and serving metadata separately from thresholding logic and schema definition.

---

## 6. Config-Driven Threshold Hardening
Thresholding was already configuration-based, but this week it was hardened and formalized.

### Single source of truth
The serving layer now relies on:
- `configs/threshold.json`
- `configs/feature_schema.json`
- `configs/model_metadata.json`

### Improvements
- explicit validation added for `threshold.json`
- required keys checked
- threshold range checked (`0.0 <= threshold <= 1.0`)
- model artifact reference checked
- serving code continues to read threshold from config instead of hardcoding values

### Why this matters
This improves:
- reproducibility
- maintainability
- auditability
- controlled policy changes without editing core serving code

---

## 7. GitHub Actions CI
A GitHub Actions workflow was added to automatically run the test suite on every push and pull request to `main`.

### Workflow file
```text
.github/workflows/tests.yml
```

### CI workflow steps
1. Check out repository
2. Set up Python 3.11
3. Upgrade pip
4. Install dependencies
5. Run test suite

### CI adjustments made
During setup, a few CI-specific issues had to be resolved:
- Windows-only dependencies were made platform-specific in `requirements.txt`
- `pytest` was added explicitly to dependencies
- test execution was updated to:
  ```bash
  python -m pytest -q tests
  ```
- API tests were made artifact-independent through mocking

### Final result
The GitHub Actions workflow completed successfully.

This means:
- preprocessing logic is verified automatically
- API contract regressions are caught automatically
- broken changes are more likely to be detected before merge

---

## 8. Files Added or Updated
The Week 18 work included changes to files such as:

### API and serving layer
- `src/api/main.py`
- `src/api/preprocess.py`
- `src/api/model_loader.py`
- `src/api/schemas.py`
- `src/api/logging_config.py`

### Config and metadata
- `configs/threshold.json`
- `configs/feature_schema.json`
- `configs/model_metadata.json`

### Tests
- `tests/test_health.py`
- `tests/test_predict.py`
- `tests/test_preprocess.py`
- `tests/test_golden.py`
- `pytest.ini`

### CI
- `.github/workflows/tests.yml`

### Report artifacts
- `reports/month5/week18/`
- `reports/report_snippets/week18_Operationalization considerations versioning, configs, CI.md`

---

## 9. Deliverables Produced
The main Week 18 deliverables are:

1. hardened inference-time preprocessing
2. structured JSON logging with request and prediction observability
3. API and preprocessing test suite
4. `/metadata` endpoint with richer serving/versioning metadata
5. config-driven thresholding with validation
6. GitHub Actions test workflow
7. CI-safe mocking strategy for model-dependent API tests
8. Week 18 report-ready documentation and snippets

---

## 10. Operationalization Summary
Week 18 moved the project significantly closer to a production-inspired ML API.

Before this week, the API was primarily a local inference demo.

After this week, the API now has:
- deterministic preprocessing
- stricter error handling
- structured logs
- reproducible config-driven serving
- explicit versioning metadata
- automated tests
- CI verification on push / PR

This reduces fragility and makes the serving layer more professional, traceable, and maintainable.

---

## 11. Suggested Report Section Title
For the final report, this week can feed directly into a section titled:

**Operationalization considerations: versioning, configs, CI, and observability**

---

## 12. Suggested Final Report Narrative
A concise report-ready summary could be:

> In Week 18, the fraud detection API was hardened from a local prototype into a more operationally robust serving layer. Deterministic preprocessing and strict feature alignment were implemented to reduce inference-time schema mismatch risk. Structured JSON logging and centralized exception handling improved observability and debugging. A pytest-based test suite validated preprocessing and API behavior, while GitHub Actions CI ensured these checks run automatically on every push and pull request. In addition, a richer `/metadata` endpoint and configuration-driven threshold management improved version traceability, reproducibility, and serving governance.

---

## 13. Completion Status
Week 18 is considered **successfully completed**.

### Completed
- [x] reproducible preprocessing
- [x] structured logging / observability
- [x] API hardening
- [x] pytest test suite
- [x] metadata/versioning endpoint
- [x] config-driven threshold hardening
- [x] GitHub Actions CI
- [x] report-ready Week 18 documentation

### Optional next improvements
- [ ] add `logs/README.md`
- [ ] export `/metadata` snapshot to `artifacts/metadata.json`
- [ ] add workflow status badge to main `README.md`
