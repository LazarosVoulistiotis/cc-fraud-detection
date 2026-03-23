# Week 19 — Dockerization, Runtime Packaging, and Local Smoke Test

## Overview
This document summarizes the work completed in **Week 19** of the Credit Card Fraud Detection project.

The goal of this week was to move the FastAPI serving layer from a local Python-only execution path to a **containerized, portable runtime** that can run reproducibly across environments. The focus was on packaging the API with Docker, reducing build noise through `.dockerignore`, adding a one-command developer interface with a `Makefile`, updating the repository documentation, and validating the entire service with a local smoke test.

---

## Week 19 Goals
The main goals for this week were:

1. Containerize the FastAPI fraud detection API with Docker
2. Use an official Python base image and follow practical Docker best practices
3. Install runtime dependencies through `requirements.txt`
4. Package the frozen model artifact and serving configs inside the image
5. Include the demo CSV required by `/predict_by_id`
6. Add `.dockerignore` to reduce build context size
7. Add a `Makefile` with simple developer commands
8. Update `README.md` with Docker usage instructions
9. Verify the container with a full local smoke test

---

## 1. Dockerfile Design and Containerization

A Dockerfile was created to package the FastAPI application into a runnable container image.

### Main design choices
- base image: `python:3.13-slim`
- working directory set to `/app`
- dependencies installed from `requirements.txt`
- application code copied from `src/`
- frozen runtime configs copied from `configs/`
- frozen model artifact copied from `models/`
- demo CSV copied for `/predict_by_id`
- service executed with Uvicorn on port `8000`
- container configured to run as a non-root user

### Final runtime command
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Why this matters
This makes the API:
- portable across machines
- easier to demonstrate
- less dependent on local Python environments
- more aligned with deployment-oriented workflows

---

## 2. Docker Best Practices Applied

The Dockerfile was intentionally structured using practical best practices.

### a. Official slim Python image
The image uses:

```dockerfile
FROM python:3.13-slim
```

This keeps the base image smaller than a full Python image while still providing a clean runtime foundation.

### b. Dependency installation via separate cached layer
`requirements.txt` was copied before the application code so that Docker can reuse the dependency-install layer when code changes but dependencies stay the same.

This improves rebuild speed significantly during iteration.

### c. Minimal runtime copy strategy
Only runtime-relevant assets were copied into the image:
- `src/`
- `configs/`
- `models/`
- `data/data_interim/splits_week8/test_with_row_id.csv`

This avoids shipping unnecessary training, reporting, or notebook files into the runtime container.

### d. Non-root execution
A dedicated application user was created and the service was run as a non-root user.  
This is a better operational practice than running the API as root inside the container.

### e. Explicit port exposure
The Dockerfile exposes port `8000`, matching the Uvicorn service port and simplifying host/container port mapping.

---

## 3. `.dockerignore` and Build Context Reduction

A `.dockerignore` file was added so that unnecessary folders and files are excluded from the Docker build context.

### Examples of excluded content
- `.git`
- `.venv/`
- `reports/`
- `.github/`
- `data/data_raw/`
- `data/data_external/`
- `data/data_processed/`
- most of `data/data_interim/`

### Explicit exception added
The following file was explicitly kept available to the build:

```text
data/data_interim/splits_week8/test_with_row_id.csv
```

This was necessary because the `/predict_by_id` endpoint depends on it for demo row lookup.

### Why it matters
Reducing the build context:
- speeds up builds
- avoids copying irrelevant artifacts
- keeps the image cleaner
- makes packaging more intentional

---

## 4. `Makefile` for One-Command Usage

A `Makefile` was added to make the project easier to run and test with short commands.

### Targets added
- `make run` → run the FastAPI app locally with reload
- `make test` → run the test suite with `pytest -q`
- `make docker` → build the Docker image as `fraud-api`
- `make docker-run-quick` → run the container and expose port `8000`

### Notes from local environment
In the current Windows Git Bash environment, `make` was not installed and produced:

```text
bash: make: command not found
```

This did **not** affect the Dockerization work itself.  
All smoke tests were completed successfully using the direct Docker commands instead.

### Why the `Makefile` is still useful
The file remains valuable because:
- it improves repository usability
- it documents the intended command shortcuts
- it supports future use on systems where `make` is available
- it makes the project look more polished and developer-friendly

---

## 5. README Update

The main `README.md` was updated to reflect the new Dockerized runtime.

### Main additions
- new **Run via Docker** section
- Docker build command:
  ```bash
  docker build -t fraud-api .
  ```
- Docker run command:
  ```bash
  docker run --rm -p 8000:8000 fraud-api
  ```
- mention of the new `Dockerfile`, `.dockerignore`, and `Makefile`
- update to the project status to indicate that Dockerization is completed

### Why it matters
This improves:
- onboarding for reviewers
- GitHub presentation quality
- reproducibility for local testing
- project clarity for the final submission

---

## 6. Local Smoke Test Procedure

The API was validated locally from the project root using Docker.

### Commands used
```bash
docker build -t fraud-api .
docker run --rm -p 8000:8000 fraud-api
```

Because `make` was unavailable in the current Git Bash environment, the smoke test used the direct Docker commands instead of `make docker` and `make docker-run-quick`.

---

## 7. Local Smoke Test Results

The local smoke test was successfully completed.

### 7.1 Docker build
The first complete build succeeded after dependency installation and file copy steps completed normally.  
A later rebuild completed very quickly using Docker cache, confirming that the layer strategy was working correctly.

### Build observations
- first full successful build completed normally
- later rebuild reused cached layers
- build context remained very small (`2.90 kB` shown in Docker output)
- final image tagged successfully as:
  ```text
  fraud-api:latest
  ```

### 7.2 Container startup
The container started successfully and Uvicorn reported:

```text
Uvicorn running on http://0.0.0.0:8000
```

This confirmed that the API booted correctly inside the container.

### 7.3 Health endpoint
The following endpoint was tested:

```bash
curl http://localhost:8000/health
```

Response:
```json
{"status":"ok"}
```

This confirmed that the service was live and reachable through the host port mapping.

### 7.4 Docs endpoint
The Swagger UI was opened through:

```text
http://localhost:8000/docs
```

This loaded successfully, confirming that the FastAPI docs and OpenAPI generation were functioning inside the container.

### 7.5 Metadata endpoint
The metadata endpoint was tested:

```bash
curl http://localhost:8000/metadata
```

It returned valid serving metadata, including:
- `model_version: xgb_final`
- `model_artifact_path: /app/models/xgb_final.joblib`
- `threshold_policy: precision_constraint_p80`
- `threshold_policy_version: week16-locked`
- `threshold_used: 0.1279`
- `schema_version: week17-freeze-v1`

This confirmed that the container had access to the frozen model, configs, and serving metadata.

### 7.6 Demo prediction endpoint
The demo endpoint was tested with:

```bash
curl -X POST "http://localhost:8000/predict_by_id" \
  -H "Content-Type: application/json" \
  -d '{"row_id": 0}'
```

Response:
```json
{
  "model_version":"xgb_final",
  "threshold_policy":"precision_constraint_p80",
  "threshold_used":0.1279,
  "fraud_probability":4.742308647109894e-6,
  "predicted_label":"legit",
  "row_id":0,
  "true_label":0
}
```

This confirmed that:
- the demo CSV was included correctly in the image
- `/predict_by_id` worked end-to-end
- the model scoring pipeline worked in containerized form

### 7.7 Direct prediction endpoint
The raw prediction endpoint was also tested with a full transaction payload using `POST /predict`.

Response:
```json
{
  "model_version":"xgb_final",
  "threshold_policy":"precision_constraint_p80",
  "threshold_used":0.1279,
  "fraud_probability":0.000014150726201478392,
  "predicted_label":"legit"
}
```

This confirmed that the full inference path worked in the container, including:
- request validation
- preprocessing
- feature engineering
- feature alignment
- model scoring
- threshold-based label generation

---

## 8. Notable Runtime Observations

### Cached rebuild performance
After the first full build, a later `docker build -t fraud-api .` completed very quickly through cached layers.  
This validates the Dockerfile structure, especially the decision to copy `requirements.txt` first.

### Favicon 404
A browser request to `/favicon.ico` returned `404 Not Found`.  
This is expected and does not indicate a problem with the API.

### Metadata placeholder
The `/metadata` response still contained:

```text
REPLACE_WITH_TRAINING_COMMIT_SHA
```

This is not a Docker issue.  
It simply indicates that the training commit placeholder in `configs/model_metadata.json` has not yet been replaced with a real commit SHA.

---

## 9. Files Added or Updated

The Week 19 work included changes to files such as:

### Runtime packaging
- `Dockerfile`
- `.dockerignore`
- `Makefile`

### Documentation
- `README.md`

### Existing runtime assets used by the container
- `src/api/main.py`
- `src/api/model_loader.py`
- `configs/threshold.json`
- `configs/feature_schema.json`
- `configs/model_metadata.json`
- `models/xgb_final.joblib`
- `data/data_interim/splits_week8/test_with_row_id.csv`

### Report artifacts
- `reports/month5/week19/`
- `reports/report_snippets/`

---

## 10. Deliverables Produced

The main Week 19 deliverables are:

1. Dockerized FastAPI runtime
2. official slim Python-based container image
3. reduced Docker build context via `.dockerignore`
4. developer convenience commands through `Makefile`
5. updated Docker documentation in `README.md`
6. successful local smoke test for the containerized service
7. report-ready Week 19 documentation and snippet

---

## 11. Operationalization Summary

Week 19 transformed the serving layer from a local Python execution workflow into a **portable containerized service**.

Before this week, the API depended primarily on the host Python environment and local execution conventions.

After this week, the project now has:
- a reproducible Docker runtime
- a more deployment-ready packaging structure
- explicit inclusion of frozen artifacts
- a smaller and cleaner build context
- a validated localhost container demo
- documented run instructions for future reviewers and users

This makes the project more practical, more portable, and closer to a real deployment workflow.

---

## 12. Suggested Report Section Title

For the final report, this week can feed directly into a section titled:

**Containerization and portable local deployment with Docker**

---

## 13. Suggested Final Report Narrative

A concise report-ready summary could be:

> In Week 19, the fraud detection API was containerized using Docker to improve portability and deployment readiness. A Dockerfile based on the official `python:3.13-slim` image was created to package the FastAPI application, frozen model artifact, serving configurations, and demo lookup data into a self-contained runtime image. A `.dockerignore` file reduced the build context, while a `Makefile` was added to document convenient developer commands. The updated container was validated through a local smoke test, where the image built successfully, the API started correctly on port 8000, and the `/health`, `/docs`, `/metadata`, `/predict_by_id`, and `/predict` endpoints all returned valid responses. This week significantly improved the portability and operational maturity of the serving layer.

---

## 14. Completion Status

Week 19 is considered **successfully completed**.

### Completed
- [x] Dockerfile created
- [x] official slim Python base image used
- [x] runtime dependencies installed inside the image
- [x] frozen model and configs copied into the image
- [x] demo CSV included for `/predict_by_id`
- [x] `.dockerignore` added
- [x] `Makefile` added
- [x] `README.md` updated with Docker usage
- [x] local smoke test completed successfully
- [x] report-ready Week 19 documentation prepared

### Optional next improvements
- [ ] replace metadata placeholder commit SHA with real training commit
- [ ] add Docker image size note to README
- [ ] add `docker compose` support if a multi-service demo is later needed
- [ ] consider splitting runtime and dev dependencies for a leaner image
