# Week 20 — Cloud Deployment, Architecture Diagram, Monitoring Concept, and Report Closure

## Overview
This document summarizes the work completed in **Week 20** of the Credit Card Fraud Detection project.

The goal of this week was to move from a **deployment-ready local container** to a **live cloud-hosted inference service**, document the deployed architecture, define a practical monitoring and drift strategy at concept level, and close the Month 5 milestone with report-ready deployment material.

Unlike the fallback plan of documenting a simulated deployment only, this week achieved a **real public deployment** of the fraud detection API on **Google Cloud Run**.

---

## Week 20 Goals
The main goals for this week were:

1. Select a student-friendly cloud deployment target
2. Deploy the Dockerized FastAPI API to the cloud
3. Make the service publicly reachable through a live endpoint
4. Validate the deployed service using real endpoint calls
5. Produce a deployment architecture diagram for the report
6. Define a lightweight monitoring and drift strategy
7. Prepare report-ready deployment and conclusion material

---

## 1. Cloud Platform Selection

Two safe deployment directions were considered:

- **Google Cloud Run** — student-friendly, container-based, managed, low-ops path for FastAPI
- **AWS** — useful for CV relevance, with Elastic Beanstalk / EC2 / container-based deployment as future alternatives

### Final decision
The primary deployment target selected for Week 20 was:

**Google Cloud Run**

### Why Google Cloud Run was selected
This choice fit the project particularly well because:

- the API had already been containerized in Week 19
- the serving layer was already hardened and smoke-tested locally
- the deployment target only needed to run the existing Dockerized runtime
- the service could be exposed publicly with minimal infrastructure management

### Secondary path retained
AWS remained documented as a valid secondary path for future extension and CV relevance, but was not selected as the primary deployment target for this week.

---

## 2. Google Cloud Project Setup

A new Google Cloud project was created and configured for deployment.

### Setup actions completed
- created a dedicated Google Cloud project for the fraud detection service
- confirmed the project was attached to the free trial billing account
- opened and used **Cloud Shell**
- verified the active project configuration
- enabled the required APIs:
  - `run.googleapis.com`
  - `cloudbuild.googleapis.com`
  - `artifactregistry.googleapis.com`

### Active project
```text
cc-fraud-detection-491222
```

### Notes
At this stage, the billing overview still showed **$0.00** cost and free trial credit remained available.

---

## 3. Repository Preparation in Cloud Shell

The GitHub repository was cloned into Cloud Shell and the deployment source directory was verified.

### Commands used
```bash
git clone https://github.com/LazarosVoulistiotis/cc-fraud-detection.git
cd cc-fraud-detection
pwd
ls
ls Dockerfile
cat Dockerfile
```

### Verified runtime packaging assumptions
The repository root already contained:

- `Dockerfile`
- `requirements.txt`
- `src/`
- `configs/`
- project documentation files

This confirmed that the repository was structured correctly for cloud build and container deployment.

---

## 4. First Cloud Run Deployment Attempt

A first deployment attempt was made using direct source deployment.

### Command used
```bash
gcloud run deploy cc-fraud-api \
  --source . \
  --region europe-west1 \
  --allow-unauthenticated \
  --port 8000
```

### What happened
The deployment flow correctly:
- detected the source directory
- prepared Artifact Registry
- started the build process

However, the build/deploy path did not complete successfully.

### Observed problem
The initial source deployment path failed during the build/deploy process and further investigation showed that the issue was not with Cloud Run itself, but with the deployment inputs required by the Docker image.

---

## 5. Build Failure Analysis and Root Cause

The Cloud Build logs showed that the container image build failed because the Docker build context did not contain all required runtime artifacts.

### Root cause identified
The Dockerfile expected the following runtime assets to exist:

- `models/xgb_final.joblib`
- `data/data_interim/splits_week8/test_with_row_id.csv`

These were not present in the Cloud Shell clone because the repository's ignore rules excluded:

- `models/`
- large parts of `data/data_interim/`

As a result, the build failed when Docker reached the following copy steps:

```dockerfile
COPY models /app/models
COPY data/data_interim/splits_week8/test_with_row_id.csv /app/data/data_interim/splits_week8/test_with_row_id.csv
```

### Why this mattered
These files are essential at runtime because:
- the serving policy points to `models/xgb_final.joblib`
- `/predict_by_id` depends on the demo CSV for row-based lookup
- the deployed API cannot score without the frozen artifact and demo lookup data

---

## 6. Fixes Applied

The deployment issue was resolved through a sequence of practical fixes.

### 6.1 Runtime artifact upload
The missing files were uploaded manually into Cloud Shell:

- `xgb_final.joblib`
- `test_with_row_id.csv`

They were then placed into the expected repository paths:

```text
models/xgb_final.joblib
data/data_interim/splits_week8/test_with_row_id.csv
```

### 6.2 `.gcloudignore` creation
A `.gcloudignore` file was created to ensure the uploaded runtime assets would be included in the Cloud Build source archive rather than silently excluded by git-based ignore behaviour.

### 6.3 Manual image build strategy
Instead of relying again on the failing source deploy wrapper, the deployment process was split into two clear steps:

1. **build and push image manually with Cloud Build**
2. **deploy the image to Cloud Run**

This produced a more controlled and debuggable deployment workflow.

---

## 7. Successful Cloud Build

After the missing runtime artifacts were uploaded and `.gcloudignore` was created, the container image build succeeded.

### Command used
```bash
gcloud builds submit . \
  --tag=europe-west1-docker.pkg.dev/cc-fraud-detection-491222/cloud-run-source-deploy/cc-fraud-api:manual-v1 \
  --region=europe-west1
```

### Build outcome
- Docker image built successfully
- runtime files copied into image correctly
- image pushed successfully to Artifact Registry

### Final image reference
```text
europe-west1-docker.pkg.dev/cc-fraud-detection-491222/cloud-run-source-deploy/cc-fraud-api:manual-v1
```

### Build status
```text
STATUS: SUCCESS
```

---

## 8. Successful Cloud Run Deployment

After the image was built successfully, it was deployed to Cloud Run as a live public service.

### Command used
```bash
gcloud run deploy cc-fraud-api \
  --image=europe-west1-docker.pkg.dev/cc-fraud-detection-491222/cloud-run-source-deploy/cc-fraud-api:manual-v1 \
  --region=europe-west1 \
  --allow-unauthenticated \
  --port=8000
```

### Deployment outcome
The deployment completed successfully and Cloud Run reported that the service was serving 100% of traffic.

### Service URL
```text
https://cc-fraud-api-726136433853.europe-west1.run.app
```

### Important observation
A browser request to the root path `/` returned:

```json
{"detail":"Not Found"}
```

This is expected because the service is an API, not a website landing page.

---

## 9. Live Endpoint Validation

The live deployment was validated through direct HTTPS calls and browser-based inspection.

### 9.1 Swagger / OpenAPI docs
The deployed docs endpoint loaded successfully at:

```text
https://cc-fraud-api-726136433853.europe-west1.run.app/docs
```

The docs page showed the expected routes:

- `GET /health`
- `GET /metadata`
- `POST /predict`
- `POST /predict_by_id`

### 9.2 Health endpoint
```bash
curl "$SERVICE_URL/health"
```

Response:
```json
{"status":"ok"}
```

This confirmed that the service was live and reachable.

### 9.3 Metadata endpoint
```bash
curl "$SERVICE_URL/metadata"
```

The endpoint returned valid serving metadata, including:
- `model_version: xgb_final`
- `model_artifact_path: /app/models/xgb_final.joblib`
- `threshold_policy: precision_constraint_p80`
- `threshold_policy_version: week16-locked`
- `threshold_used: 0.1279`
- schema and feature metadata

This confirmed that the deployed service had access to:
- the frozen model artifact
- the locked serving configuration
- the frozen schema metadata

### 9.4 Demo scoring endpoint
```bash
curl -X POST "$SERVICE_URL/predict_by_id" \
  -H "Content-Type: application/json" \
  -d '{"row_id": 0}'
```

Observed response included:
- `model_version: xgb_final`
- `threshold_policy: precision_constraint_p80`
- `threshold_used: 0.1279`
- a valid `fraud_probability`
- `predicted_label: legit`
- `row_id: 0`

This confirmed successful end-to-end live inference:
- request accepted
- lookup row loaded
- preprocessing applied
- model scored
- threshold policy applied
- JSON response returned

---

## 10. Deployment Architecture Diagram

A dedicated architecture diagram was created in draw.io / diagrams.net for use in the final report.

### Final architecture flow
```text
User / Client
→ Public Cloud Run Endpoint
→ REST API (FastAPI)
→ Preprocess / Feature Align
→ XGBoost Model
→ JSON Response
```

### Side components included
- **Logs / Monitoring**
- **Model Registry (concept)**
- **Retraining Pipeline (future)**

### Purpose of the diagram
The figure documents:
- the live deployed serving boundary
- the request path from client to response
- the deployed runtime components
- operational observability concerns
- future deployment maturity extensions

### Suggested figure title
**Deployed architecture of the Credit Card Fraud Detection API on Google Cloud Run**

### Suggested figure caption
> Client requests are routed through a public Cloud Run endpoint to a FastAPI-based inference service. The deployed service validates the request schema, performs deterministic preprocessing and feature alignment, invokes the frozen XGBoost champion model, and returns a structured JSON response. Operational logging and monitoring are shown as supporting components, while model registry integration and retraining are included as future-oriented architectural extensions.

---

## 11. Monitoring, Drift Detection, and Retraining Concept

Week 20 also introduced a small but production-aware monitoring and drift strategy at concept level.

### What should be monitored
#### Operational metrics
- request latency
- endpoint status codes
- error rate
- alert rate
- request traceability

#### Model behaviour metrics
- precision drift
- recall drift
- fraud alert rate drift

#### Input / feature drift
- feature distribution drift
- Population Stability Index (PSI) as one possible monitoring metric
- drift on engineered features such as:
  - `Hour`
  - `hour_sin`
  - `hour_cos`
  - `Amount_log1p`

### Retraining trigger strategy
A practical retraining policy would combine:

#### Scheduled trigger
- monthly review
- quarterly retraining
- regular offline validation cycle

#### Event-based trigger
- latency exceeds service expectations
- precision or recall drops materially
- PSI exceeds a defined drift threshold
- alert behaviour changes significantly over time

### Why this matters
This section makes the system look more mature by showing awareness of:
- online model degradation
- concept drift
- data drift
- operational service quality
- controlled champion/challenger replacement thinking

---

## 12. Public Endpoint and Example Usage

### Public endpoint
```text
https://cc-fraud-api-726136433853.europe-west1.run.app
```

### Example calls
```bash
export SERVICE_URL="https://cc-fraud-api-726136433853.europe-west1.run.app"

curl "$SERVICE_URL/health"
curl "$SERVICE_URL/metadata"

curl -X POST "$SERVICE_URL/predict_by_id" \
  -H "Content-Type: application/json" \
  -d '{"row_id": 0}'
```

### Docs URL
```text
https://cc-fraud-api-726136433853.europe-west1.run.app/docs
```

---

## 13. Files and Artifacts Produced or Updated

The main Week 20 artifacts included:

### Cloud deployment artifacts
- live Cloud Run service
- Artifact Registry image
- deployed public endpoint

### Diagram/report artifacts
- architecture diagram
- Week 20 report-ready deployment notes
- report snippet for final dissertation integration

### Runtime/deployment support changes during deployment work
- `.gcloudignore`
- uploaded runtime artifacts required by deployment:
  - `models/xgb_final.joblib`
  - `data/data_interim/splits_week8/test_with_row_id.csv`

---

## 14. Main Deliverables Produced

The main Week 20 deliverables are:

1. successful live deployment to Google Cloud Run
2. public HTTPS endpoint for the fraud detection API
3. live validation of `/health`, `/metadata`, `/docs`, and `/predict_by_id`
4. deployment architecture diagram for the report
5. monitoring and drift concept subsection
6. completed deployment and conclusion material for the final report

---

## 15. Deployment and Operationalization Summary

Week 20 transformed the project from a **local container demo** into a **live cloud-hosted ML inference service**.

Before this week, the serving system was:
- containerized
- smoke-tested locally
- deployment-ready in principle

After this week, the project now has:
- a live public cloud endpoint
- an operational deployment narrative
- validated live serving behaviour
- a documented deployed architecture
- a production-aware monitoring and retraining concept

This is an important milestone because it means the project is no longer only an offline modelling exercise or local demo. It now demonstrates a realistic end-to-end path from model selection to hosted ML inference.

---

## 16. Suggested Report Section Titles

For the final report, this week can feed directly into sections such as:

- **Cloud deployment of the fraud detection API**
- **Deployed architecture and live endpoint validation**
- **Monitoring, drift detection, and retraining strategy**
- **Deployment and conclusion**

---

## 17. Suggested Final Report Narrative

A concise report-ready summary could be:

> In Week 20, the fraud detection API was deployed as a live public service on Google Cloud Run. After configuring a dedicated Google Cloud project, enabling the required APIs, and preparing the repository inside Cloud Shell, the initial deployment flow was adapted into a manual build-and-deploy workflow to ensure that the required runtime artifacts were included in the container image. The final image was successfully built with Cloud Build, stored in Artifact Registry, and deployed to Cloud Run. The resulting service exposed a public endpoint together with working `/health`, `/metadata`, `/predict`, `/predict_by_id`, and `/docs` routes. In addition to the live deployment itself, a deployment architecture diagram was created and a lightweight monitoring and drift strategy was documented, including latency monitoring, alert-rate tracking, precision/recall drift, feature distribution drift, and retraining triggers. This week completed the transition from deployment-ready local packaging to a production-inspired live inference service.

---

## 18. Completion Status

Week 20 is considered **successfully completed**.

### Completed
- [x] cloud target selected
- [x] Google Cloud project created and configured
- [x] Cloud Run / Cloud Build / Artifact Registry APIs enabled
- [x] repository cloned into Cloud Shell
- [x] deployment blockers diagnosed
- [x] missing runtime artifacts uploaded
- [x] `.gcloudignore` created for build inclusion control
- [x] container image built successfully with Cloud Build
- [x] image pushed to Artifact Registry
- [x] Cloud Run service deployed successfully
- [x] public live endpoint obtained
- [x] `/docs` loaded successfully
- [x] `/health` validated successfully
- [x] `/metadata` validated successfully
- [x] `/predict_by_id` validated successfully
- [x] architecture diagram created
- [x] monitoring and drift concept documented
- [x] report-ready Week 20 material prepared

### Optional next improvements
- [ ] add a leaner `requirements-api.txt` for smaller cloud images
- [ ] replace metadata placeholder training commit SHA
- [ ] add budget alert / billing screenshot to appendix if desired
- [ ] add authenticated or private deployment variant
- [ ] add dashboard-backed monitoring and automated drift checks
- [ ] add registry-backed model promotion workflow
