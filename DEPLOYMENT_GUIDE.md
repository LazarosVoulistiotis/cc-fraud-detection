# DEPLOYMENT_GUIDE — Credit Card Fraud Detection API

## Overview
This guide explains how to deploy the **Dockerized FastAPI fraud detection API** to two cloud environments:

1. **Google Cloud Run** — recommended primary path
2. **AWS Elastic Beanstalk** — alternative path for AWS exposure

It also includes:
- image build and registry steps
- environment variable guidance
- monitoring / logs guidance
- CI/CD enablement notes

This guide is designed around the project’s current frozen serving assets:
- `models/xgb_final.joblib`
- `configs/threshold.json`
- `configs/feature_schema.json`
- `configs/model_metadata.json`
- `data/data_interim/splits_week8/test_with_row_id.csv`

---

## 1. Prerequisites

Before deploying, make sure you have:

- the project cloned locally
- Docker installed and working
- the runtime artifacts available
- the application running locally or via Docker
- access to either:
  - Google Cloud + Cloud Shell / gcloud CLI
  - AWS account + EB CLI / AWS CLI

---

## 2. Local Validation Before Any Cloud Deployment

### Build the image
```bash
docker build -t fraud-api .
```

### Run the container
```bash
docker run --rm -p 8000:8000 fraud-api
```

### Smoke test
```bash
curl http://localhost:8000/health
curl http://localhost:8000/metadata
curl -X POST "http://localhost:8000/predict_by_id" -H "Content-Type: application/json" -d '{"row_id": 0}'
```

Only continue to cloud deployment after the local container responds correctly.

---

## 3. Option A — Google Cloud Run (Recommended)

## Why this is the recommended path
Cloud Run is the best fit for this project because:
- the API is already containerized
- deployment is image-based and simple
- scale-to-zero reduces idle usage
- no VM management is required
- it is ideal for student-friendly demos and public endpoint exposure

---

## 3.1 Google Cloud Setup

### Create / select a project
Use the Google Cloud Console or Cloud Shell to create/select a project.

### Enable required APIs
```bash
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

### Set the project
```bash
gcloud config set project YOUR_PROJECT_ID
```

---

## 3.2 Build and Push the Container Image

### Recommended image tag format
```text
REGION-docker.pkg.dev/PROJECT_ID/REPOSITORY/IMAGE:TAG
```

### Example build command
```bash
gcloud builds submit .   --tag=europe-west1-docker.pkg.dev/YOUR_PROJECT_ID/cloud-run-source-deploy/cc-fraud-api:manual-v1   --region=europe-west1
```

This command:
- builds the Docker image using the repository `Dockerfile`
- pushes it into Artifact Registry
- produces an image reference that Cloud Run can deploy

---

## 3.3 Deploy to Cloud Run

### Deploy command
```bash
gcloud run deploy cc-fraud-api   --image=europe-west1-docker.pkg.dev/YOUR_PROJECT_ID/cloud-run-source-deploy/cc-fraud-api:manual-v1   --region=europe-west1   --allow-unauthenticated   --port=8000
```

### What this does
- creates a Cloud Run service
- deploys the container image
- exposes the API publicly
- routes traffic to the current revision

### Expected result
Cloud Run returns a **Service URL**, for example:
```text
https://cc-fraud-api-XXXXXXXXXXXX.europe-west1.run.app
```

---

## 3.4 Validate the Live Service

```bash
export SERVICE_URL="https://YOUR_CLOUD_RUN_URL"

curl "$SERVICE_URL/health"
curl "$SERVICE_URL/metadata"
curl -X POST "$SERVICE_URL/predict_by_id"   -H "Content-Type: application/json"   -d '{"row_id": 0}'
```

### Docs URL
```text
https://YOUR_CLOUD_RUN_URL/docs
```

### Note
A request to `/` may return:
```json
{"detail":"Not Found"}
```
This is expected because the service is an API rather than a website landing page.

---

## 3.5 Environment Variable Guidance for Cloud Run

The current project relies mainly on frozen config files rather than runtime environment variables, which is good for reproducibility.

However, if you later want environment variables, examples include:
- `APP_ENV=production`
- `LOG_LEVEL=INFO`
- `MODEL_VERSION=xgb_final`

You can set environment variables during deployment with Cloud Run flags or through the Cloud Run console.

---

## 3.6 Monitoring and Logs on Cloud Run

Recommended production-style observations:
- request latency
- request counts
- HTTP status code distribution
- error rate
- concurrency
- alert rate
- prediction behavior drift (offline)
- feature distribution drift (offline / sampled)

Operational logs should include:
- request IDs
- latency
- status code
- prediction outputs
- threshold used
- model version

Cloud Run integrates with Google Cloud Logging and Cloud Monitoring.

---

## 3.7 CI/CD Direction for Cloud Run

A lightweight CI/CD pattern can be:

1. push code to GitHub
2. GitHub Actions runs tests
3. if tests pass, build container
4. push image to Artifact Registry
5. deploy a new Cloud Run revision

This can be implemented either through:
- GitHub Actions + gcloud auth
- or Google Cloud Build triggers connected to the repository

---

## 4. Option B — AWS Elastic Beanstalk

## Why keep this option
Elastic Beanstalk is a valid secondary deployment path if you want:
- AWS exposure for CV/interview value
- a more EC2-oriented managed deployment model
- experience with AWS-managed application environments

---

## 4.1 High-Level Deployment Model

Elastic Beanstalk provisions and manages the infrastructure for your application environment.
For a Docker-based app, it can deploy a containerized web application on a managed environment.

For this project, the simplest path is:
- package the app as a Docker deployment
- use the Docker platform on Elastic Beanstalk
- deploy through the EB CLI

---

## 4.2 Prepare the Project for EB

The repository already includes a `Dockerfile`, which is the most important part.

### Optional Procfile
If you choose to use a Procfile-based command override for a platform setup that supports it, a typical example would be:

```text
web: uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Important note
For Docker-based deployments on Elastic Beanstalk, the Dockerfile remains the primary runtime packaging artifact.

---

## 4.3 Install / Initialize EB CLI

Install the EB CLI and configure AWS credentials locally.

Then initialize the application:

```bash
eb init
```

You will be asked to choose:
- region
- application name
- platform

Choose a Docker-capable Elastic Beanstalk platform.

---

## 4.4 Create the Environment

```bash
eb create cc-fraud-api-env
```

This creates a managed Elastic Beanstalk environment and provisions the required AWS resources.

---

## 4.5 Deploy Updates

After changes:

```bash
eb deploy
```

Elastic Beanstalk will package and deploy the updated application bundle.

---

## 4.6 Validate the EB Deployment

Get the environment URL:

```bash
eb status
```

Then test:

```bash
curl http://YOUR_EB_URL/health
curl http://YOUR_EB_URL/metadata
```

If the environment is configured correctly, the API should behave similarly to the local/Docker version.

---

## 4.7 Environment Variable Guidance for EB

Elastic Beanstalk allows environment properties through the console or EB CLI.

Examples:
- `APP_ENV=production`
- `LOG_LEVEL=INFO`

Because this project mainly uses frozen config files, environment variables are optional and should remain minimal.

---

## 4.8 Monitoring and Logs on EB

Recommended metrics:
- application latency
- 4xx / 5xx rates
- instance health
- CPU / memory usage
- application logs
- fraud alert rate
- offline model drift metrics

Elastic Beanstalk provides:
- environment health
- instance logs
- CloudWatch integration

---

## 4.9 CI/CD Direction for EB

A typical lightweight CI/CD pattern is:

1. push to GitHub
2. run tests in GitHub Actions
3. if tests pass, create deployable bundle
4. deploy with EB CLI or AWS-integrated deployment workflow

This can later be extended to:
- CodePipeline
- CodeBuild
- GitHub Actions → EB deployment

---

## 5. Monitoring, Drift, and Retraining Strategy

Even though the current deployment is intentionally lightweight, the production-aware monitoring plan should include:

### Operational metrics
- latency (p50 / p95)
- status codes
- error rate
- concurrency / request volume
- alert rate

### Model performance metrics
- precision drift
- recall drift
- F1 / F2 drift
- fraud detection rate changes over time

### Feature drift
Compare live feature distributions against the training distribution using metrics such as:
- PSI (Population Stability Index)

Especially monitor:
- `Time`
- `Amount`
- `Hour`
- `hour_sin`
- `hour_cos`
- `Amount_log1p`

### Retraining triggers
Use one or both of the following:

#### Scheduled trigger
- monthly review
- quarterly retraining

#### Event-based trigger
- PSI exceeds threshold
- precision drops below target
- recall degrades materially
- alert behaviour shifts significantly

---

## 6. Recommended Deployment Choice for This Project

For this project’s current state, the recommended primary path is:

**Google Cloud Run**

Because it:
- fits the current Dockerized architecture directly
- is faster to demonstrate
- is simpler to operate
- supports a clean public ML API demo
- requires less infrastructure management than EB

Elastic Beanstalk remains a valid secondary path for AWS-oriented extension.

---

## 7. Suggested CI/CD Baseline

Regardless of cloud target, the minimum recommended pipeline is:

1. run `pytest -q`
2. build Docker image
3. publish image or source bundle
4. deploy to target environment
5. validate `/health` after deployment

---

## 8. Final Practical Deliverables

By the end of deployment work, the desired artifacts are:

- public endpoint or fully documented simulated deployment
- deployment architecture diagram
- deployment narrative in the report
- monitoring and drift concept subsection
- reproducible deployment guide

---

## 9. Official Documentation References

Use the following official sources when implementing or validating the deployment process:

### Google Cloud
- Cloud Run container deployment documentation
- Cloud Run quickstart for deploying containers
- Artifact Registry integration with Cloud Run
- Cloud Build container build workflow

### AWS
- Elastic Beanstalk Docker deployment guide
- Elastic Beanstalk Docker quickstart
- Elastic Beanstalk Docker image preparation guide
- Elastic Beanstalk Buildfile / Procfile documentation
- Elastic Beanstalk platform hooks documentation

These should be used as the primary reference points for any future re-deployment or CI/CD expansion.
