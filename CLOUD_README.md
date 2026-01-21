# Cloud Deployment Runbook (DTU MLOps)

This document describes what we deployed to GCP, where artifacts live, and how to reproduce the setup.

> Scope: Cloud Run inference + Cloud Run drift detection (M27), Artifact Registry images, GCS monitoring bucket, and the
> minimal Cloud Build contexts that keep uploads small.

## TL;DR

We deployed two services to **Cloud Run** in **project `dtumlops-484111`** (region `europe-west1`):

- **Inference API** (`ai-vs-human-api`): serves `/predict` and logs lightweight feature + prediction records to GCS.
- **Drift API** (`ai-vs-human-drift`): serves `/report` which compares recent prediction logs against a training
  reference and returns an Evidently HTML report.

Monitoring bucket:

- `gs://ai-vs-human-drift-monitoring-mads/`
  - Reference: `reference/features.csv`
  - Prediction logs: `prediction/pred_*.json`

## What we deployed

### 1) Drift detection API (M27)

- **Cloud Run service name:** `ai-vs-human-drift`
- **Endpoints:**
  - `GET /health` – shows config and whether reference is loaded.
  - `GET /report?n=50` – generates an Evidently HTML report (can be slow for large `n`).

**Important behavior**

- The drift service reads the reference CSV at startup.
- If reference cannot be loaded (missing file, wrong IAM), the service stays up and `/health` reports `degraded`.
- `/report` returns `503` if reference is unavailable.

### 2) Inference API

- **Cloud Run service name:** `ai-vs-human-api`
- **Endpoints:**
  - `POST /predict` – runs inference on an uploaded image and logs feature stats to GCS.
  - `GET /health` – service health.

**Important behavior**

- The inference service loads the model from **Weights & Biases** (W&B) artifacts.
- Model loading is **lazy**: the service starts quickly, and loads the model on the first request to `/predict`.
- Prediction logging happens asynchronously via a background task.

## Where things are stored

### Container images (Artifact Registry)

- Artifact Registry location: `europe-west1`
- Repository: `container-registry`

Images:

- Drift:
  - `europe-west1-docker.pkg.dev/dtumlops-484111/container-registry/drift-api:latest`
- Inference:
  - `europe-west1-docker.pkg.dev/dtumlops-484111/container-registry/ai-vs-human-api:latest`

### Monitoring data (GCS)

Bucket:

- `gs://ai-vs-human-drift-monitoring-mads/`

Contents:

- Training reference features:
  - `gs://ai-vs-human-drift-monitoring-mads/reference/features.csv`
- Prediction feature logs:
  - `gs://ai-vs-human-drift-monitoring-mads/prediction/pred_*.json`

### Model storage (W&B)

The inference API pulls the model from W&B.

- Requires `WANDB_API_KEY` at runtime.
- Uses `WANDB_ENTITY`, `WANDB_PROJECT`, and optionally `WANDB_ARTIFACT` / `WANDB_SWEEP_ID`.

Note: we recommend storing `WANDB_API_KEY` in **Secret Manager** and mounting it into Cloud Run.

## Repo changes that enabled this

### Minimal Cloud Build contexts (key improvement)

To avoid uploading large datasets/models to Cloud Build, we use minimal contexts under `.cloudbuild/`:

- `.cloudbuild/drift/`
  - `Dockerfile`
  - `cloudbuild.yaml`
- `.cloudbuild/api/`
  - `Dockerfile`
  - `cloudbuild.yaml`

A helper script prepares these contexts by copying only the needed sources and lockfiles:

- `scripts/prepare_cloudbuild_context.py`

### Invoke tasks

We added simple invoke tasks to build/push images using minimal contexts:

- `uv run invoke cloudbuild-drift`
- `uv run invoke cloudbuild-api`

## Reproducible setup: Commands

### 0) Configure gcloud

```bash
gcloud config set project dtumlops-484111
gcloud config set run/region europe-west1
```

### 1) Create (or use) monitoring bucket

We used:

```bash
gsutil mb -l europe-west1 gs://ai-vs-human-drift-monitoring-mads
```

### 2) Upload reference features

This computes simple image statistics on the training dataset and uploads the CSV to:

- `gs://$GCS_BUCKET_NAME/reference/features.csv`

Command:

```bash
cd /home/madsl/ML_OPS/MLOps_project \
  && GCS_BUCKET_NAME=ai-vs-human-drift-monitoring-mads uv run invoke upload-reference-features
```

Verify:

```bash
gcloud storage ls gs://ai-vs-human-drift-monitoring-mads/reference/
```

### 3) IAM: Allow Cloud Run service account to use the bucket

We used this Cloud Run service account:

- `ai-vs-human-runner@dtumlops-484111.iam.gserviceaccount.com`

Grant read access (drift needs this):

```bash
gcloud storage buckets add-iam-policy-binding gs://ai-vs-human-drift-monitoring-mads \
  --member="serviceAccount:ai-vs-human-runner@dtumlops-484111.iam.gserviceaccount.com" \
  --role="roles/storage.objectViewer"
```

Grant write access (inference needs this to write prediction logs):

```bash
gcloud storage buckets add-iam-policy-binding gs://ai-vs-human-drift-monitoring-mads \
  --member="serviceAccount:ai-vs-human-runner@dtumlops-484111.iam.gserviceaccount.com" \
  --role="roles/storage.objectCreator"
```

### 4) Build + push images

Drift image:

```bash
uv run invoke cloudbuild-drift
```

Inference image:

```bash
uv run invoke cloudbuild-api
```

### 5) Deploy drift API (Cloud Run)

```bash
gcloud run deploy ai-vs-human-drift \
  --image europe-west1-docker.pkg.dev/dtumlops-484111/container-registry/drift-api:latest \
  --region europe-west1 \
  --service-account ai-vs-human-runner@dtumlops-484111.iam.gserviceaccount.com \
  --allow-unauthenticated \
  --memory 2Gi \
  --timeout 900 \
  --set-env-vars GCS_BUCKET_NAME=ai-vs-human-drift-monitoring-mads,REFERENCE_BLOB=reference/features.csv,PREDICTION_PREFIX=prediction/
```

### 6) Deploy inference API (Cloud Run)

The inference API needs W&B credentials.

Recommended: Secret Manager

```bash
# Create secret (one-time)
gcloud secrets create wandb-api-key --replication-policy="automatic"

# Add a secret version (paste key, then Ctrl-D)
gcloud secrets versions add wandb-api-key --data-file=-

# Allow Cloud Run SA to access the secret
gcloud secrets add-iam-policy-binding wandb-api-key \
  --member="serviceAccount:ai-vs-human-runner@dtumlops-484111.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

# Deploy
gcloud run deploy ai-vs-human-api \
  --image europe-west1-docker.pkg.dev/dtumlops-484111/container-registry/ai-vs-human-api:latest \
  --region europe-west1 \
  --service-account ai-vs-human-runner@dtumlops-484111.iam.gserviceaccount.com \
  --allow-unauthenticated \
  --memory 2Gi \
  --timeout 300 \
  --set-secrets WANDB_API_KEY=wandb-api-key:latest \
  --set-env-vars GCS_BUCKET_NAME=ai-vs-human-drift-monitoring-mads,PREDICTION_PREFIX=prediction/,WANDB_ENTITY=thordeibert-danmarks-tekniske-universitet-dtu,WANDB_PROJECT=MLOps_project
```

## How to generate prediction logs (make drift non-empty)

Once `ai-vs-human-api` is deployed:

```bash
API_URL="$(gcloud run services describe ai-vs-human-api --region europe-west1 --format='value(status.url)')"
IMG="data/raw/ai/aiart.jpg"

# Send a few prediction requests
for i in {1..10}; do curl -s -X POST "$API_URL/predict" -F "file=@$IMG" >/dev/null; done

# Confirm objects exist in GCS
gcloud storage ls gs://ai-vs-human-drift-monitoring-mads/prediction/ | tail
```

## How to generate drift reports

```bash
DRIFT_URL="$(gcloud run services describe ai-vs-human-drift --region europe-west1 --format='value(status.url)')"

# Report generation can be slow; allow a longer curl timeout
curl -L --max-time 600 "$DRIFT_URL/report?n=50" -o drift_report.html
```

Open the HTML locally (if `xdg-open` is unavailable):

```bash
python -m http.server 8009
# Then open: http://localhost:8009/drift_report.html
```

## Troubleshooting

### Cloud Build uploads are huge

Use the minimal contexts:

- `uv run invoke cloudbuild-drift`
- `uv run invoke cloudbuild-api`

Avoid running `gcloud builds submit .` from the repo root.

### Cloud Run says "failed to start and listen on PORT=8080"

- Ensure your container runs uvicorn on `${PORT}` (Cloud Run sets `PORT=8080`).
- We fixed this by using `sh -c "... --port ${PORT:-8080}"` in Dockerfiles.

### Drift `/report` is slow or hangs

- Try smaller `n`: `/report?n=10`
- Increase Cloud Run resources for drift:

```bash
gcloud run services update ai-vs-human-drift --region europe-west1 --memory 2Gi --timeout 900
```

### Drift returns 500/503

- If `/health` is `degraded`, fix bucket/object/IAM.
- Confirm reference exists:
  - `gcloud storage ls gs://ai-vs-human-drift-monitoring-mads/reference/`
- Confirm prediction logs exist:
  - `gcloud storage ls gs://ai-vs-human-drift-monitoring-mads/prediction/`

### Inference returns "Model not loaded"

- Check Cloud Run logs:

```bash
gcloud run services logs read ai-vs-human-api --region europe-west1 --limit 100
```

Common causes:

- Missing/invalid `WANDB_API_KEY`
- Wrong W&B entity/project/artifact settings
- Not enough memory (increase to `2Gi` or more)

## Security notes

- Do not hardcode API keys in code or paste them into chat logs.
- Prefer Secret Manager for `WANDB_API_KEY`.
- Principle of least privilege:
  - Drift service: `roles/storage.objectViewer`
  - Inference service: `roles/storage.objectCreator` (+ viewer if you later need reads)

## Course alignment (M27)

Checklist items this supports:

- Deployed drift detection API to the cloud (Cloud Run).
- Collected input-output data from deployed application (prediction feature logs in GCS).
- Compared current distribution vs reference distribution and produced a drift report (Evidently HTML).
