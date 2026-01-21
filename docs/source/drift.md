## Drift Detection API (M27)

### What it does

The drift service generates an Evidently drift report by comparing:

- **Reference**: `gs://$GCS_BUCKET_NAME/$REFERENCE_BLOB` (computed from training data)
- **Current**: the latest `N` prediction feature logs under `gs://$GCS_BUCKET_NAME/$PREDICTION_PREFIX`

The service exposes:

- `GET /health` – health/config
- `GET /report?n=50` – HTML drift report

### Prerequisites

- A GCS bucket for monitoring (example: `ai-vs-human-monitoring`)
- Cloud Run service account with **Storage Object Viewer** on the bucket

### 1) Upload reference features (one-time, or whenever you retrain)

Locally (requires GCP Application Default Credentials):

```bash
uv run invoke upload-reference-features
```

Environment variables (optional):

- `GCS_BUCKET_NAME` (default: `ai-vs-human-monitoring`)
- `REFERENCE_BLOB` (default: `reference/features.csv`)

### 2) Build + push the drift container (Cloud Build)

Use the minimal Cloud Build context to avoid uploading large datasets:

```bash
gcloud builds submit .cloudbuild/drift --config .cloudbuild/drift/cloudbuild.yaml
```

It pushes:

- `europe-west1-docker.pkg.dev/dtumlops-484111/container-registry/drift-api:latest`

### 3) Deploy to Cloud Run

Example (adjust service name/SA as needed):

```bash
gcloud run deploy ai-vs-human-drift \
  --image europe-west1-docker.pkg.dev/dtumlops-484111/container-registry/drift-api:latest \
  --region europe-west1 \
  --allow-unauthenticated \
  --set-env-vars GCS_BUCKET_NAME=ai-vs-human-monitoring,REFERENCE_BLOB=reference/features.csv,PREDICTION_PREFIX=prediction/
```

### 4) Verify

- Open `/health` to confirm config
- Open `/report?n=50` to generate the drift report

### Notes

- If `/report` is slow, reduce `n`.
- For a more “production” setup, schedule periodic calls to `/report` via Cloud Scheduler and store the report HTML in GCS.

### Making reports non-empty

The drift report compares the training reference to the latest prediction feature logs.
To populate `prediction/`, configure your inference API (Cloud Run) to write logs to the same bucket:

- Set `GCS_BUCKET_NAME` to your monitoring bucket (e.g. `ai-vs-human-drift-monitoring-mads`)
- Set `PREDICTION_PREFIX=prediction/`
- Grant the inference service account `roles/storage.objectCreator` on the bucket
