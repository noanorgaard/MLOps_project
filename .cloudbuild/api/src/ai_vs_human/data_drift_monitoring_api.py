import json
import os
from pathlib import Path
from typing import Any, Dict, List

import anyio
import pandas as pd
from evidently.legacy.metrics import DataDriftTable
from evidently.legacy.report import Report
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.responses import HTMLResponse
from google.cloud import storage


GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "ai-vs-human-monitoring")
REFERENCE_BLOB = os.getenv("REFERENCE_BLOB", "reference/features.csv")
PREDICTION_PREFIX = os.getenv("PREDICTION_PREFIX", "prediction/")

REPORT_PATH = "data_drift_report.html"
PRED_CACHE_DIR = Path("./pred_cache")


training_reference: pd.DataFrame | None = None
feature_columns: List[str] = []
reference_load_error: str | None = None


def download_reference_csv(bucket_name: str, blob_name: str) -> pd.DataFrame:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    if not blob.exists(client):
        raise FileNotFoundError(f"Reference blob not found: gs://{bucket_name}/{blob_name}")

    content = blob.download_as_bytes()
    df = pd.read_csv(pd.io.common.BytesIO(content))
    return df


def list_latest_prediction_blobs(bucket_name: str, prefix: str, n: int) -> List[storage.Blob]:
    """Returns latest N blobs under prefix."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    blobs = list(bucket.list_blobs(prefix=prefix))
    blobs = [b for b in blobs if b.name.lower().endswith(".json")]

    if not blobs:
        return []

    blobs.sort(key=lambda b: (b.updated or 0), reverse=True)
    return blobs[:n]


def download_latest_predictions_to_dir(bucket_name: str, prefix: str, n: int, directory: Path) -> None:
    """Download latest N predictions to a fixed local directory,
    then we glob and load locally.
    """
    directory.mkdir(parents=True, exist_ok=True)

    blobs = list_latest_prediction_blobs(bucket_name, prefix, n=n)
    if not blobs:
        return

    for blob in blobs:
        filename = os.path.basename(blob.name)
        local_path = directory / filename
        blob.download_to_filename(str(local_path))


def load_latest_files(directory: Path, n: int) -> List[Dict[str, Any]]:
    """Load the N latest prediction files from the directory."""
    files = sorted(directory.glob("*.json"), key=os.path.getmtime)
    latest_files = files[-n:]

    records: List[Dict[str, Any]] = []
    for file in latest_files:
        with file.open("r", encoding="utf-8") as f:
            records.append(json.load(f))

    return records


def build_current_dataframe(prediction_records: List[Dict[str, Any]], feature_columns: List[str]) -> pd.DataFrame:
    """Builds the dataframe for drift from prediction json records."""
    rows: List[Dict[str, Any]] = []
    for r in prediction_records:
        feats = r.get("features", {})
        if not isinstance(feats, dict):
            feats = {}

        row = {col: feats.get(col, None) for col in feature_columns}
        rows.append(row)

    return pd.DataFrame(rows, columns=feature_columns)


def run_drift_report(reference: pd.DataFrame, current: pd.DataFrame, out_path: str) -> None:
    report = Report(metrics=[DataDriftTable()])
    report.run(reference_data=reference, current_data=current)
    report.save_html(out_path)


async def lifespan(app: FastAPI):
    """Load reference data once at startup."""
    global training_reference, feature_columns, reference_load_error

    try:
        df = download_reference_csv(GCS_BUCKET_NAME, REFERENCE_BLOB)

        non_feature_cols = {"index", "timestamp"}
        cols = [c for c in df.columns if c not in non_feature_cols]

        training_reference = df[cols].copy()
        feature_columns = cols
        reference_load_error = None
    except Exception as exc:
        # Cloud Run needs the process to start and listen on $PORT.
        # If the bucket/object/IAM is misconfigured, we keep the service up
        # and surface the error on /health and /report.
        training_reference = None
        feature_columns = []
        reference_load_error = f"{type(exc).__name__}: {exc}"

    yield

    training_reference = None
    feature_columns = []
    reference_load_error = None


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health() -> Dict[str, Any]:
    """Health endpoint for Cloud Run."""
    ok = training_reference is not None and not reference_load_error
    return {
        "status": "ok" if ok else "degraded",
        "gcs_bucket": GCS_BUCKET_NAME,
        "reference_blob": REFERENCE_BLOB,
        "prediction_prefix": PREDICTION_PREFIX,
        "reference_loaded": training_reference is not None,
        "reference_error": reference_load_error,
    }


@app.get("/report", response_class=HTMLResponse)
async def get_report(n: int = 50):
    """Generate and return the report."""
    if training_reference is None or reference_load_error:
        raise HTTPException(
            status_code=503,
            detail=(
                "Reference data not available. "
                "Check that the GCS object exists and that the Cloud Run service account has roles/storage.objectViewer. "
                f"Current error: {reference_load_error}"
            ),
        )

    download_latest_predictions_to_dir(
        bucket_name=GCS_BUCKET_NAME,
        prefix=PREDICTION_PREFIX,
        n=n,
        directory=PRED_CACHE_DIR,
    )

    records = load_latest_files(PRED_CACHE_DIR, n=n)

    current = build_current_dataframe(records, feature_columns)

    run_drift_report(training_reference, current, REPORT_PATH)

    async with await anyio.open_file(REPORT_PATH, encoding="utf-8") as f:
        html_content = await f.read()

    return HTMLResponse(content=html_content, status_code=200)
