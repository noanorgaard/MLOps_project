import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import anyio
import pandas as pd
from evidently.legacy.metrics import DataDriftTable
from evidently.legacy.report import Report
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from google.cloud import storage


GCS_BUCKET_NAME = "ai-vs-human-monitoring"
REFERENCE_BLOB = "reference/features.csv"
PREDICTION_PREFIX = "prediction/"

REPORT_PATH = "data_drift_report.html"
PRED_CACHE_DIR = Path("./pred_cache")


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
    global training_reference, feature_columns

    df = download_reference_csv(GCS_BUCKET_NAME, REFERENCE_BLOB)

    non_feature_cols = {"index", "timestamp"}
    cols = [c for c in df.columns if c not in non_feature_cols]

    training_reference = df[cols].copy()
    feature_columns = cols

    yield

    del training_reference, feature_columns


app = FastAPI(lifespan=lifespan)


@app.get("/report", response_class=HTMLResponse)
async def get_report(n: int = 50):
    """Generate and return the report."""
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
