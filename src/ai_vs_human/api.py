"""FastAPI application for AI vs Human image classification inference.

Loads model from W&B artifacts at runtime using WANDB_API_KEY environment variable.
"""

import io
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
import torch.nn as nn
import wandb
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from ai_vs_human.model import get_model

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan handler to load model on startup."""
    await load_model_from_wandb()
    yield


app = FastAPI(
    title="AI vs Human Classifier",
    description="API for classifying if an image is AI-generated or human-made",
    version="1.0.0",
    lifespan=lifespan,
)

# Global model state
model: nn.Module | None = None
device: torch.device | None = None

# ==================== Prometheus Metrics ====================
# Request metrics
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total number of API requests",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "api_request_duration_seconds",
    "Request latency in seconds",
    ["method", "endpoint"],
)

# Prediction metrics
PREDICTION_COUNT = Counter(
    "predictions_total",
    "Total number of predictions made",
    ["prediction_class"],
)

PREDICTION_CONFIDENCE = Histogram(
    "prediction_confidence",
    "Distribution of prediction confidence scores",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
)

# Error metrics
ERROR_COUNT = Counter(
    "api_errors_total",
    "Total number of API errors",
    ["endpoint", "error_type"],
)

# Model state
MODEL_LOADED = Gauge(
    "model_loaded",
    "Whether the model is successfully loaded (1=loaded, 0=not loaded)",
)


def _download_best_sweep_artifact(
    entity: str,
    project: str,
    sweep_id: str,
    metric: str,
    artifact_type: str = "model",
) -> Path:
    """Download the best model artifact from a W&B sweep using the given metric."""

    api = wandb.Api()
    sweep_path = f"{entity}/{project}/{sweep_id}"
    sweep = api.sweep(sweep_path)

    best_run = None
    best_value = float("-inf")

    for run in sweep.runs:
        if run.state != "finished":
            continue

        value = run.summary.get(metric)
        if value is None:
            continue

        if value > best_value:
            best_value = value
            best_run = run

    if best_run is None:
        raise ValueError(f"No finished runs with metric '{metric}' found in sweep {sweep_id}")

    logger.info(
        "Best run from sweep %s: %s (%.4f)",
        sweep_id,
        best_run.id,
        best_value,
    )

    artifacts = [artifact for artifact in best_run.logged_artifacts() if artifact.type == artifact_type]
    if not artifacts:
        raise ValueError(f"No {artifact_type} artifact found in best run {best_run.id}")

    artifact = artifacts[0]
    artifact_dir = artifact.download()
    logger.info("Downloaded artifact %s to %s", artifact.name, artifact_dir)
    return Path(artifact_dir)


# Middleware to track all requests
@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    """Middleware to instrument all HTTP requests with Prometheus metrics."""
    start_time = time.time()

    try:
        response = await call_next(request)
        duration = time.time() - start_time

        # Track request count and latency
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code,
        ).inc()

        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=request.url.path,
        ).observe(duration)

        return response

    except Exception:
        duration = time.time() - start_time

        # Track failed requests
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code=500,
        ).inc()

        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=request.url.path,
        ).observe(duration)

        raise


@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


async def load_model_from_wandb() -> None:
    """Load model from W&B artifact using WANDB_API_KEY from environment."""
    global model, device

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    logger.info(f"Using device: {device}")

    try:
        # Verify WANDB_API_KEY is set
        api_key = os.getenv("WANDB_API_KEY")
        if not api_key:
            raise ValueError("WANDB_API_KEY environment variable is not set")

        logger.info("WANDB_API_KEY found in environment")

        # W&B configuration from environment variables
        entity = os.getenv("WANDB_ENTITY", "thordeibert-danmarks-tekniske-universitet-dtu")
        project = os.getenv("WANDB_PROJECT", "MLOps_project")
        artifact_name = os.getenv("WANDB_ARTIFACT", "ai_vs_human_model:latest")
        sweep_id = os.getenv("WANDB_SWEEP_ID")
        sweep_metric = os.getenv("WANDB_SWEEP_METRIC", "train/epoch_acc")

        # Download model: prefer best from sweep if sweep id provided, else use explicit artifact
        if sweep_id:
            logger.info("Loading best model from sweep %s using metric '%s'", sweep_id, sweep_metric)
            artifact_dir = _download_best_sweep_artifact(
                entity=entity,
                project=project,
                sweep_id=sweep_id,
                metric=sweep_metric,
            )
        else:
            artifact_path = f"{entity}/{project}/{artifact_name}"
            logger.info(f"Loading model artifact from W&B: {artifact_path}")

            # Initialize wandb (automatically uses WANDB_API_KEY from environment)
            run = wandb.init(
                project=project,
                entity=entity,
                job_type="inference",
            )

            logger.info("Successfully authenticated with W&B")

            # Download artifact to temporary directory
            logger.info("Downloading artifact...")
            artifact = run.use_artifact(artifact_path, type="model")
            artifact_dir = Path(artifact.download())

        logger.info(f"Artifact downloaded to: {artifact_dir}")

        # Find model file in artifact
        model_files = list(artifact_dir.glob("*.pth"))
        if not model_files:
            raise FileNotFoundError(f"No .pth model file found in artifact at {artifact_dir}")

        model_path = model_files[0]
        logger.info(f"Loading model from: {model_path}")

        # Initialize and load model
        model = get_model()
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
        model = model.to(device)
        model.eval()

        if sweep_id is None:
            wandb.finish()

        logger.info("Model loaded successfully from W&B artifact")
        MODEL_LOADED.set(1)  # Set metric to indicate model is loaded

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        logger.warning("WANDB_API_KEY not set - cannot authenticate with W&B")
        MODEL_LOADED.set(0)  # Model failed to load
        raise
    except Exception as e:
        logger.error(f"Failed to load model from W&B: {e}")
        logger.warning("Starting API without loaded model - predictions will fail")
        MODEL_LOADED.set(0)  # Model failed to load
        raise


@app.post("/predict")
async def predict(file: Annotated[UploadFile, File()]) -> JSONResponse:
    """
    Classify an image as AI-generated or human-made.

    Args:
        file: Image file to classify (JPG, PNG, etc.)

    Returns:
        JSONResponse with prediction and confidence score

    Raises:
        HTTPException: If model not loaded or file processing fails
    """
    if model is None or device is None:
        logger.error("Model not loaded - cannot make predictions")
        raise HTTPException(
            status_code=503, detail="Model not loaded. Check server logs for W&B artifact loading errors."
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Invalid file type: {file.content_type}. Must be an image.")

    try:
        # Read and preprocess image (in-memory, no disk I/O)
        image_data = await file.read()
        img = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Resize and normalize (same as training)
        img = img.resize((224, 224), Image.LANCZOS)  # type: ignore[attr-defined]
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = np.clip(arr, 0.0, 1.0)
        arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW

        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(arr).float().unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            logit = model(image_tensor).squeeze()
            confidence = torch.sigmoid(logit).item()
            prediction = "AI-generated" if confidence > 0.5 else "Human-made"

        # Record prediction metrics
        PREDICTION_COUNT.labels(prediction_class=prediction).inc()
        PREDICTION_CONFIDENCE.observe(confidence)

        logger.info(f"Prediction: {prediction} (confidence: {confidence:.4f})")

        return JSONResponse(
            {
                "prediction": prediction,
                "confidence": confidence,
                "logit": logit.item(),
            }
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        ERROR_COUNT.labels(endpoint="/predict", error_type=type(e).__name__).inc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/health")
async def health() -> JSONResponse:
    """Health check endpoint."""
    return JSONResponse(
        {
            "status": "healthy" if model is not None else "model_not_loaded",
            "model_loaded": model is not None,
            "device": str(device) if device else "not_initialized",
        }
    )


@app.get("/")
async def root() -> JSONResponse:
    """Root endpoint with API information."""
    return JSONResponse(
        {
            "name": "AI vs Human Classifier",
            "version": "1.0.0",
            "status": "running",
            "endpoints": {
                "predict": "/predict (POST) - Upload image for classification",
                "health": "/health (GET) - Check API health",
                "metrics": "/metrics (GET) - Prometheus metrics for monitoring",
                "docs": "/docs (GET) - Interactive API documentation",
            },
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
