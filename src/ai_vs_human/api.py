"""FastAPI application for AI vs Human image classification inference.

Loads model from W&B artifacts at runtime using WANDB_API_KEY environment variable.
"""

import io
import logging
import os
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
import torch.nn as nn
import wandb
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from ai_vs_human.model import get_model

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI vs Human Classifier",
    description="API for classifying if an image is AI-generated or human-made",
    version="1.0.0",
)

# Global model state
model: nn.Module | None = None
device: torch.device | None = None


@app.on_event("startup")
async def load_model_from_wandb() -> None:
    """Load model from W&B artifact using WANDB_API_KEY from environment."""
    global model, device

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
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
        
        # Construct full artifact path
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
        artifact_dir = artifact.download()
        
        logger.info(f"Artifact downloaded to: {artifact_dir}")
        
        # Find model file in artifact
        model_files = list(Path(artifact_dir).glob("*.pth"))
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
        
        wandb.finish()
        
        logger.info("Model loaded successfully from W&B artifact")

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        logger.warning("WANDB_API_KEY not set - cannot authenticate with W&B")
        raise
    except Exception as e:
        logger.error(f"Failed to load model from W&B: {e}")
        logger.warning("Starting API without loaded model - predictions will fail")
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
            status_code=503,
            detail="Model not loaded. Check server logs for W&B artifact loading errors."
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Must be an image."
        )

    try:
        # Read and preprocess image (in-memory, no disk I/O)
        image_data = await file.read()
        img = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Resize and normalize (same as training)
        img = img.resize((224, 224), Image.LANCZOS)
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

        logger.info(f"Prediction: {prediction} (confidence: {confidence:.4f})")

        return JSONResponse({
            "prediction": prediction,
            "confidence": confidence,
            "logit": logit.item(),
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/health")
async def health() -> JSONResponse:
    """Health check endpoint."""
    return JSONResponse({
        "status": "healthy" if model is not None else "model_not_loaded",
        "model_loaded": model is not None,
        "device": str(device) if device else "not_initialized",
    })


@app.get("/")
async def root() -> JSONResponse:
    """Root endpoint with API information."""
    return JSONResponse({
        "name": "AI vs Human Classifier",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "predict": "/predict (POST) - Upload image for classification",
            "health": "/health (GET) - Check API health",
            "docs": "/docs (GET) - Interactive API documentation",
        },
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
