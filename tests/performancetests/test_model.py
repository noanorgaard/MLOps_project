from ai_vs_human.model import get_model
import wandb
import os
import time
from dotenv import load_dotenv
import logging
from pathlib import Path
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()


def load_model():
    # Verify WANDB_API_KEY is set
    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        raise ValueError("WANDB_API_KEY environment variable is not set")

    logger.info("WANDB_API_KEY found in environment")

    # W&B configuration from environment variables
    entity = os.getenv("WANDB_ENTITY", "thordeibert-danmarks-tekniske-universitet-dtu-org/registry")
    project = os.getenv("WANDB_PROJECT", "MLOps_project")
    artifact_name = os.getenv("WANDB_ARTIFACT", "ai_vs_human_model:latest")

    # Construct full artifact path
    artifact_path = f"{entity}/{project}/{artifact_name}"

    logger.info(f"Loading model artifact from W&B: {artifact_path}")
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )

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

    return model


def test_model_speed():
    model = load_model()
    start = time.time()
    for _ in range(100):
        model(torch.rand(1, 3, 224, 224))
    end = time.time()
    assert end - start < 1
