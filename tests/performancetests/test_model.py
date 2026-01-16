from ai_vs_human.model import get_model
import wandb
import os
import time
from dotenv import load_dotenv
import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

api = wandb.Api()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def load_model():
    # Verify WANDB_API_KEY is set
    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        raise ValueError("WANDB_API_KEY environment variable is not set")

    logger.info("WANDB_API_KEY found in environment")

    wandb.login(key=api_key, relogin=True)

    # W&B configuration from environment variables
    # Use non-empty fallbacks even if env vars are defined as empty strings in CI
    entity = os.getenv("WANDB_ENTITY") or "thordeibert-danmarks-tekniske-universitet-dtu"
    project = os.getenv("WANDB_PROJECT") or "MLOps_project"
    artifact_name = os.getenv("WANDB_ARTIFACT") or "ai_vs_human_model:latest"

    # Construct full artifact path
    artifact_path = f"{entity}/{project}/{artifact_name}"

    logger.info(f"Loading model artifact from W&B: {artifact_path}")

    artifact = api.artifact(artifact_path, type="model")
    artifact_dir = artifact.download()

    model_path = os.path.join(artifact_dir, "checkpoint.pth")
    model = get_model()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    model.eval()

    return model


# This test will run, whenever at new model with alias 'latest' is pushed to W&B
# model will test on model with that alias
def test_model_speed():
    model = load_model()
    start = time.time()
    for _ in range(100):
        model(torch.rand(1, 3, 224, 224))
    end = time.time()
    assert end - start < 5
