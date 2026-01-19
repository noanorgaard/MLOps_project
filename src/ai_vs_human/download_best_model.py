"""Download the best model from W&B sweep based on validation accuracy."""

import os
from pathlib import Path

import torch
import wandb
from dotenv import load_dotenv

from ai_vs_human.model import get_model

# Load environment variables
load_dotenv()


def download_best_model_from_sweep(
    entity: str | None = None,
    project: str = "MLOps_project",
    sweep_id: str | None = None,
    metric: str = "train/epoch_acc",
    output_dir: str = "models",
) -> Path:
    """
    Download the best model from a W&B sweep.

    Args:
        entity: W&B entity (username or team). If None, uses env variable.
        project: W&B project name
        sweep_id: Sweep ID. If None, finds the most recent sweep.
        metric: Metric to optimize (should match sweep config)
        output_dir: Directory to save the model

    Returns:
        Path to the downloaded model checkpoint
    """
    # Initialize wandb API
    api = wandb.Api()

    # Get entity from environment if not provided
    if entity is None:
        entity = os.getenv("WANDB_ENTITY", "thordeibert-danmarks-tekniske-universitet-dtu")

    print(f"Searching for best model in {entity}/{project}")

    # Get sweep
    if sweep_id:
        sweep_path = f"{entity}/{project}/{sweep_id}"
        sweep = api.sweep(sweep_path)
        print(f"Using sweep: {sweep_id}")
    else:
        # Find the most recent sweep
        sweeps = api.project(f"{entity}/{project}").sweeps()
        if not sweeps:
            raise ValueError(f"No sweeps found in {entity}/{project}")
        sweep = sweeps[0]
        sweep_id = sweep.id
        print(f"Using most recent sweep: {sweep_id}")

    # Get all runs from the sweep
    runs = sweep.runs

    if not runs:
        raise ValueError(f"No runs found in sweep {sweep_id}")

    print(f"Found {len(runs)} runs in sweep")

    # Find the best run based on metric
    best_run = None
    best_metric_value = float("-inf")

    for run in runs:
        if run.state != "finished":
            continue

        # Get the metric value
        summary = run.summary
        if metric in summary:
            metric_value = summary[metric]
            if metric_value > best_metric_value:
                best_metric_value = metric_value
                best_run = run

    if best_run is None:
        raise ValueError(f"No finished runs with metric '{metric}' found")

    print(f"\nBest run: {best_run.name} (ID: {best_run.id})")
    print(f"Best {metric}: {best_metric_value:.4f}")
    print(f"Config: {best_run.config}")

    # Download the model artifact from the best run
    artifacts = best_run.logged_artifacts()
    model_artifact = None

    for artifact in artifacts:
        if artifact.type == "model":
            model_artifact = artifact
            break

    if model_artifact is None:
        raise ValueError(f"No model artifact found in run {best_run.id}")

    print(f"\nDownloading artifact: {model_artifact.name}")
    artifact_dir = model_artifact.download(root=output_dir)

    # Find the checkpoint file
    checkpoint_files = list(Path(artifact_dir).glob("*.pth"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No .pth file found in {artifact_dir}")

    checkpoint_path = checkpoint_files[0]
    print(f"Model saved to: {checkpoint_path}")

    # Optionally copy to models/best_checkpoint.pth
    output_path = Path(output_dir) / "best_checkpoint.pth"
    torch.save(torch.load(checkpoint_path, weights_only=True), output_path)
    print(f"Also saved as: {output_path}")

    return output_path


def verify_model(checkpoint_path: Path) -> None:
    """Verify that the downloaded model loads correctly."""
    print(f"\nVerifying model from {checkpoint_path}...")

    model = get_model()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()

    print("✓ Model loaded successfully")
    print(f"✓ Model has {sum(p.numel() for p in model.parameters())} parameters")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download best model from W&B sweep")
    parser.add_argument("--entity", type=str, help="W&B entity (uses env variable if not provided)")
    parser.add_argument("--project", type=str, default="MLOps_project", help="W&B project name")
    parser.add_argument("--sweep-id", type=str, help="Sweep ID (uses most recent if not provided)")
    parser.add_argument(
        "--metric", type=str, default="train/epoch_acc", help="Metric to optimize (default: train/epoch_acc)"
    )
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory for model")

    args = parser.parse_args()

    try:
        checkpoint_path = download_best_model_from_sweep(
            entity=args.entity,
            project=args.project,
            sweep_id=args.sweep_id,
            metric=args.metric,
            output_dir=args.output_dir,
        )

        verify_model(checkpoint_path)

        print(f"\n✅ Best model downloaded successfully to {checkpoint_path}")

    except Exception as e:
        print(f"❌ Error: {e}")
        raise
