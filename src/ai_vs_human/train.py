import torch
from torch.utils.data import DataLoader
from ai_vs_human.model import get_model
from ai_vs_human.data import MyDataset

import wandb
from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Adjusted to point to project root


def _wandb_setup() -> None:
    """
    Make W&B work from any profile (Docker/CI/local) by:
    - Using WANDB_API_KEY if provided (non-interactive)
    - Writing wandb files into the repo (not ~/.config, ~/.cache)
    """

    wandb_dir = PROJECT_ROOT / ".wandb" / "wandb"
    cache_dir = PROJECT_ROOT / ".wandb_cache"
    config_dir = PROJECT_ROOT / ".wandb_config"

    wandb_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    # Force W&B to use these locations (no setdefault)
    os.environ["WANDB_DIR"] = str(wandb_dir)
    os.environ["WANDB_CACHE_DIR"] = str(cache_dir)
    os.environ["WANDB_CONFIG_DIR"] = str(config_dir)

    # Force online so it fails loudly if auth/network is wrong
    os.environ.setdefault("WANDB_MODE", "online")

    print("WANDB_DIR in code =", os.environ.get("WANDB_DIR"))
    print("WANDB_MODE in code =", os.environ.get("WANDB_MODE"))

    api_key = os.getenv("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key, relogin=True)
    else:
        wandb.login()


# Accuracy calculation
def _binary_accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Accuracy for BCEWithLogitsLoss (binary classification)."""
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).long()
    return (preds == labels.long()).float().mean().item()


def train(config: dict | None = None):
    """Train the model with hyperparameters from config or defaults.

    Args:
        config: Optional dict with hyperparameters. If None, uses defaults.
                When run in a sweep, wandb.init() provides wandb.config.
    """
    (PROJECT_ROOT / "models").mkdir(exist_ok=True)
    _wandb_setup()

    from ai_vs_human.data import prepare_data

    from pathlib import Path

    prepare_data(raw_dir=Path("data/raw"), processed_dir=Path("data/processed"))

    # Initialize the W&B run (sweep or standalone)
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "MLOps_project"),
        entity=os.getenv("WANDB_ENTITY"),
        config=config or {"lr": 1e-4, "batch_size": 64, "epochs": 2},
    )

    # Get hyperparameters from wandb.config (works for both sweeps and regular runs)
    cfg = wandb.config
    lr = cfg.lr
    batch_size = cfg.batch_size
    epochs = cfg.epochs

    dataset = MyDataset(raw_dir="data/raw", processed_dir="data/processed", train=True)

    print(f"dataset found at {dataset.processed_dir}")
    print(f"Training with lr={lr}, batch_size={batch_size}, epochs={epochs}")
    trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = get_model()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model = model.to(device)

    # Tracking weights/gradient over time
    wandb.watch(model, log="all", log_freq=100)

    # Only parameters that require gradients are optimized
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # Loss function for binary classification
    criterion = torch.nn.BCEWithLogitsLoss()

    global_step = 0

    print(f"Model: {model.__class__.__name__}, Device: {device}")
    print("Initiating training loop...")

    try:
        for epoch in range(epochs):
            model.train()  # Set mode to training (enables dropout/batchnorm)
            print(f"Starting epoch {epoch+1}/{epochs}...")
            running_loss = 0.0
            running_acc = 0.0
            n_batches = 0

            for i, (images, labels) in enumerate(trainloader):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = model(images)

                # Squeeze output to match labels
                loss = criterion(outputs.squeeze(), labels.float())

                loss.backward()
                optimizer.step()

                # Metrics
                with torch.no_grad():
                    acc = _binary_accuracy_from_logits(outputs.detach(), labels)

                # Per-step logging
                wandb.log(
                    {"train/loss": loss.item(), "train/acc": acc, "epoch": epoch + 1},
                    step=global_step,
                )
                global_step += 1

                running_loss += loss.item()
                running_acc += acc
                n_batches += 1

                # Images + gradient histogram
                if i % 100 == 0:
                    print(f"Epoch {epoch}, iter {i}, loss: {loss.item():.4f}")

                    try:
                        imgs = [wandb.Image(img.detach().cpu()) for img in images[:5]]
                        wandb.log({"images": imgs}, step=global_step)
                    except Exception:
                        pass

                    grads = torch.cat(
                        [p.grad.flatten() for p in model.parameters() if p.grad is not None],
                        0,
                    )
                    wandb.log({"gradients": wandb.Histogram(grads)}, step=global_step)

            epoch_loss = running_loss / max(n_batches, 1)
            epoch_acc = running_acc / max(n_batches, 1)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
            wandb.log(
                {"train/epoch_loss": epoch_loss, "train/epoch_acc": epoch_acc},
                step=global_step,
            )

        # Save + artifact
        ckpt_path = PROJECT_ROOT / "models" / "checkpoint.pth"
        torch.save(model.state_dict(), ckpt_path)

        artifact = wandb.Artifact(
            name="ai_vs_human_model",
            type="model",
            description="Model trained on ai_vs_human dataset",
        )
        artifact.add_file(str(ckpt_path))
        run.log_artifact(artifact)

        ENTITY = "MLOps model"  # Registry name
        COLLECTION_NAME = "ai_vs_human_model"  # Collection name

        # Directly linkning the artifact to the model registry
        run.link_artifact(
            artifact,
            f"wandb-registry-{ENTITY}/{COLLECTION_NAME}",
            aliases=["latest"],
        )

    finally:
        wandb.finish()

    # Final model save
    final_path = PROJECT_ROOT / "models" / "final_model.pth"
    torch.save(model.state_dict(), final_path)
    print("Training complete. Model Saved.")


if __name__ == "__main__":
    train()
