import torch
from torch.utils.data import DataLoader
from ai_vs_human.model import get_model
from ai_vs_human.data import MyDataset

import wandb
from pathlib import Path



print(f"dataset found at {MyDataset().processed_dir}")


 # Accuracy calculation
def _binary_accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Accuracy for BCEWithLogitsLoss (binary classification)."""
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).long()
    return (preds == labels.long()).float().mean().item()

def train():
    # Hyperparameters
    lr = 1e-4
    batch_size = 64
    epochs = 2

    Path("models").mkdir(exist_ok=True)

    dataset = MyDataset() # <--- put something meaningful here 
    trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = get_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)

    # Initialize the W&B run
    run = wandb.init(
    project="MLOps_project",
    config={"lr": lr, "batch_size": batch_size, "epochs": epochs, "device": str(device)},
    )

    # Tracking weights/gradient over time
    wandb.watch(model, log="all", log_freq=100)

    # Only parameters that require gradients are optimized
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr
    )
    
    # Loss function for binary classification
    criterion = torch.nn.BCEWithLogitsLoss()

    global_step = 0

    print(f"Model: {model.__class__.__name__}, Device: {device}")
    print("Initiating training loop...")

    try:
        for epoch in range(epochs):
            model.train() # Set mode to training (enables dropout/batchnorm)
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
                        images_log = wandb.Image(
                            images[:5].detach().cpu(), caption="Input images"
                        )
                        wandb.log({"images": images_log})
                    except Exception:
                        pass

                    grads = torch.cat(
                        [p.grad.flatten() for p in model.parameters() if p.grad is not None],
                        0,
                    )
                    wandb.log({"gradients": wandb.Histogram(grads)})

            epoch_loss = running_loss / max(n_batches, 1)
            epoch_acc = running_acc / max(n_batches, 1)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
            wandb.log(
                {"train/epoch_loss": epoch_loss, "train/epoch_acc": epoch_acc}
            )

        # Save + artifact
        ckpt_path = "models/checkpoint.pth"
        torch.save(model.state_dict(), ckpt_path)

        artifact = wandb.Artifact(
            name="ai_vs_human_model",
            type="model",
            description="Model trained on ai_vs_human dataset",
        )
        artifact.add_file(ckpt_path)
        run.log_artifact(artifact)

    finally:
        wandb.finish()

    torch.save(model.state_dict(), "models/final_model.pth")
    print("Training complete. Model Saved.")

if __name__ == "__main__":
    train()