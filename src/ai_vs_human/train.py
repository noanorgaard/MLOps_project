import torch
import wandb
from torch.utils.data import DataLoader
from ai_vs_human.model import get_model
from ai_vs_human.data import MyDataset
import argparse

print(f"dataset found at {MyDataset().processed_dir}")


def train():
    # Hyperparameters
    parser = argparse.ArgumentParser(description="Train AI vs Human classifier")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    lr = args.lr
    batch_size = args.batch_size
    epochs = args.epochs

    dataset = MyDataset()  # <--- put something meaningful here
    trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = get_model()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model = model.to(device)

    # Only parameters that require gradients are optimized
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # Loss function for binary classification
    criterion = torch.nn.BCEWithLogitsLoss()

    print(f"Model: {model.__class__.__name__}, Device: {device}")
    print("Initiating training loop...")

    model.train()  # Set mode to training (enables dropout/batchnorm)
    wandb.init(project="MLOps_project", name="cloud-run-001",config={
        "lr": lr,
        "batch_size": batch_size,
        "epochs": epochs,},)
    artifact = wandb.Artifact(name="ai_vs_human_model",type="model",metadata={
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,},)

    for epoch in range(epochs):
        print(f"Starting epoch {epoch+1}/{epochs}...")
        running_loss = 0.0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            # Squeeze output from [Batch, 1] to [Batch] to match labels
            loss = criterion(outputs.squeeze(), labels.float())

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(trainloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "loss": epoch_loss})
    torch.save(model.state_dict(), "models/checkpoint.pth")
    print("Training complete. Model saved.")
    artifact.add_file("models/checkpoint.pth")
    wandb.log_artifact(artifact)
    wandb.finish()


if __name__ == "__main__":
    train()
