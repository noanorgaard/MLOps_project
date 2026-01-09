import torch
from torch.utils.data import DataLoader
from ai_vs_human.model import get_model
from ai_vs_human.data import MyDataset

print(f"dataset found at {MyDataset().processed_dir}")

def train():
    # Hyperparameters
    lr = 1e-4
    batch_size = 64
    epochs = 2

    dataset = MyDataset() # <--- put something meaningful here 
    trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = get_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)

    # Only parameters that require gradients are optimized
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr
    )
    
    # Loss function for binary classification
    criterion = torch.nn.BCEWithLogitsLoss()

    print(f"Model: {model.__class__.__name__}, Device: {device}")
    print("Initiating training loop...")

    model.train() # Set mode to training (enables dropout/batchnorm)

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

    torch.save(model.state_dict(), "models/checkpoint.pth")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    train()