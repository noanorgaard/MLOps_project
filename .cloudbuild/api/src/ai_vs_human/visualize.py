import torch
from torch.utils.data import DataLoader
from pathlib import Path
from ai_vs_human.model import get_model
from ai_vs_human.data import MyDataset
import matplotlib.pyplot as plt
import numpy as np


def visualize():
    dataset = MyDataset(Path("data/processed"))
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    model = get_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model.load_state_dict(torch.load("models/checkpoint.pth", map_location=device))
    model.eval()
    print(f"Model: {model.__class__.__name__}, Device: {device}")
    print("Starting visualization...")

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            probs = torch.sigmoid(outputs).view(-1).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            images = images.cpu().numpy()
            labels = labels.cpu().numpy()

            n = min(10, images.shape[0])
            fig, axes = plt.subplots(2, 5, figsize=(15, 6))
            axes = axes.flatten()

            for i in range(n):
                img = np.transpose(images[i], (1, 2, 0))
                axes[i].imshow(img)
                axes[i].set_title(
                    f"Pred: {'AI' if preds[i] else 'Human'} ({probs[i]:.2f})\n"
                    f"True: {'AI' if labels[i] else 'Human'}"
                )
                axes[i].axis("off")

            plt.tight_layout()
            plt.savefig("reports/figures/visualization.png")

            break


if __name__ == "__main__":
    visualize()
