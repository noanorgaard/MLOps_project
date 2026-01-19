import numpy as np
import pandas as pd
import torch

from evidently.legacy.metrics import DataDriftTable
from evidently.legacy.report import Report

from transformers import CLIPModel, CLIPProcessor

from ai_vs_human.data import MyDataset


def extract_features(images: torch.Tensor) -> np.ndarray:
    if images.ndim == 3:
        images = images.unsqueeze(1)

    x = images.float().numpy()
    gray = x.mean(axis=1)

    brightness = gray.mean(axis=(1, 2))
    contrast = gray.std(axis=(1, 2))

    gy, gx = np.gradient(gray, axis=(1, 2))
    sharpness = (np.abs(gx) + np.abs(gy)).mean(axis=(1, 2))

    channel_std = x.std(axis=1).mean(axis=(1, 2))

    return np.stack([brightness, contrast, sharpness, channel_std], axis=1)


def collect_images(ds) -> torch.Tensor:
    return torch.stack([ds[i][0] for i in range(len(ds))])


def extract_clip_features(images: torch.Tensor, model, processor, batch_size: int = 16) -> np.ndarray:
    feats = []

    if images.ndim == 3:
        images = images.unsqueeze(1)

    for i in range(0, images.size(0), batch_size):
        batch = images[i : i + batch_size]  # [B, C, H, W]

        if batch.size(1) == 1:
            batch = batch.repeat(1, 3, 1, 1)

        batch = batch.permute(0, 2, 3, 1).numpy()  # [B, H, W, C] float in [0,1]
        inputs = processor(images=list(batch), return_tensors="pt")

        with torch.no_grad():
            emb = model.get_image_features(inputs["pixel_values"])

        feats.append(emb.numpy())

    return np.concatenate(feats, axis=0)


def main():
    train_ds = MyDataset(train=True)
    test_ds = MyDataset(train=False)

    train_images = collect_images(train_ds)
    test_images = collect_images(test_ds)

    train_manual = extract_features(train_images)
    test_manual = extract_features(test_images)

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    train_clip = extract_clip_features(train_images, model, processor)
    test_clip = extract_clip_features(test_images, model, processor)

    train_features = np.concatenate([train_manual, train_clip], axis=1)
    test_features = np.concatenate([test_manual, test_clip], axis=1)

    manual_cols = ["brightness", "contrast", "sharpness", "channel_std"]
    clip_cols = [f"clip_{i}" for i in range(train_clip.shape[1])]
    columns = manual_cols + clip_cols

    reference = pd.DataFrame(train_features, columns=columns)
    current = pd.DataFrame(test_features, columns=columns)

    report = Report(metrics=[DataDriftTable()])
    report.run(reference_data=reference, current_data=current)
    report.save_html("data_drift.html")


if __name__ == "__main__":
    main()
