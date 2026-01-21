import datetime
import io
import numpy as np
import pandas as pd
import torch
from transformers import CLIPModel, CLIPProcessor
from evidently.legacy.metrics import DataDriftTable
from evidently.legacy.report import Report

from google.cloud import storage
from ai_vs_human.data import MyDataset

GCS_BUCKET_NAME = "mlops-project-22-monitoring"
GCS_OBJECT_NAME = "reference/features.csv"


def extract_features(img_chw: np.ndarray) -> dict:
    gray = img_chw.mean(axis=0)

    brightness = float(gray.mean())
    contrast = float(gray.std())

    gy, gx = np.gradient(gray, axis=(0, 1))
    sharpness = float((np.abs(gx) + np.abs(gy)).mean())

    channel_std = float(img_chw.std(axis=0).mean())

    return {
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
        "channel_std": channel_std,
    }


def extract_clip_features(images: torch.Tensor, model, processor, batch_size: int = 16) -> np.ndarray:
    feats = []

    if images.ndim == 3:
        images = images.unsqueeze(1)

    for i in range(0, images.size(0), batch_size):
        batch = images[i : i + batch_size]

        batch = batch.permute(0, 2, 3, 1).numpy()
        inputs = processor(images=list(batch), return_tensors="pt")

        with torch.no_grad():
            emb = model.get_image_features(inputs["pixel_values"])

        feats.append(emb.numpy())

    return np.concatenate(feats, axis=0)


def upload_training_features() -> None:
    train_ds = MyDataset(train=True)

    rows = []
    timestamp = datetime.datetime.now(tz=datetime.UTC).isoformat()

    for i in range(len(train_ds)):
        img = train_ds[i][0]
        img_np = img.float().numpy()

        features = extract_features(img_np)

        rows.append(
            {
                "index": i,
                "timestamp": timestamp,
                **features,
            }
        )

    df = pd.DataFrame(rows)

    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(GCS_OBJECT_NAME)

    blob.upload_from_string(
        buffer.getvalue(),
        content_type="text/csv",
    )


def data_drift_train_and_test():
    train_ds = MyDataset(train=True)
    test_ds = MyDataset(train=False)

    train_images = torch.stack([train_ds[i][0] for i in range(len(train_ds))])
    test_images = torch.stack([test_ds[i][0] for i in range(len(test_ds))])

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
    upload_training_features()
