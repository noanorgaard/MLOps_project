import datetime
import io
import numpy as np
import pandas as pd
import torch

from google.cloud import storage
from ai_vs_human.data import MyDataset

GCS_BUCKET_NAME = "ai-vs-human-monitoring"
GCS_OBJECT_NAME = "reference/train_features.csv"


def extract_features_1(img_chw: np.ndarray) -> dict:
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


def main() -> None:
    train_ds = MyDataset(train=True)

    rows = []
    timestamp = datetime.datetime.now(tz=datetime.UTC).isoformat()

    for i in range(len(train_ds)):
        img = train_ds[i][0]          # torch.Tensor [C, H, W]
        img_np = img.float().numpy()

        features = extract_features_1(img_np)

        rows.append(
            {
                "index": i,
                "timestamp": timestamp,
                **features,
            }
        )

    df = pd.DataFrame(rows)

    # Write CSV to memory (no local file)
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    # Upload to GCS
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(GCS_OBJECT_NAME)

    blob.upload_from_string(
        buffer.getvalue(),
        content_type="text/csv",
    )


if __name__ == "__main__":
    main()
