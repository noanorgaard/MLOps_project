from pathlib import Path
from typing import Optional, Callable, Tuple, List, Dict
import argparse
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np


class MyDataset(Dataset):
    """Simple dataset loading processed tensors saved to disk.

    Expects raw data in `data/raw/ai` and `data/raw/human` when preprocessing.
    """

    def __init__(
        self,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
        image_size: int = 224,
        train: bool = True,
        train_split: float = 0.8,
        transform: Optional[Callable] = None,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.image_size = image_size
        self.train = train
        self.transform = transform
        self.train_split = train_split

        self.processed_dir.mkdir(parents=True, exist_ok=True)

        train_images_path = self.processed_dir / "train_images.pt"
        test_images_path = self.processed_dir / "test_images.pt"
        if not train_images_path.exists() or not test_images_path.exists():
            if not self.raw_dir.exists():
                raise FileNotFoundError(f"Raw data directory not found: {self.raw_dir}")
            print("Processed files not found — running preprocessing from raw data...")
            preprocess_and_save(
                raw_dir=self.raw_dir,
                processed_dir=self.processed_dir,
                image_size=self.image_size,
                train_split=self.train_split,
            )

        if self.train:
            images_file = self.processed_dir / "train_images.pt"
            targets_file = self.processed_dir / "train_target.pt"
        else:
            images_file = self.processed_dir / "test_images.pt"
            targets_file = self.processed_dir / "test_target.pt"

        self.images = torch.load(images_file)
        self.targets = torch.load(targets_file)

        if self.images.dtype != torch.float32:
            self.images = self.images.float()

        if len(self.images.shape) == 3:
            # add channel dim if missing
            self.images = self.images.unsqueeze(1)

    def __len__(self) -> int:
        return self.images.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = self.images[idx]
        label = int(self.targets[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# Helper functions
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def _is_image_file(p: Path) -> bool:
    return p.suffix.lower() in _IMAGE_EXTENSIONS


def _load_and_process_image(path: Path, size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr).float()


def preprocess_and_save(
    raw_dir: str | Path,
    processed_dir: str | Path,
    image_size: int = 224,
    train_split: float = 0.8,
    seed: int = 42,
) -> Dict[str, Path]:
    """Preprocess images stored in `raw_dir/ai` and `raw_dir/human`.

    Saves tensors to `processed_dir`.
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    expected = ["ai", "human"]
    classes: List[str] = []
    for c in expected:
        d = raw_dir / c
        if not d.exists() or not d.is_dir():
            raise ValueError(f"Expected class folder '{c}' in {raw_dir}. Please create {raw_dir / 'ai'} and {raw_dir / 'human'} and populate with images.")
        classes.append(c)

    class_to_idx = {c: i for i, c in enumerate(classes)}

    images: List[torch.Tensor] = []
    labels: List[int] = []

    for cls in classes:
        cls_dir = raw_dir / cls
        for p in sorted(cls_dir.iterdir()):
            if p.is_file() and _is_image_file(p):
                try:
                    t = _load_and_process_image(p, image_size)
                    images.append(t)
                    labels.append(class_to_idx[cls])
                except Exception as exc:
                    print(f"Warning: failed to process {p}: {exc}")

    if not images:
        raise ValueError(f"No images found under {raw_dir}")

    images_tensor = torch.stack(images)  # (N, 3, H, W)
    images_tensor = images_tensor.float()
    images_tensor = torch.clamp(images_tensor, 0.0, 1.0)
    targets_tensor = torch.tensor(labels, dtype=torch.long)

    rng = random.Random(seed)
    idxs = list(range(images_tensor.size(0)))
    rng.shuffle(idxs)
    images_tensor = images_tensor[idxs]
    targets_tensor = targets_tensor[idxs]

    n_train = int(len(idxs) * train_split)
    train_images = images_tensor[:n_train]
    train_targets = targets_tensor[:n_train]
    test_images = images_tensor[n_train:]
    test_targets = targets_tensor[n_train:]

    train_images_path = processed_dir / "train_images.pt"
    train_targets_path = processed_dir / "train_target.pt"
    test_images_path = processed_dir / "test_images.pt"
    test_targets_path = processed_dir / "test_target.pt"

    torch.save(train_images, train_images_path)
    torch.save(train_targets, train_targets_path)
    torch.save(test_images, test_images_path)
    torch.save(test_targets, test_targets_path)

    print(f"Saved: {train_images_path} ({train_images.shape})")
    print(f"Saved: {train_targets_path} ({train_targets.shape})")
    print(f"Saved: {test_images_path} ({test_images.shape})")
    print(f"Saved: {test_targets_path} ({test_targets.shape})")

    meta = {"class_to_idx": class_to_idx}
    torch.save(meta, processed_dir / "meta.pt")

    return {
        "train_images": train_images_path,
        "train_targets": train_targets_path,
        "test_images": test_images_path,
        "test_targets": test_targets_path,
        "meta": processed_dir / "meta.pt",
    }


def prepare_from_cli() -> None:
    parser = argparse.ArgumentParser(description="Preprocess raw image dataset into torch tensors")
    parser.add_argument("--raw-dir", default="data/raw", help="Raw data directory containing 'ai' and 'human' subfolders")
    parser.add_argument("--processed-dir", default="data/processed", help="Where to write processed tensors")
    parser.add_argument("--image-size", type=int, default=224, help="Size (H and W) to resize images to")
    parser.add_argument("--train-split", type=float, default=0.8, help="Fraction of data to use for training")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        raise RuntimeError(f"Raw directory {raw_dir} not found. Please create it with 'ai' and 'human' subfolders and add images.")

    preprocess_and_save(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        image_size=args.image_size,
        train_split=args.train_split,
        seed=args.seed,
    )


if __name__ == "__main__":
    prepare_from_cli()
