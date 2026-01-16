from pathlib import Path
from typing import Optional, Callable, Tuple, List, Dict
import random
import typer
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import shutil


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
    img = img.resize((size, size), Image.LANCZOS)  # type: ignore[attr-defined]
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
    """Preprocess images stored in `raw/ai` and `raw/human`.

    Saves tensors to `processed`.
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    expected = ["ai", "human"]
    classes: List[str] = []
    for c in expected:
        d = raw_dir / c
        if not d.exists() or not d.is_dir():
            raise ValueError(
                f"Expected class folder '{c}' in {raw_dir}. Please create {raw_dir / 'ai'} and {raw_dir / 'human'} and populate with images."
            )
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

    images_tensor = torch.stack(images)
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


def download_data(raw_dir: Path) -> Path:
    """Download the Kaggle dataset and populate ``raw_dir``.

    Downloads "hassnainzaidi/ai-art-vs-human-art" using kagglehub, then attempts
    to copy image files into ``raw_dir/ai`` and ``raw_dir/human`` based on folder
    names found in the dataset path.

    Args:
        raw_dir: Destination raw data directory. Subfolders ``ai`` and ``human``
            will be created if missing.

    Returns:
        Path to the downloaded Kaggle dataset root.

    Raises:
        RuntimeError: If kagglehub is not available or no images could be
            classified into expected folders.
    """
    try:
        import kagglehub  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "kagglehub is required to download the dataset. Install it with 'uv add kagglehub'."
        ) from exc

    print("Downloading dataset from Kaggle (hassnainzaidi/ai-art-vs-human-art)...")
    dataset_path_str = kagglehub.dataset_download("hassnainzaidi/ai-art-vs-human-art")
    dataset_path = Path(dataset_path_str)
    print(f"Path to dataset files: {dataset_path}")

    ai_dir = raw_dir / "ai"
    human_dir = raw_dir / "human"
    ai_dir.mkdir(parents=True, exist_ok=True)
    human_dir.mkdir(parents=True, exist_ok=True)

    # Explicit mapping for this dataset
    ai_source = dataset_path / "Art" / "AiArtData"
    human_source = dataset_path / "Art" / "RealArt"

    def _copy_from(src: Path, dst: Path) -> int:
        n = 0
        if not src.exists():
            return 0
        for p in src.rglob("*"):
            if p.is_file() and _is_image_file(p):
                shutil.copy2(p, dst / p.name)
                n += 1
        return n

    copied_ai = 0
    copied_human = 0

    if ai_source.exists() or human_source.exists():
        copied_ai += _copy_from(ai_source, ai_dir)
        copied_human += _copy_from(human_source, human_dir)
    else:
        # Fallback heuristic if expected folders are missing
        def _infer_label_from_path(p: Path) -> Optional[str]:
            try:
                rel = p.relative_to(dataset_path)
            except Exception:
                return None
            dir_parts = [s.lower() for s in rel.parts[:-1]]
            if any("realart" in s or "human" in s for s in dir_parts):
                return "human"
            if any("aiartdata" in s or "ai" in s or "generated" in s for s in dir_parts):
                return "ai"
            return None

        for p in dataset_path.rglob("*"):
            if p.is_file() and _is_image_file(p):
                label = _infer_label_from_path(p)
                if label == "ai":
                    shutil.copy2(p, ai_dir / p.name)
                    copied_ai += 1
                elif label == "human":
                    shutil.copy2(p, human_dir / p.name)
                    copied_human += 1

    if copied_ai == 0 and copied_human == 0:
        raise RuntimeError(
            f"No images copied from {dataset_path}. Please inspect dataset structure and adjust labeling mapping."
        )

    print(f"Copied {copied_ai} AI images and {copied_human} Human images to {raw_dir}.")
    return dataset_path


"""
def prepare_data(
    raw_dir: Path = typer.Option(Path("data/raw"), help="Raw data directory containing 'ai' and 'human' subfolders"),
    processed_dir: Path = typer.Option(Path("data/processed"), help="Where to write processed tensors"),
    image_size: int = typer.Option(224, help="Size (H and W) to resize images to"),
    train_split: float = typer.Option(0.8, help="Fraction of data to use for training"),
    seed: int = typer.Option(42, help="Shuffle seed"),
) -> None:
"""


def prepare_data(
    raw_dir: Path = Path("data/raw"),
    processed_dir: Path = Path("data/processed"),
    image_size: int = 224,
    train_split: float = 0.8,
    seed: int = 42,
) -> None:
    """Preprocess raw image dataset into torch tensors."""

    # check if raw dir exists
    if not raw_dir.exists():
        print(f"Raw directory {raw_dir} not found — creating and downloading Kaggle dataset...")
        raw_dir.mkdir(parents=True, exist_ok=True)
        download_data(raw_dir)

    # check if subfolders exist and are non-empty
    ai_dir = raw_dir / "ai"
    human_dir = raw_dir / "human"
    ai_missing = not ai_dir.exists() or not any(ai_dir.iterdir())
    human_missing = not human_dir.exists() or not any(human_dir.iterdir())
    if ai_missing or human_missing:
        print("Raw folders missing or empty — attempting Kaggle download...")
        download_data(raw_dir)

    # preprocess and save data
    print(f"Preprocessing and saving images to: {processed_dir}")
    preprocess_and_save(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        image_size=image_size,
        train_split=train_split,
        seed=seed,
    )


if __name__ == "__main__":
    typer.run(prepare_data)
