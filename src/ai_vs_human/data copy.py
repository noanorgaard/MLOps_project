from pathlib import Path

import typer
from torch.utils.data import Dataset
import kagglehub
import zipfile

# download data (if not downloaded)
# preprocess data
    # The data format should be aligned
    # size should be consistent (RGB, same height/width) (should align with torch)
    # lenx3x256x256
    # split into train, test
    # save in preprocessed folder as train_images.pt and test_images.pt files
    # label should be also be added test_target.pt and train_taget.pt

class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path

    @staticmethod
    def download_data(data_dir: Path = Path("data/raw")) -> Path:
        """Download the dataset from Kaggle into data_dir if the folder is empty.

        Args:
            data_dir: Path to the raw data directory.

        Returns:
            Path to the raw data directory containing the dataset.
        """
        data_dir = data_dir.expanduser().resolve()
        data_dir.mkdir(parents=True, exist_ok=True)
        if any(data_dir.iterdir()):
            return data_dir
        path = kagglehub.dataset_download("hassnainzaidi/ai-art-vs-human-art", str(data_dir))
        downloaded = Path(path) if isinstance(path, (str, Path)) else None
        if downloaded and downloaded.exists() and downloaded.is_file():
            try:
                with zipfile.ZipFile(downloaded, "r") as z:
                    z.extractall(data_dir)
                downloaded.unlink()
            except Exception:
                pass
        return data_dir

    def __len__(self) -> int:
        """Return the length of the dataset."""

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""

def preprocess(data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(data_path)
    dataset.preprocess(output_folder)

def main(data_path: Path = Path("data/raw"), output_folder: Path = Path("data/processed")) -> None:
    """Download raw data (if needed) and run preprocessing.

    Args:
        data_path: Location for raw data (download target).
        output_folder: Location to write processed data.
    """
    MyDataset.download_data(data_path)
    preprocess(data_path, output_folder)


if __name__ == "__main__":
    typer.run(main)
