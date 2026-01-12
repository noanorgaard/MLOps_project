import torch
from torch.utils.data import Dataset

from src.ai_vs_human.data import MyDataset


def test_my_dataset():
    """Test the MyDataset class."""
    dataset = MyDataset()
    assert isinstance(dataset, Dataset)
