from src.ai_vs_human.data import MyDataset
from torch.utils.data import Dataset


def test_data_exists_at_expected_location():
    dataset = MyDataset()  # <--- put something meaningful here
    assert dataset.processed_dir is not None


def test_data_is_instance_of_dataset():
    dataset = MyDataset()  # <--- put something meaningful here
    assert isinstance(dataset, Dataset)
