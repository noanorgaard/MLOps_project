from ai_vs_human.data import MyDataset
import torch
import pytest
import os

REQUIRED_FILES = [
    "data/processed/train_images.pt",
    "data/processed/train_target.pt",
    "data/processed/test_images.pt",
    "data/processed/test_target.pt",
]

@pytest.mark.skipif(
    not all(os.path.exists(f) for f in REQUIRED_FILES),
    reason="Processed dataset files not found",
)
@pytest.mark.parametrize(
    "train_flag, expected_len",
    [
        (True, 776),
        (False, 194),
    ],
)
def test_my_dataset(train_flag, expected_len):
    dataset = MyDataset(processed_dir="data/processed", train=train_flag)

    assert len(dataset) == expected_len

    for x, y in dataset:
        assert x.dtype == torch.float32
        assert x.shape[0] in (1, 3)     
        assert x.shape[1] == x.shape[2] 

        assert y in (0, 1)

    targets = torch.unique(dataset.targets)
    assert torch.equal(targets, torch.tensor([0, 1]))

