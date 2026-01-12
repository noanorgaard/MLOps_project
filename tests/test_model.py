import torch
from torch.utils.data import Dataset
from src.ai_vs_human.model import AIOrNotClassifier
from src.ai_vs_human.data import MyDataset

def test_my_model():
    """Test the MyModel class."""
    model = AIOrNotClassifier()
    assert isinstance(model, torch.nn.Module)