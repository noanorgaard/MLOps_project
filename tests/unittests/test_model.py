import torch
from ai_vs_human.model import AIOrNotClassifier
import pytest


@pytest.mark.parametrize("batch_size, dropout_rate", [(1, 0.3), (4, 0.5)])
def test_model_forward_shape(batch_size, dropout_rate):
    model = AIOrNotClassifier(dropout_rate=dropout_rate)
    model.eval()

    x = torch.randn(batch_size, 3, 224, 224)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (batch_size, 1)
