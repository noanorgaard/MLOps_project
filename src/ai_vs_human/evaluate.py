"""Model evaluation functions for AI vs Human classifier."""

from typing import Tuple

import torch


def predict_from_logit(logit: torch.Tensor) -> Tuple[str, float]:
    """
    Convert model logit to prediction label and confidence.

    Args:
        logit: Raw model output (single value or batch)

    Returns:
        Tuple of (prediction label, confidence score)
        - prediction: "AI-generated" if confidence > 0.5, else "Human-made"
        - confidence: Probability score between 0 and 1
    """
    confidence = torch.sigmoid(logit).item()
    prediction = "AI-generated" if confidence > 0.5 else "Human-made"
    return prediction, confidence


def evaluate_batch(
    model: torch.nn.Module, images: torch.Tensor, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluate a batch of images.

    Args:
        model: The trained model
        images: Batch of preprocessed image tensors [N, C, H, W]
        device: Device to run inference on

    Returns:
        Tuple of (logits, confidences) as tensors
    """
    model.eval()
    images = images.to(device)

    with torch.no_grad():
        logits = model(images).squeeze(-1)
        confidences = torch.sigmoid(logits)

    return logits, confidences
