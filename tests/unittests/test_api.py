"""Unit tests for FastAPI application endpoints.

Tests cover:
- Successful prediction with valid image
- Health check endpoint
- Root endpoint
- Error handling (invalid files, model not loaded, etc.)
- Edge cases (file types, image formats, etc.)

Uses pytest with FastAPI TestClient for API testing.
"""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from ai_vs_human.api import app


@pytest.fixture
def client():
    """Create a TestClient for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_model():
    """Create a mock model that returns predictable outputs."""
    model = MagicMock()
    model.eval = MagicMock()
    # Mock forward pass to return a tensor
    mock_output = torch.tensor([0.8])  # Confidence > 0.5 => AI-generated
    model.return_value = mock_output
    return model


@pytest.fixture
def sample_image_bytes():
    """Create a sample RGB image in memory as bytes."""
    # Create a simple 224x224 RGB image
    img = Image.new("RGB", (224, 224), color="red")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    return img_bytes


@pytest.fixture
def setup_model_loaded(mock_model):
    """Fixture to simulate model being loaded successfully."""
    with patch("ai_vs_human.api.model", mock_model), patch("ai_vs_human.api.device", torch.device("cpu")):
        yield


@pytest.fixture
def setup_model_not_loaded():
    """Fixture to simulate model not being loaded."""
    with patch("ai_vs_human.api.model", None), patch("ai_vs_human.api.device", None):
        yield


class TestRootEndpoint:
    """Tests for the root endpoint (/)."""

    def test_root_endpoint_returns_200(self, client):
        """Test that root endpoint returns 200 status code."""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_endpoint_returns_json(self, client):
        """Test that root endpoint returns valid JSON."""
        response = client.get("/")
        data = response.json()
        assert isinstance(data, dict)

    def test_root_endpoint_contains_required_fields(self, client):
        """Test that root endpoint contains expected fields."""
        response = client.get("/")
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "status" in data
        assert "endpoints" in data

    def test_root_endpoint_has_correct_api_info(self, client):
        """Test that root endpoint has correct API information."""
        response = client.get("/")
        data = response.json()
        assert data["name"] == "AI vs Human Classifier"
        assert data["version"] == "1.0.0"
        assert data["status"] == "running"


class TestHealthEndpoint:
    """Tests for the health check endpoint (/health)."""

    def test_health_endpoint_returns_200(self, client):
        """Test that health endpoint returns 200 status code."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_endpoint_returns_json(self, client):
        """Test that health endpoint returns valid JSON."""
        response = client.get("/health")
        data = response.json()
        assert isinstance(data, dict)

    def test_health_endpoint_contains_required_fields(self, client):
        """Test that health endpoint contains expected fields."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "device" in data

    def test_health_endpoint_model_loaded(self, client, setup_model_loaded):
        """Test health endpoint when model is loaded."""
        response = client.get("/health")
        data = response.json()
        assert data["model_loaded"] is True
        assert data["status"] == "healthy"
        assert data["device"] == "cpu"

    def test_health_endpoint_model_not_loaded(self, client, setup_model_not_loaded):
        """Test health endpoint when model is not loaded."""
        response = client.get("/health")
        data = response.json()
        assert data["model_loaded"] is False
        assert data["status"] == "model_not_loaded"


class TestPredictEndpoint:
    """Tests for the prediction endpoint (/predict)."""

    def test_predict_with_valid_image_returns_200(self, client, setup_model_loaded, sample_image_bytes):
        """Test successful prediction with valid image."""
        response = client.post("/predict", files={"file": ("test.png", sample_image_bytes, "image/png")})
        assert response.status_code == 200

    def test_predict_returns_json_with_required_fields(self, client, setup_model_loaded, sample_image_bytes):
        """Test that prediction response contains required fields."""
        response = client.post("/predict", files={"file": ("test.png", sample_image_bytes, "image/png")})
        data = response.json()
        assert "prediction" in data
        assert "confidence" in data
        assert "logit" in data

    def test_predict_returns_valid_prediction(self, client, setup_model_loaded, sample_image_bytes):
        """Test that prediction is either AI-generated or Human-made."""
        response = client.post("/predict", files={"file": ("test.png", sample_image_bytes, "image/png")})
        data = response.json()
        assert data["prediction"] in ["AI-generated", "Human-made"]

    def test_predict_confidence_is_probability(self, client, setup_model_loaded, sample_image_bytes):
        """Test that confidence is between 0 and 1."""
        response = client.post("/predict", files={"file": ("test.png", sample_image_bytes, "image/png")})
        data = response.json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_ai_generated_when_confidence_high(self, client, sample_image_bytes):
        """Test that prediction is AI-generated when confidence > 0.5."""
        # Mock model to return high confidence (> 0.5)
        mock_model = MagicMock()
        mock_output = torch.tensor([1.5])  # sigmoid(1.5) ≈ 0.82 > 0.5
        mock_model.return_value = mock_output

        with patch("ai_vs_human.api.model", mock_model), patch("ai_vs_human.api.device", torch.device("cpu")):
            response = client.post("/predict", files={"file": ("test.png", sample_image_bytes, "image/png")})
            data = response.json()
            assert data["prediction"] == "AI-generated"
            assert data["confidence"] > 0.5

    def test_predict_human_made_when_confidence_low(self, client, sample_image_bytes):
        """Test that prediction is Human-made when confidence < 0.5."""
        # Mock model to return low confidence (< 0.5)
        mock_model = MagicMock()
        mock_output = torch.tensor([-1.5])  # sigmoid(-1.5) ≈ 0.18 < 0.5
        mock_model.return_value = mock_output

        with patch("ai_vs_human.api.model", mock_model), patch("ai_vs_human.api.device", torch.device("cpu")):
            response = client.post("/predict", files={"file": ("test.png", sample_image_bytes, "image/png")})
            data = response.json()
            assert data["prediction"] == "Human-made"
            assert data["confidence"] < 0.5

    def test_predict_with_jpg_image(self, client, setup_model_loaded):
        """Test prediction with JPEG image."""
        img = Image.new("RGB", (224, 224), color="blue")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)

        response = client.post("/predict", files={"file": ("test.jpg", img_bytes, "image/jpeg")})
        assert response.status_code == 200

    def test_predict_with_different_image_sizes(self, client, setup_model_loaded):
        """Test that images are properly resized."""
        # Test with non-standard size
        img = Image.new("RGB", (512, 512), color="green")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        response = client.post("/predict", files={"file": ("test.png", img_bytes, "image/png")})
        assert response.status_code == 200

    def test_predict_without_model_returns_503(self, client, setup_model_not_loaded, sample_image_bytes):
        """Test that prediction fails with 503 when model is not loaded."""
        response = client.post("/predict", files={"file": ("test.png", sample_image_bytes, "image/png")})
        assert response.status_code == 503
        assert "Model not loaded" in response.json()["detail"]

    def test_predict_with_invalid_file_type_returns_400(self, client, setup_model_loaded):
        """Test that non-image files are rejected."""
        text_file = io.BytesIO(b"This is not an image")
        response = client.post("/predict", files={"file": ("test.txt", text_file, "text/plain")})
        assert response.status_code == 400
        assert "Invalid file type" in response.json()["detail"]

    def test_predict_with_no_file_returns_422(self, client, setup_model_loaded):
        """Test that request without file is rejected."""
        response = client.post("/predict")
        assert response.status_code == 422  # Unprocessable Entity

    def test_predict_with_corrupted_image_returns_500(self, client, setup_model_loaded):
        """Test handling of corrupted image data."""
        corrupted_data = io.BytesIO(b"corrupted image data")
        response = client.post("/predict", files={"file": ("test.png", corrupted_data, "image/png")})
        assert response.status_code == 500
        assert "Prediction failed" in response.json()["detail"]


class TestModelIntegration:
    """Integration tests with model preprocessing."""

    def test_image_preprocessing_pipeline(self, client, setup_model_loaded):
        """Test that image preprocessing works correctly."""
        # Create a known image
        img = Image.new("RGB", (100, 100), color=(255, 0, 0))  # Red image
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        with patch("ai_vs_human.api.model") as mock_model:
            mock_output = torch.tensor([0.0])
            mock_model.return_value = mock_output

            response = client.post("/predict", files={"file": ("test.png", img_bytes, "image/png")})

            # Verify model was called with correct tensor shape
            assert mock_model.called
            call_args = mock_model.call_args[0][0]
            assert call_args.shape == (1, 3, 224, 224)  # Batch, Channels, Height, Width
            assert call_args.dtype == torch.float32

    def test_grayscale_image_converted_to_rgb(self, client, setup_model_loaded):
        """Test that grayscale images are converted to RGB."""
        # Create grayscale image
        img = Image.new("L", (224, 224), color=128)  # Grayscale
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        response = client.post("/predict", files={"file": ("test.png", img_bytes, "image/png")})
        assert response.status_code == 200


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_model_exception_during_inference_returns_500(self, client, sample_image_bytes):
        """Test handling of model errors during inference."""
        mock_model = MagicMock()
        mock_model.side_effect = RuntimeError("Model inference error")

        with patch("ai_vs_human.api.model", mock_model), patch("ai_vs_human.api.device", torch.device("cpu")):
            response = client.post("/predict", files={"file": ("test.png", sample_image_bytes, "image/png")})
            assert response.status_code == 500
            assert "Prediction failed" in response.json()["detail"]

    def test_empty_file_returns_500(self, client, setup_model_loaded):
        """Test handling of empty file."""
        empty_file = io.BytesIO(b"")
        response = client.post("/predict", files={"file": ("test.png", empty_file, "image/png")})
        assert response.status_code == 500


class TestConcurrency:
    """Tests for concurrent requests."""

    def test_multiple_predictions_sequential(self, client, setup_model_loaded):
        """Test multiple sequential predictions."""
        for i in range(5):
            img = Image.new("RGB", (224, 224), color="red")
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="PNG")
            img_bytes.seek(0)

            response = client.post("/predict", files={"file": (f"test{i}.png", img_bytes, "image/png")})
            assert response.status_code == 200
