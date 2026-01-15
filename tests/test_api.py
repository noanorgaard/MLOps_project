"""Tests for the FastAPI application."""

from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from ai_vs_human.api import app


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_image() -> bytes:
    """Generate a sample image for testing."""
    img = Image.new("RGB", (224, 224), color="red")
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)
    return img_byte_arr.getvalue()


@pytest.fixture
def sample_image_jpg() -> bytes:
    """Generate a sample JPG image for testing."""
    img = Image.new("RGB", (224, 224), color="blue")
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)
    return img_byte_arr.getvalue()


class TestHealthEndpoint:
    """Test the health check endpoint."""

    def test_health_check(self, client: TestClient) -> None:
        """Test that health endpoint returns 200 and correct status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "device" in data


class TestRootEndpoint:
    """Test the root endpoint."""

    def test_root_endpoint(self, client: TestClient) -> None:
        """Test that root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "AI vs Human Classifier"
        assert "endpoints" in data
        assert "predict" in data["endpoints"]


class TestPredictEndpoint:
    """Test the prediction endpoint."""

    @patch("ai_vs_human.api.model")
    def test_predict_with_valid_image(
        self, mock_model: MagicMock, client: TestClient, sample_image: bytes
    ) -> None:
        """Test prediction with a valid image."""
        # Mock the model to return a logit
        mock_model.return_value = torch.tensor([0.8])

        response = client.post(
            "/predict",
            files={"file": ("test.png", BytesIO(sample_image), "image/png")},
        )

        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "confidence" in data
        assert "logit" in data
        assert data["prediction"] in ["AI-generated", "Human-made"]
        assert 0 <= data["confidence"] <= 1

    @patch("ai_vs_human.api.model", None)
    def test_predict_model_not_loaded(
        self, client: TestClient, sample_image: bytes
    ) -> None:
        """Test prediction when model is not loaded."""
        response = client.post(
            "/predict",
            files={"file": ("test.png", BytesIO(sample_image), "image/png")},
        )

        assert response.status_code == 503
        data = response.json()
        assert "Model not loaded" in data["detail"]

    def test_predict_with_non_image_file(self, client: TestClient) -> None:
        """Test prediction with a non-image file."""
        response = client.post(
            "/predict",
            files={"file": ("test.txt", BytesIO(b"not an image"), "text/plain")},
        )

        assert response.status_code == 400
        data = response.json()
        assert "must be an image" in data["detail"]

    @patch("ai_vs_human.api.model")
    def test_predict_jpg_image(
        self, mock_model: MagicMock, client: TestClient, sample_image_jpg: bytes
    ) -> None:
        """Test prediction with JPG image."""
        mock_model.return_value = torch.tensor([-0.5])

        response = client.post(
            "/predict",
            files={"file": ("test.jpg", BytesIO(sample_image_jpg), "image/jpeg")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == "Human-made"

    @patch("ai_vs_human.api.model")
    def test_predict_high_confidence_ai(
        self, mock_model: MagicMock, client: TestClient, sample_image: bytes
    ) -> None:
        """Test prediction with high confidence for AI-generated."""
        mock_model.return_value = torch.tensor([2.0])

        response = client.post(
            "/predict",
            files={"file": ("test.png", BytesIO(sample_image), "image/png")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == "AI-generated"
        assert data["confidence"] > 0.85

    @patch("ai_vs_human.api.model")
    def test_predict_high_confidence_human(
        self, mock_model: MagicMock, client: TestClient, sample_image: bytes
    ) -> None:
        """Test prediction with high confidence for human-made."""
        mock_model.return_value = torch.tensor([-2.0])

        response = client.post(
            "/predict",
            files={"file": ("test.png", BytesIO(sample_image), "image/png")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == "Human-made"
        assert data["confidence"] < 0.15

    def test_predict_missing_file(self, client: TestClient) -> None:
        """Test prediction without providing a file."""
        response = client.post("/predict")

        assert response.status_code == 422  # Unprocessable Entity


class TestEndpointValidation:
    """Test general endpoint validation."""

    def test_api_documentation(self, client: TestClient) -> None:
        """Test that Swagger documentation is available."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_schema(self, client: TestClient) -> None:
        """Test that OpenAPI schema is available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "paths" in data
        assert "/predict" in data["paths"]
        assert "/health" in data["paths"]
