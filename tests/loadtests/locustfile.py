"""Locust load testing configuration for AI vs Human classifier API.

This file defines load testing scenarios for the FastAPI application.
Run with: uv run locust -f tests/loadtests/locustfile.py --host=http://localhost:8000
"""

import io
import random

from locust import HttpUser, between, task
from PIL import Image


class AIvsHumanUser(HttpUser):
    """Simulates a user interacting with the AI vs Human classifier API.

    Attributes:
        wait_time: Time between consecutive requests (1-3 seconds)
        test_images: List of test image paths to use for predictions
    """

    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks

    def on_start(self):
        """Initialize test data when user starts."""
        # Create synthetic test images for load testing
        self.test_images = []
        self._create_test_images()

    def _create_test_images(self):
        """Create in-memory test images for load testing."""
        # Generate 5 different test images to simulate variety
        for i in range(5):
            # Create a random RGB image
            img = Image.new("RGB", (224, 224), color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
            
            # Convert to bytes
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="JPEG")
            img_bytes.seek(0)
            
            self.test_images.append(img_bytes.getvalue())

    @task(10)
    def predict_image(self):
        """Test the /predict endpoint with an image file.

        This task is weighted at 10, making it the primary load test.
        """
        # Select a random test image
        image_data = random.choice(self.test_images)
        
        # Create file-like object
        files = {"file": ("test_image.jpg", io.BytesIO(image_data), "image/jpeg")}
        
        with self.client.post(
            "/predict",
            files=files,
            catch_response=True,
            name="/predict"
        ) as response:
            if response.status_code == 200:
                result = response.json()
                if "prediction" in result and "confidence" in result:
                    response.success()
                else:
                    response.failure("Invalid response format")
            elif response.status_code == 503:
                response.failure("Model not loaded")
            else:
                response.failure(f"Unexpected status code: {response.status_code}")

    @task(3)
    def health_check(self):
        """Test the /health endpoint.

        This task is weighted at 3, making it run less frequently than predict.
        """
        with self.client.get("/health", catch_response=True, name="/health") as response:
            if response.status_code == 200:
                result = response.json()
                if "status" in result and "model_loaded" in result:
                    if result["model_loaded"]:
                        response.success()
                    else:
                        response.failure("Model not loaded")
                else:
                    response.failure("Invalid health response format")
            else:
                response.failure(f"Health check failed: {response.status_code}")

    @task(1)
    def get_root(self):
        """Test the root endpoint.

        This task is weighted at 1, making it run occasionally.
        """
        with self.client.get("/", catch_response=True, name="/") as response:
            if response.status_code == 200:
                result = response.json()
                if "name" in result and "endpoints" in result:
                    response.success()
                else:
                    response.failure("Invalid root response format")
            else:
                response.failure(f"Root endpoint failed: {response.status_code}")


class StressTestUser(HttpUser):
    """Aggressive load test user for stress testing.

    This user makes rapid consecutive requests to test system limits.
    """

    wait_time = between(0.1, 0.5)  # Very short wait time for stress testing

    def on_start(self):
        """Initialize test data when user starts."""
        # Create one test image for quick stress testing
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        self.test_image = img_bytes.getvalue()

    @task
    def rapid_predict(self):
        """Rapidly send prediction requests for stress testing."""
        files = {"file": ("test.jpg", io.BytesIO(self.test_image), "image/jpeg")}
        self.client.post("/predict", files=files, name="/predict [stress]")
