from locust import HttpUser, task, between
import os

# Path to a sample MRI test image
TEST_IMAGE_PATH = os.path.join("data", "test", "sample.jpg")  # Adjust path to any valid image file in your repo

class BrainTumorAPIUser(HttpUser):
    # Wait between 1 and 3 seconds between tasks per simulated user
    wait_time = between(1, 3)

    @task(1)
    def test_health(self):
        """Tests the lightweight GET /health endpoint."""
        self.client.get("/health")

    @task(3)
    def test_predict(self):
        """Tests the POST /predict endpoint with an image upload."""
        if not os.path.exists(TEST_IMAGE_PATH):
            print(f"Warning: Test image not found at {TEST_IMAGE_PATH}")
            return

        with open(TEST_IMAGE_PATH, "rb") as image_file:
            files = {
                "file": ("sample.jpg", image_file, "image/jpeg")
            }
            # Sends multipart/form-data request
            self.client.post("/predict", files=files)