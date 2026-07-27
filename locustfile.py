import os
import io
from locust import HttpUser, task, between
from PIL import Image

class BrainTumorUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        """Prepare sample image payload in memory"""
        # Create a simple 224x224 RGB dummy image in memory if no file exists
        img = Image.new('RGB', (224, 224), color=(73, 109, 137))
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        self.sample_image = img_byte_arr.getvalue()

    @task(3)
    def test_health(self):
        self.client.get("/health")

    @task(7)
    def test_predict(self):
        files = {'file': ('test.jpg', self.sample_image, 'image/jpeg')}
        self.client.post("/predict", files=files)