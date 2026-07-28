from locust import HttpUser, task, between
import io
from PIL import Image

class BrainTumorUser(HttpUser):
    wait_time = between(1, 3)

    @task(2)
    def test_health(self):
        self.client.get("/")

    @task(1)
    def test_prediction(self):
        # Create a dummy RGB image for performance testing
        img = Image.new('RGB', (150, 150), color='white')
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_bytes = img_byte_arr.getvalue()

        files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}
        self.client.post("/predict", files=files)