import cv2
import numpy as np

def preprocess_image(image: np.ndarray, target_size=(150, 150)) -> np.ndarray:
    """Preprocesses an OpenCV image for model prediction."""
    # 1. Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 2. Resize image to 150x150 (matching the trained Keras model shape)
    resized = cv2.resize(image_rgb, target_size)
    
    # 3. Normalize pixels (0 to 1)
    normalized = resized / 255.0
    
    # 4. Add batch dimension: (1, 150, 150, 3)
    batch_image = np.expand_dims(normalized, axis=0)
    
    return batch_image