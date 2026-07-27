import os
import numpy as np
import tensorflow as tf

CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Automatically locate models/ directory relative to this file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "brain_tumor_model.keras")


class ModelPredictor:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"Successfully loaded model from {self.model_path}")
        else:
            self.model = None
            print(f"Warning: Model file not found at {self.model_path}")

    def predict(self, processed_image: np.ndarray):
        if self.model is None:
            self.load_model()
            if self.model is None:
                raise RuntimeError(f"Model file not found at {self.model_path}. Please check your models directory.")
        
        preds = self.model.predict(processed_image)
        class_idx = int(np.argmax(preds[0]))
        confidence = float(preds[0][class_idx]) * 100  # Percentage
        return CLASSES[class_idx], round(confidence, 2)