import os
import numpy as np
import tensorflow as tf
import keras

# Patch GlorotUniform initializer to prevent deserialization errors due to version mismatches.
classes_to_patch = set()
for lib in (keras, tf.keras):
    try:
        classes_to_patch.add(lib.initializers.GlorotUniform)
    except AttributeError:
        pass

for cls in classes_to_patch:
    orig_init = cls.__init__
    if not hasattr(orig_init, "_is_patched"):
        def make_patched_init(old_init):
            def patched_init(self, *args, **kwargs):
                kwargs.pop("input_axes", None)
                kwargs.pop("output_axes", None)
                return old_init(self, *args, **kwargs)
            patched_init._is_patched = True
            return patched_init
        cls.__init__ = make_patched_init(orig_init)

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
