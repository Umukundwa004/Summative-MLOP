import os
import numpy as np
from PIL import Image
import tensorflow as tf

# Define standard brain tumor class labels matching dataset alphabetical/folder order
CLASS_NAMES = ["Glioma Tumor", "Meningioma Tumor", "No Tumor", "Pituitary Tumor"]

# Path to the trained model file
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "brain_tumor_model.keras")

# Global model cache to avoid re-reading from disk on every prediction
_MODEL = None


def load_model():
    """
    Loads and caches the trained Keras model.
    Checks for .keras first, then falls back to .h5 if necessary.
    """
    global _MODEL
    if _MODEL is None:
        if os.path.exists(MODEL_PATH):
            _MODEL = tf.keras.models.load_model(MODEL_PATH)
        else:
            alt_path = os.path.join(os.path.dirname(__file__), "..", "models", "brain_tumor_model.h5")
            if os.path.exists(alt_path):
                _MODEL = tf.keras.models.load_model(alt_path)
            else:
                raise FileNotFoundError(f"Model file not found at {MODEL_PATH} or {alt_path}")
    return _MODEL


def get_model_target_size():
    """
    Dynamically inspects the loaded model's input shape to avoid dimension mismatch errors.
    Defaults to (128, 128) if unassigned.
    """
    model = load_model()
    try:
        # Extract height and width from model input shape tuple: (None, height, width, channels)
        input_shape = model.input_shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        
        height = input_shape[1] if input_shape[1] is not None else 128
        width = input_shape[2] if input_shape[2] is not None else 128
        return (height, width)
    except Exception:
        return (128, 128)


def preprocess_image(image: Image.Image):
    """
    Preprocesses the raw uploaded image into a normalized 4D tensor ready for TensorFlow prediction.
    """
    target_size = get_model_target_size()
    
    # Ensure RGB format (converts RGBA or Grayscale scans to 3 channels)
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    # Resize to the model's required input dimensions
    image = image.resize(target_size)
    
    # Convert PIL Image to numpy array and normalize pixels to [0, 1] range
    img_array = np.array(image, dtype=np.float32) / 255.0
    
    # Add batch dimension: (1, height, width, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def predict_image(image: Image.Image):
    """
    Executes model inference on an uploaded image.
    
    Returns:
        tuple: (predicted_class_name, confidence_percentage, probabilities_dictionary)
    """
    model = load_model()
    processed_tensor = preprocess_image(image)
    
    # Run prediction
    raw_predictions = model.predict(processed_tensor, verbose=0)[0]
    
    # Identify top prediction index
    top_index = int(np.argmax(raw_predictions))
    predicted_class = CLASS_NAMES[top_index]
    confidence = float(raw_predictions[top_index])
    
    # Construct probability map across all target classes
    probabilities = {CLASS_NAMES[i]: float(raw_predictions[i]) for i in range(len(CLASS_NAMES))}
    
    return predicted_class, confidence, probabilities

def get_model_target_size(model):
    """
    Extracts the target input image dimensions (height, width) expected by the model.
    """
    input_shape = model.input_shape
    # input_shape is typically (None, height, width, channels)
    if len(input_shape) >= 3 and input_shape[1] is not None and input_shape[2] is not None:
        return (input_shape[1], input_shape[2])
    
    # Fallback default target size if input_shape is dynamic
    return (128, 128)