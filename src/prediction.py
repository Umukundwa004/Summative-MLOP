import os
import numpy as np
from PIL import Image
import tensorflow as tf

# Define standard default target dimensions for fallback
DEFAULT_TARGET_SIZE = (128, 128)

def load_model(model_path="models/model.h5"):
    """
    Loads and caches the trained Keras model from disk.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at path: {model_path}")
    
    model = tf.keras.models.load_model(model_path)
    return model


def get_model_target_size(model=None):
    """
    Extracts the required image input height and width (H, W) from the loaded model.
    Falls back to DEFAULT_TARGET_SIZE if model is missing or shape is indeterminate.
    """
    if model is None:
        return DEFAULT_TARGET_SIZE

    try:
        # Handles standard Keras models with input_shape attribute
        input_shape = model.input_shape
        
        # If input_shape is a list (e.g. multi-input models), take the first input
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        # Expecting shape like (None, height, width, channels)
        if input_shape and len(input_shape) >= 3:
            height = input_shape[1]
            width = input_shape[2]
            if height is not None and width is not None:
                return (int(height), int(width))
    except Exception:
        pass

    return DEFAULT_TARGET_SIZE


def preprocess_image(image: Image.Image, target_size: tuple) -> np.ndarray:
    """
    Preprocesses an uploaded PIL image for model inference.
    Resizes image, converts to array, scales pixel values to [0, 1], and adds batch dimension.
    """
    # Ensure RGB channel format
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Resize to target dimension (Width, Height) expected by PIL
    image = image.resize(target_size)
    
    # Convert PIL Image to numpy array and normalize pixels to [0, 1]
    img_array = np.array(image, dtype=np.float32) / 255.0
    
    # Expand dims to create batch dimension (1, height, width, channels)
    img_batch = np.expand_dim(img_array, axis=0)
    
    return img_batch


def predict(model, image: Image.Image, class_names: list = None):
    """
    Executes prediction on a single PIL image using the loaded Keras model.
    """
    target_size = get_model_target_size(model)
    processed_image = preprocess_image(image, target_size)
    
    # Perform forward pass prediction
    predictions = model.predict(processed_image)
    
    # Multi-class vs Binary classification processing
    if predictions.shape[-1] > 1:
        probs = tf.nn.softmax(predictions[0]).numpy() if not np.isclose(np.sum(predictions[0]), 1.0) else predictions[0]
        class_idx = int(np.argmax(probs))
        confidence = float(probs[class_idx])
    else:
        # Single output sigmoid output
        confidence = float(predictions[0][0])
        class_idx = 1 if confidence >= 0.5 else 0
        if class_idx == 0:
            confidence = 1.0 - confidence

    predicted_label = class_names[class_idx] if class_names and class_idx < len(class_names) else str(class_idx)

    return {
        "class_index": class_idx,
        "label": predicted_label,
        "confidence": confidence,
        "raw_predictions": predictions.tolist()
    }