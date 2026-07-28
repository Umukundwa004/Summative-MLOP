import os
import numpy as np
from PIL import Image
import keras
import tensorflow as tf

# Default fallback input dimension (Height, Width)
DEFAULT_TARGET_SIZE = (150, 150)

# Brain Tumor Diagnostic Classes
CLASS_NAMES = ["Glioma Tumor", "Meningioma Tumor", "No Tumor", "Pituitary Tumor"]


def load_model(model_path=None):
    """
    Loads and returns the trained Keras 3 model from disk.
    """
    candidate_paths = [
        model_path,
        "models/brain_tumor_model.keras",
        "models/model.keras",
        "models/model.h5",
        "models/brain_tumor_model.h5",
        "model.keras",
        "model.h5"
    ]
    
    valid_candidates = [p for p in candidate_paths if p is not None]
    
    for path in valid_candidates:
        if os.path.exists(path):
            try:
                # Use keras directly for Keras 3 formats
                return keras.models.load_model(path, compile=False)
            except Exception:
                # Fallback to tf.keras
                return tf.keras.models.load_model(path, compile=False)
            
    raise FileNotFoundError(
        f"Model file not found or could not be loaded. Checked paths: {valid_candidates}."
    )


def get_model_target_size(model=None):
    """
    Dynamically extracts input spatial dimensions (Height, Width) from the model.
    """
    if model is None:
        return DEFAULT_TARGET_SIZE

    try:
        input_shape = model.input_shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        if input_shape and len(input_shape) >= 3:
            height, width = input_shape[1], input_shape[2]
            if height is not None and width is not None:
                return (int(height), int(width))
    except Exception:
        pass

    return DEFAULT_TARGET_SIZE


def preprocess_image(image: Image.Image, target_size: tuple) -> np.ndarray:
    """
    Preprocesses a PIL Image for model evaluation.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # PIL resize expects (width, height)
    image = image.resize((target_size[1], target_size[0]))
    
    # Convert image to float32 numpy array and normalize
    img_array = np.array(image, dtype=np.float32) / 255.0
    
    # Add batch dimension: (1, height, width, channels)
    return np.expand_dims(img_array, axis=0)


def predict_image(image: Image.Image, model=None, class_names=None):
    """
    Runs model inference on a single PIL image.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    if model is None:
        model = load_model()

    target_size = get_model_target_size(model)
    processed_image = preprocess_image(image, target_size)

    # Perform forward pass
    raw_preds = model.predict(processed_image)

    # Multi-class output processing
    if raw_preds.shape[-1] > 1:
        probs = tf.nn.softmax(raw_preds[0]).numpy() if not np.isclose(np.sum(raw_preds[0]), 1.0) else raw_preds[0]
        class_idx = int(np.argmax(probs))
        confidence = float(probs[class_idx])
        
        prob_dict = {
            class_names[i] if i < len(class_names) else f"Class {i}": float(probs[i])
            for i in range(len(probs))
        }
    else:
        confidence = float(raw_preds[0][0])
        class_idx = 1 if confidence >= 0.5 else 0
        if class_idx == 0:
            confidence = 1.0 - confidence
        
        prob_dict = {
            class_names[0]: 1.0 - confidence,
            class_names[1]: confidence
        }

    predicted_label = class_names[class_idx] if class_idx < len(class_names) else str(class_idx)

    return {
        "label": predicted_label,
        "confidence": confidence,
        "probabilities": prob_dict,
        "class_index": class_idx
    }