import os
import numpy as np
from PIL import Image
import tensorflow as tf

# Default fallback target size (Height, Width)
DEFAULT_TARGET_SIZE = (128, 128)

# Class labels for Brain Tumor Classification
CLASS_NAMES = ["Glioma Tumor", "Meningioma Tumor", "No Tumor", "Pituitary Tumor"]


def load_model(model_path=None):
    """
    Loads and returns the trained Keras model from disk.
    Automatically checks standard `.keras` and `.h5` locations.
    """
    candidate_paths = [
        model_path,
        "models/model.keras",
        "models/brain_tumor_model.keras",
        "model/model.keras",
        "model.keras",
        "models/model.h5"
    ]
    
    # Filter out None values
    valid_candidates = [p for p in candidate_paths if p is not None]
    
    for path in valid_candidates:
        if os.path.exists(path):
            return tf.keras.models.load_model(path)
            
    raise FileNotFoundError(
        f"Model file not found. Checked locations: {valid_candidates}. "
        "Please ensure your .keras file is committed to GitHub inside the models/ directory."
    )


def get_model_target_size(model=None):
    """
    Extracts the image input height and width (H, W) expected by the Keras model.
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
    Preprocesses a PIL Image for Keras model prediction.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Resize image (Width, Height) for PIL
    image = image.resize((target_size[1], target_size[0]))
    
    # Scale pixels to [0, 1]
    img_array = np.array(image, dtype=np.float32) / 255.0
    
    # Add batch dimension (1, height, width, channels)
    return np.expand_dims(img_array, axis=0)


def predict_image(image: Image.Image, model=None, class_names=None):
    """
    Performs inference on a single PIL image and returns predictions, confidence,
    and a formatted class probability dictionary for Streamlit UI rendering.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    if model is None:
        model = load_model()

    target_size = get_model_target_size(model)
    processed_image = preprocess_image(image, target_size)

    # Perform forward pass
    raw_preds = model.predict(processed_image)

    # Multi-class output (Softmax / Multi-logit)
    if raw_preds.shape[-1] > 1:
        probs = tf.nn.softmax(raw_preds[0]).numpy() if not np.isclose(np.sum(raw_preds[0]), 1.0) else raw_preds[0]
        class_idx = int(np.argmax(probs))
        confidence = float(probs[class_idx])
        
        # Format class probabilities dictionary
        prob_dict = {
            class_names[i] if i < len(class_names) else f"Class {i}": float(probs[i])
            for i in range(len(probs))
        }
    else:
        # Binary classification fallback
        confidence = float(raw_preds[0][0])
        class_idx = 1 if confidence >= 0.5 else 0
        if class_idx == 0:
            confidence = 1.0 - confidence
        
        prob_dict = {
            class_names[0]: 1.0 - confidence,
            class_names[1]: confidence
        }

    predicted_label = class_names[class_idx] if class_idx < len(class_names) else str(class_idx)

    # Return structured dictionary compatible with app.py
    return {
        "label": predicted_label,
        "confidence": confidence,
        "probabilities": prob_dict,
        "class_index": class_idx
    }