import os
import numpy as np
from PIL import Image
import tensorflow as tf

# Default fallback target size (Height, Width)
DEFAULT_TARGET_SIZE = (128, 128)


def load_model(model_path="models/model.h5"):
    """
    Loads and returns the trained Keras model from disk.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at path: {model_path}")
    
    return tf.keras.models.load_model(model_path)


def get_model_target_size(model=None):
    """
    Extracts the image input height and width (H, W) expected by the model.
    Defaults to (128, 128) if model is missing or shape is unspecified.
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
    Preprocesses a PIL Image for model prediction.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Resize to target dimension (Width, Height) for PIL
    image = image.resize((target_size[1], target_size[0]))
    
    # Scale pixels to [0, 1]
    img_array = np.array(image, dtype=np.float32) / 255.0
    
    # Expand dims for batch processing (1, height, width, channels)
    return np.expand_dims(img_array, axis=0)


def predict(model, image: Image.Image, class_names: list = None):
    """
    Executes prediction using the model on a single PIL image.
    """
    target_size = get_model_target_size(model)
    processed_image = preprocess_image(image, target_size)
    
    predictions = model.predict(processed_image)
    
    if predictions.shape[-1] > 1:
        probs = tf.nn.softmax(predictions[0]).numpy() if not np.isclose(np.sum(predictions[0]), 1.0) else predictions[0]
        class_idx = int(np.argmax(probs))
        confidence = float(probs[class_idx])
    else:
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


def predict_image(image: Image.Image, model=None, class_names: list = None):
    """
    Wrapper alias matching app.py import expected signature: predict_image(image, model, class_names)
    """
    if model is None:
        # Load default model if not explicitly passed
        model = load_model()
    return predict(model, image, class_names)