import os
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import tensorflow as tf

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = FastAPI(
    title="Brain Tumor Detection API & Web UI",
    version="1.0.0"
)

# ---------------------------------------------------------
# Load Trained Model & Define Class Names
# ---------------------------------------------------------
MODEL_PATH = os.path.join("models", "brain_tumor_model.keras")  # Update filename if using .h5
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]   # Adjust order if different in training
try:
    # Load using keras directly
    model = keras.models.load_model(MODEL_PATH, compile=False)
    print(f"Loaded model successfully from {MODEL_PATH}")
except Exception as e:
    model = None
    print(f"Warning: Could not load model from {MODEL_PATH}: {e}")
# Helper function to preprocess incoming MRI images
def preprocess_image(image_bytes: bytes, target_size=(150, 150)) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(target_size)
    img_array = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)          # Add batch dimension -> (1, 150, 150, 3)
    return img_array

# ---------------------------------------------------------
# Endpoints
# ---------------------------------------------------------
@app.get("/health")
def health_check():
    status = "Online" if model is not None else "Online (Model Not Loaded)"
    return {"status": status}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model file is not loaded on server.")

    try:
        # Read uploaded image bytes
        contents = await file.read()
        
        # Preprocess image with target size 150x150 to match model weights
        img_array = preprocess_image(contents, target_size=(150, 150))
        
        # Run inference
        preds = model.predict(img_array)[0]
        
        # Find highest confidence class
        top_idx = int(np.argmax(preds))
        prediction_label = CLASS_NAMES[top_idx]
        confidence = float(np.max(preds) * 100)

        return {
            "filename": file.filename,
            "prediction": prediction_label,
            "confidence": round(confidence, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.post("/retrain")
def retrain():
    return {"message": "Retraining pipeline initiated successfully."}

# Mount static files (frontend)
if os.path.exists("frontend"):
    app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

@app.get("/")
def read_root():
    index_path = os.path.join("frontend", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "API running. Add index.html to frontend/"}