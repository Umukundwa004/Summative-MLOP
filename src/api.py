import os
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Import ModelPredictor from prediction.py in the root directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prediction import ModelPredictor

app = FastAPI(
    title="Brain Tumor Detection API",
    version="1.0.0"
)

# Initialize predictor
predictor = ModelPredictor()

# Mount frontend directory if it exists
if os.path.exists("frontend"):
    app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

@app.get("/")
def read_root():
    index_path = os.path.join("frontend", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"status": "Online", "message": "Brain Tumor Detection API is running."}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": predictor.model is not None}

@app.post("/predict")
async def predict_tumor(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((150, 150))
        
        # Preprocess array
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Run inference
        label, confidence = predictor.predict(img_array)
        
        return {
            "prediction": label,
            "confidence": f"{confidence}%"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")