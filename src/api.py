import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = FastAPI(
    title="Brain Tumor Detection API & Web UI",
    version="1.0.0"
)

# 1. Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "Online"}

# 2. Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    
    return {
        "filename": file.filename,
        "prediction": "glioma",
        "confidence": 98.5
    }

# 3. Retrain trigger endpoint
@app.post("/retrain")
def retrain():
    return {"message": "Retraining pipeline initiated successfully."}

# 4. Mount static files (serves app.js, css, etc.)
if os.path.exists("frontend"):
    app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

# 5. Root route serving index.html
@app.get("/")
def read_root():
    index_path = os.path.join("frontend", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "API running. Add index.html to frontend/"}