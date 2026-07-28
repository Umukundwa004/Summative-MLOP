Markdown
# Brain Tumor Detection API & MLOps Pipeline

An end-to-end Machine Learning Operations (MLOps) application for classifying brain MRI scans into four distinct categories (`glioma`, `meningioma`, `notumor`, `pituitary`). Built with FastAPI, TensorFlow/Keras, and OpenCV, featuring asynchronous image streaming, automated configuration via `.env`, and load-testing capabilities via Locust.

---

## 🛠️ Tech Stack & Requirements

- **Framework**: FastAPI, Uvicorn
- **ML / Computer Vision**: TensorFlow, Keras, OpenCV (`cv2`), NumPy, Pillow
- **Testing & Benchmarking**: Locust, PyTest
- **Configuration & Environment**: `python-dotenv`
- **Deployment**: Docker, Hugging Face Spaces / Render

---
- youtube link:
- brain tumor detection app:https://summative-mlop-8dqfuormvgbx6xdf8yku8z.streamlit.app/
- brain tumor locust testing:https://brain-tumor-locust-testing.onrender.com/

## 📁 Project Directory Structure

```text
Summative-MLOP/
├── data/                  # Dataset storage (raw and test samples)
├── models/                # Trained Keras model weights (.keras / .h5)
│   └── tumor_model.keras  # Primary model file (150x150 input target)
├── src/
│   ├── api.py             # FastAPI REST endpoints (/predict, /retrain, /health)
│   ├── preprocessing.py   # OpenCV image loading, resizing & normalization
│   ├── prediction.py      # Model instantiation & inference handler
│   └── retrain.py         # Retraining orchestration module
├── frontend/              # Web user interface
├── Dockerfile             # Container configuration for production
├── locustfile.py          # Locust load testing configuration
├── .env.example           # Environment variables template
├── .gitignore             # Git exclusion rules
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
⚙️ Section 1: Local Development & Setup
1. Prerequisites
Python: Version 3.10+ (Python 3.11 recommended)

Git

2. Clone Repository & Setup Virtual Environment
On Windows (PowerShell):
PowerShell
```bash
# Clone repository
git clone <your-repository-url>
cd Summative-MLOP
```

# Create and activate virtual environment
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
On macOS / Linux:
```Bash
git clone <your-repository-url>
cd Summative-MLOP

python3 -m venv .venv
```
source .venv/bin/activate
3. Install Dependencies
```Bash
pip install --upgrade pip
pip install -r requirements.txt
```
If requirements.txt is missing or needs updating:

```Bash
pip install fastapi uvicorn tensorflow opencv-python numpy pillow python-dotenv locust pytest
pip freeze > requirements.txt
```
4. Configuration (.env Setup)
Create a local .env file by copying .env.example:

```Bash
cp .env.example .env
```
Verify your .env settings:

Code snippet
```
# Server Configuration
HOST=127.0.0.1
PORT=8000
DEBUG=True
```

# Model Paths & Configurations
```
MODEL_PATH=models/tumor_model.keras
TARGET_IMAGE_SIZE=150
Ensure your trained Keras model is located inside the models/ directory matching MODEL_PATH:
```
Plaintext
```
models/tumor_model.keras
5. Running the API Locally
#Launch the Streamlit App
```
python -m streamlit run app.py
```
 Section 4: Testing API Endpoints
1. Using Swagger UI (/docs)locally 
Navigate to http://127.0.0.1:8000/docs.

Select POST /predict and click Try it out.

Upload an MRI image (.jpg or .png) and click Execute.

2. Using curl
```Bash
curl -X 'POST' \
  '[http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@data/test/sample.jpg'
```
Expected Response
```
JSON
{
  "prediction": "glioma",
  "confidence": 98.5
}
```
🛡️ Section 5: Version Control (.gitignore)
Ensure your .gitignore contains the following rules to prevent committing virtual environments, large model binary files, and environment secrets:

```
# Environment Variables & Secrets
.env
*.env

# Python & Virtual Environments
__pycache__/
*.py[cod]
.venv/
venv/

# Large Model Weights & Datasets
models/*.keras
data/raw/

# Testing & Logs
.pytest_cache/

# IDE & OS Files
.vscode/
```
