FROM python:3.10-slim

WORKDIR /app

# Install curl
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Ensure models directory exists
RUN mkdir -p models

# Download model directly from GitHub Releases during Docker build
RUN curl -L -o models/brain_tumor_model.keras "https://github.com/Umukundwa004/Summative-MLOP/releases/download/v1.0.0/brain_tumor_model.keras"

# Copy the rest of the application code
COPY . .

EXPOSE 7860

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "7860"]