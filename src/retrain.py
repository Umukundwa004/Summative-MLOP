import os
import sqlite3
import numpy as np
from PIL import Image
import tensorflow as tf
import keras

from src.prediction import load_model, get_model_target_size, CLASS_NAMES

# Database and directory paths
DB_PATH = os.path.join("data", "retrain_metadata.db")
UPLOAD_DIR = os.path.join("data", "uploads")


def init_db():
    """
    Ensures data directories and SQLite schema exist.
    Recreates table if legacy schema is missing the file_path column.
    """
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check existing table columns if table exists
    cursor.execute("PRAGMA table_info(retrain_images)")
    columns = [col[1] for col in cursor.fetchall()]
    
    # If table exists but lacks file_path column, drop and recreate
    if columns and "file_path" not in columns:
        cursor.execute("DROP TABLE retrain_images")
        conn.commit()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS retrain_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT NOT NULL,
            label TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def save_uploaded_image_and_metadata(uploaded_file, label: str):
    """
    Saves an uploaded Streamlit file to local disk and records its path and metadata in SQLite.
    """
    init_db()
    
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO retrain_images (file_path, label) VALUES (?, ?)",
        (file_path, label)
    )
    conn.commit()
    conn.close()


def load_retrain_data(target_size: tuple):
    """
    Loads saved image records from SQLite, resizes images to the model's expected 
    target_size (e.g. 150x150), and builds normalized numpy feature and target arrays.
    """
    init_db()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT file_path, label FROM retrain_images")
    records = cursor.fetchall()
    conn.close()

    if not records:
        return None, None

    images = []
    labels = []

    # Map label strings to integer indices matching CLASS_NAMES
    label_to_idx = {name.lower(): idx for idx, name in enumerate(CLASS_NAMES)}

    for file_path, label in records:
        if os.path.exists(file_path):
            try:
                img = Image.open(file_path)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                
                # Resize image precisely to model target dimension (Width, Height)
                img = img.resize((target_size[1], target_size[0]))
                
                # Normalize pixel intensities
                img_array = np.array(img, dtype=np.float32) / 255.0
                images.append(img_array)

                # Match label string to index (defaulting to 0 if unmatched)
                clean_label = label.strip().lower()
                idx = label_to_idx.get(clean_label, 0)
                labels.append(idx)
            except Exception as e:
                print(f"Error loading image {file_path}: {e}")

    if not images:
        return None, None

    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    return X, y


def run_retraining_pipeline():
    """
    Executes incremental fine-tuning on the existing base model using newly added records.
    Dynamically aligns tensor resolution to avoid dense layer shape mismatch.
    """
    # 1. Load existing trained model
    model = load_model()
    
    # 2. Extract input dimensions expected by model architecture (e.g., 150, 150)
    target_size = get_model_target_size(model)

    # 3. Load and preprocess training samples
    X_train, y_train = load_retrain_data(target_size)

    if X_train is None or len(X_train) == 0:
        return {"status": "no_data", "message": "No new records available for retraining."}

    # 4. Compile model with a conservative learning rate for transfer learning / fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"]
    )

    # 5. Execute fine-tuning epochs
    history = model.fit(
        X_train, 
        y_train, 
        epochs=3, 
        batch_size=min(16, len(X_train)), 
        verbose=1
    )

    # 6. Save updated weights back to disk
    save_path = "models/brain_tumor_model.keras"
    os.makedirs("models", exist_ok=True)
    
    try:
        model.save(save_path)
    except Exception:
        # Fallback save using standard Keras API
        keras.models.save_model(model, save_path)

    final_acc = float(history.history["accuracy"][-1]) if "accuracy" in history.history else 1.0

    return {
        "status": "success",
        "final_accuracy": final_acc,
        "samples_trained": len(X_train)
    }