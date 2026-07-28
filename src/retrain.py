import os
import sqlite3
import datetime
from PIL import Image
import numpy as np
import tensorflow as tf

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "..", "data", "retrain_metadata.db")
UPLOAD_DIR = os.path.join(BASE_DIR, "..", "data", "uploaded_for_retrain")
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "brain_tumor_model.keras")

CLASS_NAMES = ["Glioma Tumor", "Meningioma Tumor", "No Tumor", "Pituitary Tumor"]

# -----------------------------------------------------------------------------
# TRIGGER 1: DATA FILE UPLOADING + SAVING TO DATABASE
# -----------------------------------------------------------------------------
def init_db():
    """Create a database table to record incoming training files."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS retrain_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            filepath TEXT,
            label TEXT,
            uploaded_at TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_uploaded_image_and_metadata(uploaded_file, label: str):
    """Saves the raw uploaded image file and logs its metadata to the SQLite database."""
    init_db()
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    filename = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uploaded_file.name}"
    filepath = os.path.join(UPLOAD_DIR, filename)
    
    # Save file to disk
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    # Log metadata to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO retrain_images (filename, filepath, label, uploaded_at) VALUES (?, ?, ?, ?)",
        (filename, filepath, label, datetime.datetime.now())
    )
    conn.commit()
    conn.close()
    print(f"[Trigger 1] Saved {filename} and logged entry in SQLite database.")
    return filepath

# -----------------------------------------------------------------------------
# TRIGGER 2: DATA PREPROCESSING OF UPLOADED DATA
# -----------------------------------------------------------------------------
def preprocess_retrain_data(target_size=(224, 224)):
    """Fetches uploaded image records from DB and preprocesses them into tensors."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT filepath, label FROM retrain_images")
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        print("[Trigger 2] No new data found in database for preprocessing.")
        return None, None
        
    x_data, y_data = [], []
    for filepath, label in rows:
        if os.path.exists(filepath) and label in CLASS_NAMES:
            img = Image.open(filepath).convert("RGB")
            img = img.resize(target_size)
            img_arr = np.array(img, dtype=np.float32) / 255.0  # Normalize pixel values
            
            x_data.append(img_arr)
            y_data.append(CLASS_NAMES.index(label))
            
    if len(x_data) == 0:
        return None, None
        
    X = np.array(x_data)
    y = tf.keras.utils.to_categorical(np.array(y_data), num_classes=len(CLASS_NAMES))
    print(f"[Trigger 2] Preprocessed {len(X)} image tensors for fine-tuning.")
    return X, y

# -----------------------------------------------------------------------------
# TRIGGER 3: RETRAINING USING CUSTOM MODEL AS PRE-TRAINED BASE
# -----------------------------------------------------------------------------
def run_retraining_pipeline():
    """Loads existing custom model as a pre-trained base and fine-tunes on new data."""
    X, y = preprocess_retrain_data()
    
    if X is None or len(X) == 0:
        print("[Trigger 3] Skipping retraining: Insufficient data samples.")
        return {"status": "skipped", "reason": "No preprocessed data in DB"}
        
    print("[Trigger 3] Loading existing custom model as pre-trained model...")
    pretrained_model = tf.keras.models.load_model(MODEL_PATH)
    
    # Freeze initial feature extraction layers; keep top classification layer trainable
    for layer in pretrained_model.layers[:-2]:
        layer.trainable = False
        
    pretrained_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # Fine-tune model on new uploaded data
    print("[Trigger 3] Executing model retraining...")
    history = pretrained_model.fit(X, y, epochs=3, batch_size=4, verbose=1)
    
    # Save updated retrained model weights
    pretrained_model.save(MODEL_PATH)
    print(f"[Trigger 3] Retraining complete. Updated model saved to {MODEL_PATH}")
    
    return {
        "status": "success",
        "samples_trained": len(X),
        "final_loss": float(history.history["loss"][-1]),
        "final_accuracy": float(history.history["accuracy"][-1])
    }