import os
import io
import numpy as np
import streamlit as st
from PIL import Image
import keras

# ---------------------------------------------------------
# Page Configuration & Custom CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="centered"
)

# Custom Styling for clean UI layout and restricted image sizes
st.markdown("""
    <style>
    /* Cap uploaded image height and width to keep layout compact */
    .stImage img {
        max-width: 350px !important;
        margin: 0 auto;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    /* Style prediction result cards */
    .result-card {
        padding: 20px;
        border-radius: 10px;
        margin-top: 15px;
        margin-bottom: 20px;
        text-align: center;
    }
    .tumor-detected {
        background-color: #fff3cd;
        border-left: 6px solid #ffc107;
        color: #856404;
    }
    .no-tumor {
        background-color: #d4edda;
        border-left: 6px solid #28a745;
        color: #155724;
    }
    </style>
""", unsafe_allow_html=True)

# Dataset class list in standard Keras alphabetical order
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
MODEL_PATH = os.path.join("models", "brain_tumor_model.keras")

# ---------------------------------------------------------
# Load Model (Cached for Speed)
# ---------------------------------------------------------
@st.cache_resource
def load_brain_tumor_model():
    if os.path.exists(MODEL_PATH):
        try:
            model = keras.models.load_model(MODEL_PATH, compile=False)
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    return None

model = load_brain_tumor_model()

# ---------------------------------------------------------
# Preprocessing Pipeline
# ---------------------------------------------------------
def preprocess_image(image: Image.Image, target_size=(150, 150)) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = np.array(image, dtype=np.float32) / 255.0  # Rescale to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)          # Batch shape (1, 150, 150, 3)
    return img_array

# ---------------------------------------------------------
# Main User Interface
# ---------------------------------------------------------
st.title(" Brain Tumor Classification")
st.write("Upload an MRI scan to perform classification across tumor categories.")

if model is None:
    st.error(
        f"**Model file missing or failed to load.**\n\n"
        f"Please verify that `brain_tumor_model.keras` exists inside the `models/` directory."
    )
else:
    uploaded_file = st.file_uploader("Upload MRI Scan Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Center image display using Streamlit columns
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Uploaded MRI Scan", use_container_width=False)

        if st.button("Classify Scan", type="primary", use_container_width=True):
            with st.spinner("Analyzing MRI scan..."):
                processed_img = preprocess_image(image)
                preds = model.predict(processed_img)[0]
                
                top_idx = int(np.argmax(preds))
                prediction_label = CLASS_NAMES[top_idx]
                confidence = float(np.max(preds) * 100)

            st.markdown("---")
            
            # Custom styled output block
            if prediction_label == "notumor":
                st.markdown(
                    f"""
                    <div class="result-card no-tumor">
                        <h3>Result: NO TUMOR DETECTED</h3>
                        <p style="font-size: 18px; margin:0;">Confidence: <b>{confidence:.2f}%</b></p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-card tumor-detected">
                        <h3>Result: {prediction_label.upper()} TUMOR DETECTED</h3>
                        <p style="font-size: 18px; margin:0;">Confidence: <b>{confidence:.2f}%</b></p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

            # Interactive confidence breakdown for verification
            with st.expander("📊 View Probability Breakdown Across All Classes", expanded=True):
                for idx, class_name in enumerate(CLASS_NAMES):
                    prob = float(preds[idx] * 100)
                    st.write(f"**{class_name.capitalize()}**: {prob:.2f}%")
                    st.progress(int(prob))