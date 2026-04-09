import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from src.calorie_map import get_calories
from src.utils import preprocess_image, get_top_predictions

# Page config
st.set_page_config(
    page_title="AI Food & Calorie Tracker",
    page_icon="🍲",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 15px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .calorie-box {
        font-size: 2.5em;
        font-weight: bold;
        color: #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)

# App Title
st.title("🍲 AI Food Recognition & Calorie Estimation")
st.markdown("---")

# Sidebar
st.sidebar.title("Settings")
model_choice = st.sidebar.selectbox("Select Model Architecture", ["MobileNetV2", "ResNet50", "ScratchModel"])
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.0)

# Load labels
@st.cache_resource
def load_labels():
    with open("src/class_labels.txt", "r") as f:
        return [line.strip() for line in f.readlines()]

class_labels = load_labels()

# Load Model
def load_model(name):
    model_path = f"models/{name}_best.keras"
    if os.path.exists(model_path):
        try:
            return tf.keras.models.load_model(model_path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    else:
        st.warning(f"No trained model found at {model_path}. Please train (src/train.py) or generate a mock/pre-trained model.")
        return None

model = load_model(model_choice)

# Main UI
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📸 Upload Food Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

with col2:
    st.subheader("🧠 Prediction Results")
    
    if uploaded_file is not None:
        if model is not None:
            with st.spinner("Analyzing image..."):
                # Preprocess
                img_array = preprocess_image(image)
                
                # Predict
                top_preds = get_top_predictions(model, img_array, class_labels, k=3)
                best_pred = top_preds[0]
                
                if best_pred['confidence'] >= confidence_threshold:
                    # Display Result
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h3>Predicted Class</h3>
                        <h2 style='color: #2e7d32;'>{best_pred['label'].replace('_', ' ').title()}</h2>
                        <p>Confidence: <b>{best_pred['confidence']:.2%}</b></p>
                        <hr>
                        <h3>Estimated Calories</h3>
                        <div class="calorie-box">{get_calories(best_pred['label'])} kcal</div>
                        <p style='color: #666;'>*Estimated per serving</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show Top-3 as chart
                    st.markdown("### Top 3 Predictions")
                    labels = [p['label'].replace('_', ' ').title() for p in top_preds]
                    scores = [p['confidence'] for p in top_preds]
                    st.bar_chart(dict(zip(labels, scores)))
                    
                else:
                    st.warning(f"Low confidence ({best_pred['confidence']:.2%}), but here is the best guess:")
                    st.info(f"Predicted Food: **{best_pred['label'].replace('_', ' ').title()}** | Estimated Calories: **{get_calories(best_pred['label'])} kcal**")
        else:
            st.info("Upload an image and ensure a model is trained to see results.")
    else:
        st.info("Waiting for image upload...")

# Footer
st.markdown("---")
st.markdown("Developed by Senior AI/ML Engineer | Dataset: Food-101")
