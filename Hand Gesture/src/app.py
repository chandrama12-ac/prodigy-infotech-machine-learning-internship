import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
import os

# Page Config
st.set_page_config(page_title="Hand Gesture Recognition", page_icon="✋", layout="wide")

# CSS for better UI
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
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
        border-radius: 10px;
        background-color: #262730;
        text-align: center;
        border: 2px solid #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)

# Configuration & Constants
MODEL_PATH = 'models/hand_gesture_model.h5'
PLOT_PATH = 'models/training_curves.png'
GESTURES = {
    0: 'Palm', 1: 'L', 2: 'Fist', 3: 'Fist Moved', 4: 'Thumb',
    5: 'Index', 6: 'OK', 7: 'Palm Moved', 8: 'C', 9: 'Down'
}

@st.cache_resource
def load_trained_model():
    if os.path.exists(MODEL_PATH):
        return load_model(MODEL_PATH)
    return None

def preprocess_image(image):
    # Convert PIL to numpy
    img = np.array(image.convert('L')) # Grayscale
    img = cv2.resize(img, (64, 64))
    img = img.astype('float32') / 255.0
    img = img.reshape(1, 64, 64, 1)
    return img

def main():
    st.title("✋ Hand Gesture Recognition System")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("Navigation")
    app_mode = st.sidebar.selectbox("Choose Mode", ["Project Overview", "Image Prediction", "Real-time Setup Guide"])
    
    model = load_trained_model()
    
    if app_mode == "Project Overview":
        st.header("Project Overview")
        st.info("This project uses a Deep Convolutional Neural Network (CNN) to recognize hand gestures from the Leap GestRecog dataset.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Gesture Classes")
            st.write(", ".join(GESTURES.values()))
            
        with col2:
            st.subheader("Model Status")
            if model:
                st.success("Model Loaded Successfully!")
            else:
                st.error("Model Not Found. Please run `train.py` first.")
                
        if os.path.exists(PLOT_PATH):
            st.subheader("Training performance")
            st.image(PLOT_PATH)
            
    elif app_mode == "Image Prediction":
        st.header("Image Upload & Prediction")
        
        if not model:
            st.warning("Please train the model first to enable predictions.")
            return
            
        uploaded_file = st.file_uploader("Upload a hand gesture image...", type=["jpg", "png", "jpeg"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_container_width=True)
            
            if st.button("Predict Gesture"):
                processed_img = preprocess_image(image)
                prediction = model.predict(processed_img)
                class_idx = np.argmax(prediction)
                confidence = prediction[0][class_idx]
                
                gesture_name = GESTURES[class_idx]
                
                st.markdown(f"""
                <div class="prediction-card">
                    <h1>Predicted: {gesture_name}</h1>
                    <h3>Confidence: {confidence*100:.2f}%</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Show Probabilities
                st.subheader("Class Probabilities")
                for i, prob in enumerate(prediction[0]):
                    st.progress(float(prob), text=f"{GESTURES[i]}: {prob*100:.1f}%")

    elif app_mode == "Real-time Setup Guide":
        st.header("Real-time Webcam Instructions")
        st.markdown("""
        1. Ensure your webcam is connected.
        2. Run the following command in your terminal:
           ```bash
           python src/realtime_inference.py
           ```
        3. A window will open showing your webcam feed.
        4. Place your hand inside the **Green Box**.
        5. Press **'q'** to exit the real-time recognition window.
        """)
        st.info("💡 **No Webcam?** The script now includes a **Demo Mode**. If no camera is detected, it will automatically switch to using random samples from the dataset so you can still test the predictions!")
        st.warning("Note: Browsers have limitations with direct webcam access in Streamlit locally. Using the standalone OpenCV script is recommended for better performance.")

if __name__ == "__main__":
    main()
