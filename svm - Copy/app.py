import streamlit as st
import cv2
import numpy as np
import joblib
import os
from PIL import Image
from skimage.feature import hog
from skimage import exposure
from sklearn.preprocessing import StandardScaler

# Page Config
st.set_page_config(page_title="Cat vs Dog Image Classifier - SVM", page_icon="🐾", layout="wide")

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #121212;
        color: white;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
    }
    .stAlert {
        border-radius: 10px;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Helper function for HOG
def extract_hog_features(img_np, img_size=(64, 64)):
    # Resize
    img_resized = cv2.resize(img_np, img_size)
    # Grayscale
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    # HOG
    fd, hog_image = hog(img_gray, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualize=True)
    return fd.reshape(1, -1), hog_image, img_resized

# Load Model and Scaler
@st.cache_resource
def load_resources():
    model_path = os.path.join("models", "svm_model.pkl")
    scaler_path = os.path.join("models", "scaler.pkl")
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    return None, None

# Sidebar Content
st.sidebar.title("🐾 SVM Classifier")
st.sidebar.info("Upload an image of a Cat or Dog to see the prediction!")
st.sidebar.markdown("### Project Details")
st.sidebar.write("**Model:** Support Vector Machine (SVM)")
st.sidebar.write("**Features:** HOG (Histogram of Oriented Gradients)")
st.sidebar.write("**Accuracy:** Check evaluation metrics in report")

# Header
st.title("🐱 Cats vs 🐶 Dogs Classifier")
st.write("A professional machine learning project using **SVM** and **HOG** features.")

# Load resources
model, scaler = load_resources()

if model is None:
    st.error("No trained model found! Please run the training script (`src/train.py`) first.")
else:
    # Create Tabs
    tab1, tab2 = st.tabs(["🔮 Classification", "🧬 HOG Explorer"])

    with tab1:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📤 Upload Image")
            uploaded_file = st.file_uploader("Choose a cat or dog image...", type=["jpg", "jpeg", "png"], key="predict_upload")

            if uploaded_file is not None:
                # Load and display user image
                image = Image.open(uploaded_file)
                img_np = np.array(image)
                st.image(image, caption='Uploaded Image', use_container_width=True)

                if st.button("🔮 Predict Now"):
                    with st.spinner("Analyzing structural features..."):
                        try:
                            # Preprocess and Extract HOG
                            features, hog_viz, resized_img = extract_hog_features(img_np)
                            
                            # Scale
                            features_scaled = scaler.transform(features)
                            
                            # Predict
                            prediction = model.predict(features_scaled)
                            probs = model.predict_proba(features_scaled)[0]
                            
                            # Results
                            class_names = ['Cat', 'Dog']
                            label = class_names[prediction[0]]
                            confidence = probs[prediction[0]] * 100

                            with col2:
                                st.subheader("🎯 Prediction Result")
                                result_color = "#3498db" if label == "Cat" else "#e67e22"
                                st.markdown(f"""
                                    <div style='background-color: {result_color}; padding: 20px; border-radius: 10px; text-align: center;'>
                                        <h2 style='color: white; margin: 0;'>It's a {label.upper()}!</h2>
                                        <p style='color: white; margin: 0; opacity: 0.8;'>Confidence: {confidence:.2f}%</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # HOG Visualization (Summary)
                                st.subheader("🧬 Structural Features")
                                st.image(hog_viz, caption='HOG Visualization', use_container_width=True)
                                st.info("This is the structural 'skeleton' the SVM model uses for classification.")

                        except Exception as e:
                            st.error(f"Error during prediction: {e}")

    with tab2:
        st.subheader("🔍 HOG Feature Detail Explorer")
        st.write("Understand how HOG captures edges, shapes, and textures.")
        
        exp_file = st.file_uploader("Upload any image to visualize its HOG features...", type=["jpg", "jpeg", "png"], key="explore_upload")
        
        if exp_file:
            e_col1, e_col2 = st.columns([1, 2])
            
            with e_col1:
                st.markdown("### ⚙️ HOG Parameters")
                orientations = st.slider("Orientations", 4, 12, 9)
                ppc = st.slider("Pixels per Cell", 4, 16, 8)
                cpb = st.slider("Cells per Block", 1, 4, 2)
                
            with e_col2:
                # Load image
                e_img = Image.open(exp_file)
                e_img_np = np.array(e_img)
                
                # Preprocess
                e_resized = cv2.resize(e_img_np, (128, 128))
                e_gray = cv2.cvtColor(e_resized, cv2.COLOR_RGB2GRAY)
                
                # Extract and Visualize
                fd, hog_image = hog(e_gray, orientations=orientations, 
                                    pixels_per_cell=(ppc, ppc),
                                    cells_per_block=(cpb, cpb), visualize=True)
                
                # --- FIX: ENSURE DATA IS WITHIN [0.0, 1.0] ---
                # Step 1: Best Practice - use exposure.rescale_intensity to map 
                # the actual range of HOG features to the full [0, 1] display range.
                hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range='image', out_range=(0, 1))
                
                # Step 2: Optional Enhancement - apply gamma correction on the 
                # already normalized image for better visibility of faint gradients.
                hog_image_final = np.power(hog_image_rescaled, 0.5) 
                
                # Step 3: Streamlit Display
                # Use 'clamp=True' as an additional safety measure (Alternative Fix)
                st.image(hog_image_final, 
                         caption=f'HOG Visualization (Resized 128x128, {fd.shape[0]} features)', 
                         use_container_width=True,
                         clamp=True)
                
                with st.expander("📝 What are we seeing?"):
                    st.write("""
                    **HOG (Histogram of Oriented Gradients)** works by:
                    1. Calculating the gradient (change in intensity) for every pixel.
                    2. Grouping pixels into cells (e.g., 8x8 pixels).
                    3. For each cell, creating a histogram of gradient orientations (e.g., 9 directions).
                    4. Normalizing blocks of cells to make the descriptor robust to lighting changes.
                    
                    The white lines represent the dominant orientations of edges in that region.
                    """)

    # SVM vs CNN Comparison Section
    st.divider()
    st.subheader("💡 SVM vs CNN: A Brief Comparison")
    
    comp_col1, comp_col2 = st.columns(2)
    
    with comp_col1:
        st.markdown("""
        **Support Vector Machines (SVM)**:
        - ✅ Excellent for small to medium datasets.
        - ✅ Uses handcrafted features like **HOG**.
        - ✅ Mathematically robust and faster to train on CPUs.
        - ❌ Complexity increases sharply with large image datasets.
        - ❌ Higher manual effort in feature engineering.
        """)
        
    with comp_col2:
        st.markdown("""
        **Convolutional Neural Networks (CNN)**:
        - ✅ State-of-the-art for image recognition.
        - ✅ Learns features **automatically** from raw pixels.
        - ✅ Handles huge datasets (Big Data) efficiently on GPUs.
        - ❌ Requires massive data and compute power.
        - ❌ "Black Box" nature—harder to explain why a decision was made.
        """)

    # Footer
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: gray;'>Build by Antigravity AI | Senior ML Engineer Demo</p>", unsafe_allow_html=True)
