import streamlit as st
import pickle
import numpy as np
import os

# --- Page Config ---
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

# --- Custom Styling ---
st.markdown("""
<style>
    .stApp {
        background-color: #f8f9fa;
    }
    div.stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px;
        border: none;
        transition: background-color 0.3s;
    }
    div.stButton > button:hover {
        background-color: #45a049;
    }
    .title {
        text-align: center;
        color: #2c3e50;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    .subtitle {
        text-align: center;
        color: #7f8c8d;
        margin-bottom: 30px;
        font-size: 18px;
    }
    .prediction-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Model and Scaler ---
@st.cache_resource
def load_models():
    model_path = "models/linear_model.pkl"
    scaler_path = "models/scaler.pkl"
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    else:
        return None, None

model, scaler = load_models()

# --- UI Components ---
st.markdown("<h1 class='title'>🏠 House Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Predict the value of your next home based on its features.</p>", unsafe_allow_html=True)

if model is None or scaler is None:
    st.error("Model files not found! Please run `python src/model.py` first to train and save the model.")
else:
    st.markdown("### 📋 Enter Property Details")
    
    # Input fields arranged in columns
    col1, col2 = st.columns(2)
    
    with col1:
        sqft = st.number_input("Square Footage (GrLivArea)", min_value=500, max_value=10000, value=1500, step=50, help="Total above grade living area in square feet.")
        bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3, step=1)
        
    with col2:
        bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=5, value=2, step=1)
        
        # Adding a little spacing
        st.write("")
        st.write("")
        predict_btn = st.button("Predict Price 💰")
    
    st.markdown("---")
    
    # Prediction logic
    if predict_btn:
        # Create input array (needs to match the feature list order used in training: GrLivArea, BedroomAbvGr, FullBath)
        features = np.array([[sqft, bedrooms, bathrooms]])
        
        # Scale the features
        features_scaled = scaler.transform(features)
        
        # Predict
        predicted_price = model.predict(features_scaled)[0]
        
        # Ensure we don't output negative prices
        predicted_price = max(0, predicted_price)
        
        # Display the prediction beautifully
        st.markdown(f"""
        <div class="prediction-box">
            <h3 style="color: #34495e;">Estimated House Value</h3>
            <h1 style="color: #27ae60; font-size: 48px;">${predicted_price:,.2f}</h1>
        </div>
        """, unsafe_allow_html=True)
        
        st.balloons()
        
        st.info("💡 Note: This is an estimated price based on Linear Regression using the House Prices dataset. Actual market values may vary.")
