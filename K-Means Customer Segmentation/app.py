import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Set page configuration
st.set_page_config(page_title="Customer Segmentation App", page_icon="🛍️", layout="wide")

# Basic styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .cluster-card {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    # Load the saved model and scaler
    model_path = os.path.join('models', 'kmeans_model.pkl')
    scaler_path = os.path.join('models', 'scaler.pkl')
    
    try:
        with open(model_path, 'rb') as f:
            kmeans = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        return kmeans, scaler
    except Exception as e:
        st.error(f"Error loading models. Have you run `train_model.py` yet? Details: {e}")
        return None, None

def get_cluster_profile(kmeans, cluster_id, scaler):
    """
    Dynamically determine the business profile of a cluster based on its centroid.
    """
    centroids_scaled = kmeans.cluster_centers_
    centroids = scaler.inverse_transform(centroids_scaled)
    cluster_centroid = centroids[cluster_id]
    
    income, spending = cluster_centroid[0], cluster_centroid[1]
    
    # Simple heuristic to name clusters based on Mall Customers dataset ranges
    # Income range roughly 15-140k. Mean ~60k
    # Spending range 1-100. Mean ~50
    
    profile = ""
    color = ""
    
    if income > 70 and spending > 70:
        profile = "🎯 Target Customers (High Income, High Spending)"
        color = "#2e7d32" # Green
    elif income > 70 and spending <= 40:
        profile = "⚠️ Careful Customers (High Income, Low Spending)"
        color = "#d84315" # Orange-Red
    elif income <= 40 and spending > 70:
        profile = "💸 Careless/Impulsive Customers (Low Income, High Spending)"
        color = "#c2185b" # Pink-Red
    elif income <= 40 and spending <= 40:
        profile = "🛡️ Sensible Customers (Low Income, Low Spending)"
        color = "#1565c0" # Blue
    else:
        profile = "⭐ Standard Customers (Average Income, Average Spending)"
        color = "#f9a825" # Yellow
        
    return profile, color

def main():
    st.title("🛍️ Customer Segmentation Predictor")
    st.write("Predict which segment a customer belongs to based on their Annual Income and Spending Score.")
    
    kmeans, scaler = load_models()
    
    if kmeans is None or scaler is None:
        st.stop()
        
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Customer Details")
        st.write("Enter the customer's information below:")
        
        income = st.slider("Annual Income (k$)", min_value=10, max_value=150, value=60, step=1)
        spending = st.slider("Spending Score (1-100)", min_value=1, max_value=100, value=50, step=1)
        
        predict_btn = st.button("Predict Segment")
        
    with col2:
        st.subheader("Prediction Result")
        
        if predict_btn:
            # Prepare input data
            input_data = np.array([[income, spending]])
            input_scaled = scaler.transform(input_data)
            
            # Predict
            cluster_id = kmeans.predict(input_scaled)[0]
            
            # Get business interpretation
            profile, bg_color = get_cluster_profile(kmeans, cluster_id, scaler)
            
            # Display result
            st.markdown(f"""
            <div class="cluster-card" style="background-color: {bg_color};">
                <h3>Customer belongs to Cluster {cluster_id}</h3>
                <h4>{profile}</h4>
                <p><strong>Income:</strong> ${income}k | <strong>Spending Score:</strong> {spending}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("---")
            st.write("**Marketing Action:**")
            if "Target" in profile:
                st.success("Prioritize these customers! Send them premium offers and loyalty programs.")
            elif "Careful" in profile:
                st.info("These customers have money but don't spend it easily. Try offering high-value, quality-focused promotions.")
            elif "Impulsive" in profile:
                st.warning("These customers love to spend, but have low income. Promote attractive sales and discounts to them.")
            elif "Sensible" in profile:
                st.info("Don't target these customers too aggressively. They are on a tight budget and spend cautiously.")
            else:
                st.info("These are your regular everyday customers. Standard marketing strategies apply.")
                
    st.write("---")
    st.header("📊 Visualization Insights")
    st.write("Here is the visual mapping of all our customer segments from the training phase:")
    
    try:
        st.image("plots/clusters.png", use_column_width=True)
    except FileNotFoundError:
        st.warning("Cluster visualization plot not found. Make sure to run the training script first!")

if __name__ == '__main__':
    main()
