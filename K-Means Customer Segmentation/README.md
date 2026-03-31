# 🛍️ Customer Segmentation using K-Means Clustering

Welcome to the **Customer Segmentation** project! This is a complete end-to-end Machine Learning pipeline that uses the K-Means algorithm to group retail mall customers based on their Annual Income and Spending Score. 

The project includes an interactive **Streamlit Web Application** to predict the cluster for any new customer and provides actionable business insights for targeted marketing.

---

## 🎯 Project Features
1. **Data Preprocessing & Scaling**: Cleans and scales the data using `StandardScaler`.
2. **Exploratory Data Analysis**: Generates scatter plots to visually understand the relationship between income and spending.
3. **Elbow Method**: Automatically determines the optimal number of clusters (k=5).
4. **K-Means Clustering**: Trains the model and saves it using `pickle` for inference.
5. **Streamlit Deployment**: A beautiful, user-friendly UI to input new customer data and instantly get their segment profile.

---

## 💼 Business Insights (The 5 Customer Segments)

Based on the algorithm, our customers naturally fall into **5 distinct groups**:

1. 🎯 **Target Customers (High Income, High Spending)**
   - These are your most valuable customers. They earn a lot and spend a lot. 
   - **Strategy:** Prioritize them! Send premium offers, VIP loyalty programs, and exclusive product launches.

2. ⚠️ **Careful Customers (High Income, Low Spending)**
   - They have the money but are very cautious about spending it. 
   - **Strategy:** Offer them quality-focused, high-value promotions. They respond well to guarantees and premium services over simple discounts.

3. 💸 **Impulsive Customers (Low Income, High Spending)**
   - They love shopping despite having lower incomes.
   - **Strategy:** Promote attractive sales, easy-payment plans, and lifestyle-driven marketing.

4. 🛡️ **Sensible Customers (Low Income, Low Spending)**
   - They earn less and spend cautiously out of necessity.
   - **Strategy:** Don't target them too aggressively. Offer budget-friendly options and essential commodities.

5. ⭐ **Standard Customers (Average Income, Average Spending)**
   - The middle ground. They represent the typical, everyday shopper.
   - **Strategy:** Standard marketing strategies apply. Keep them engaged with seasonal promotions.

---

## 🚀 How to Run the Project Locally

### 1. Install Dependencies
Make sure you have Python installed. Open your terminal or command prompt in the project directory and run:

```bash
pip install -r requirements.txt
```

### 2. Train the Model
First, we need to train the K-Means model to generate the necessary `.pkl` files and visualization plots.

```bash
python train_model.py
```
*This will create a `models/` directory with the saved model and scaler, and a `plots/` directory with EDA and Cluster visualizations.*

### 3. Run the Streamlit Web App
Launch the interactive web application to see the predictions and insights in action!

```bash
streamlit run app.py
```
*This will automatically open the app in your default web browser.*

---

## 📁 Files Included
- `train_model.py`: Core machine learning script for training and exporting the model.
- `app.py`: Streamlit frontend application.
- `requirements.txt`: Python package dependencies.
- `Mall_Customers.csv`: The dataset.
- `models/`: Folder containing `kmeans_model.pkl` and `scaler.pkl`.
- `plots/`: Folder containing generated matplotlib charts.

Enjoy segmenting your customers! 🚀
