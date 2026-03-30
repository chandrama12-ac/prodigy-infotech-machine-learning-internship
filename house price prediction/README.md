# 🏠 House Price Prediction Project

An end-to-end Machine Learning project that predicts house prices based on Square Footage, Bedrooms, and Bathrooms. Built using Python, Scikit-Learn, and Streamlit.

## 📂 Project Structure
```text
house-price-prediction/
│
├── data/                   # Place your Kaggle train.csv here
├── models/                 # Saved pickle models and scalers (generated automatically)
├── outputs/                # EDA graphs and evaluation charts (generated automatically)
├── src/                    # Source code scripts
│   ├── generate_dummy_data.py # Optional: Generates dummy test data
│   ├── eda.py              # Script for Exploratory Data Analysis
│   └── model.py            # Data preprocessing, model training, and evaluation
├── app.py                  # Streamlit web app for deployment
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## 🚀 Setup Instructions

1. **Install Dependencies:**
   Ensure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare the Dataset:**
   - Download the "House Prices - Advanced Regression Techniques" dataset from [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data).
   - Alternatively, you can run the dummy data generator for quick testing:
     ```bash
     python src/generate_dummy_data.py
     ```
   - Place the `train.csv` file inside the `data/` directory.

3. **Run Exploratory Data Analysis (EDA):**
   ```bash
   python src/eda.py
   ```
   Visualizations will be saved inside the `outputs/` folder.

4. **Train the Model:**
   ```bash
   python src/model.py
   ```
   This handles data imputations, scaling, outlier removal, and training using `LinearRegression`. It will save the compiled model as a `.pkl` file into the `models/` directory.

5. **Start the Web Application:**
   ```bash
   streamlit run app.py
   ```
   This will run a local web server displaying a beautiful user interface to predict exact house values.

## 📊 Methodology and Features
- **Data Preprocessing**: Handling missing values using median imputations, removal of outliers (GrLivArea > 4000 sqft), using standard normalization via `StandardScaler`.
- **Exploratory Data Analysis**: Contains correlation heatmaps and bivariate scatter plots for deep understanding.
- **Model**: Linear Regression. Uses variables `GrLivArea`, `BedroomAbvGr`, and `FullBath` to predict the sale price.
- **Evaluation**: Computes Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² Accuracy.
