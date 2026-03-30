import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_and_evaluate(data_path="data/train.csv", model_dir="models", output_dir="outputs"):
    """Loads data, preprocesses, trains a linear regression model, and evaluates it."""
    print("Loading dataset...")
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}!")
        return
        
    df = pd.read_csv(data_path)
    
    # 1. Select relevant features
    features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
    target = 'SalePrice'
    
    # Ensure columns exist
    missing_cols = [col for col in features + [target] if col not in df.columns]
    if missing_cols:
        print(f"Missing columns in dataset: {missing_cols}")
        return
        
    df_selected = df[features + [target]].copy()
    
    # 2. Handle missing values
    print("\n--- Handling Missing Values ---")
    print(f"Missing values before imputation:\n{df_selected.isnull().sum()}")
    
    # Fill missing values with median
    for col in features:
        df_selected[col] = df_selected[col].fillna(df_selected[col].median())
    
    print("Missing values after imputation:\n", df_selected.isnull().sum())
    
    # Remove outliers: In Kaggle's house prices, GrLivArea > 4000 are recommended to be removed
    print("\n--- Outlier Handling ---")
    print(f"Records before outlier removal: {len(df_selected)}")
    df_selected = df_selected[df_selected['GrLivArea'] < 4000]
    print(f"Records after outlier removal: {len(df_selected)}")
    
    # 3. Train-Test Split (80/20)
    X = df_selected[features]
    y = df_selected[target]
    
    print("\nSplitting data into 80% train and 20% test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Normalize / Scale numerical features
    print("Scaling features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Model Implementation
    print("\nTraining Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Show coefficients
    print("\n--- Model Parameters ---")
    print(f"Intercept: {model.intercept_:.2f}")
    for feature, coef in zip(features, model.coef_):
        print(f"Coefficient for {feature}: {coef:.2f}")
        
    # 6. Evaluation
    y_pred = model.predict(X_test_scaled)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- Evaluation Metrics ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    # Save Actual vs Predicted Graph
    print("\nSaving True vs Predicted graph...")
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='royalblue')
    
    # Ideal prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect Prediction")
    
    plt.xlabel("Actual Sale Price")
    plt.ylabel("Predicted Sale Price")
    plt.title("Actual vs Predicted House Prices")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "actual_vs_predicted.png"))
    plt.close()
    
    # 7. Save the Model and Scaler using pickle
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "linear_model.pkl"), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join(model_dir, "scaler.pkl"), 'wb') as f:
        pickle.dump(scaler, f)
        
    print(f"\n✅ Model and scaler saved successfully to '{model_dir}/'.")

if __name__ == "__main__":
    train_and_evaluate()
