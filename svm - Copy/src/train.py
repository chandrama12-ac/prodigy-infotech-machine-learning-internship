import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from preprocessing import load_data

def train_svm_model(data_path, models_dir='../models'):
    """
    Load data, train SVM with hyperparameter tuning, and save the model.
    """
    # 1. Load Data (subset_size=1000 per class = 2000 images)
    print("Step 1: Loading data and extracting HOG features...")
    X, y, _ = load_data(data_path, subset_size=1000)
    
    # 2. Split Data (80% train, 20% test)
    print("\nStep 2: Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Scaling (Important for SVM)
    print("Step 3: Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Hyperparameter Tuning using GridSearchCV
    print("\nStep 4: Tuning hyperparameters using GridSearchCV...")
    # Smaller grid for the demo (RBF vs Linear)
    param_grid = [
        {'C': [1, 10], 'kernel': ['linear']},
        {'C': [1, 10], 'gamma': [0.001, 0.01], 'kernel': ['rbf']}
    ]
    
    svc = SVC(probability=True) # probability=True for Streamlit app confidence display
    grid_search = GridSearchCV(svc, param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"\nBest parameters found: {grid_search.best_params_}")
    
    # 5. Evaluation
    print("\nStep 5: Evaluating the model...")
    y_pred = best_model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Cat', 'Dog']))
    
    # 6. Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'])
    plt.title('Confusion Matrix: Cats vs Dogs SVM')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(models_dir, 'confusion_matrix.png'))
    print(f"\nConfusion matrix saved to {os.path.join(models_dir, 'confusion_matrix.png')}")
    
    # 7. Save Model and Scaler
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    joblib.dump(best_model, os.path.join(models_dir, 'svm_model.pkl'))
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    print(f"\nModel and scaler saved to {models_dir}")
    
    return best_model, scaler

if __name__ == '__main__':
    # Ensure relative paths work if run from this directory or root
    DATA_PATH = os.path.abspath("../kagglecatsanddogs_3367a/PetImages")
    MODELS_DIR = os.path.abspath("../models")
    
    if not os.path.exists(DATA_PATH):
        # Fallback for different execution contexts
        DATA_PATH = os.path.abspath("kagglecatsanddogs_3367a/PetImages")
        MODELS_DIR = os.path.abspath("models")
        
    train_svm_model(DATA_PATH, MODELS_DIR)
