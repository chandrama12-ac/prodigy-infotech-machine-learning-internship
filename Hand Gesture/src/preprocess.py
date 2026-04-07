import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Configuration
IMG_SIZE = 64
DATA_DIR = 'leapGestRecog'

def load_data(data_dir=DATA_DIR):
    """
    Loads images and labels from the Leap Gesture Recognition dataset directory.
    Structure: leapGestRecog/00/01_palm/*.png
    """
    X = []
    y = []
    
    # The dataset has 10 folders (00 to 09) for each person
    person_folders = [f for f in os.listdir(data_dir) if f.isdigit() and len(f) == 2]
    
    for person in person_folders:
        person_path = os.path.join(data_dir, person)
        
        # Each person folder has 10 gesture folders
        for gesture in os.listdir(person_path):
            gesture_path = os.path.join(person_path, gesture)
            if not os.path.isdir(gesture_path):
                continue
            
            # Extract the label from the folder name (e.g., '01_palm' -> 0)
            # Labels in the dataset are indexed 01 to 10
            label = int(gesture.split('_')[0]) - 1 
            
            for img_file in os.listdir(gesture_path):
                img_path = os.path.join(gesture_path, img_file)
                
                try:
                    # Read image as grayscale
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                        
                    # Resize to target size
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    
                    X.append(img)
                    y.append(label)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
                    
    X = np.array(X, dtype='float32') / 255.0  # Normalize
    X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # Reshape for CNN
    y = to_categorical(y, num_classes=10) # One-hot encoding
    
    return X, y

def get_train_val_test_splits():
    """
    Convenience function to get splits.
    """
    print("Loading and preprocessing data... this may take a while.")
    X, y = load_data()
    
    # First split: Train (80%) and Temp (20%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Second split: Validation (10%) and Test (10%) from Temp
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print(f"Data Loaded Successfully!")
    print(f"Train set: {X_train.shape[0]} images")
    print(f"Val set: {X_val.shape[0]} images")
    print(f"Test set: {X_test.shape[0]} images")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    # Test loading
    X_train, X_val, X_test, y_train, y_val, y_test = get_train_val_test_splits()
