import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from tqdm import tqdm

def load_data(data_path, subset_size=1000, img_size=(64, 64)):
    """
    Load a balanced subset of cat and dog images, preprocess them, and extract HOG features.
    
    Args:
        data_path (str): Path to PetImages directory.
        subset_size (int): Number of images to load per class.
        img_size (tuple): Target image dimensions (width, height).
        
    Returns:
        features (np.array): Extracted HOG features.
        labels (np.array): Encoded labels (0 for Cat, 1 for Dog).
        sample_images (dict): A few raw images for visualization.
    """
    features = []
    labels = []
    sample_images = {'Cat': [], 'Dog': []}
    
    classes = ['Cat', 'Dog']
    
    for idx, category in enumerate(classes):
        category_path = os.path.join(data_path, category)
        image_names = os.listdir(category_path)
        
        # Shuffle to get a random subset if needed, but here we just take the first valid ones
        count = 0
        print(f"Loading {category} images...")
        
        for img_name in tqdm(image_names):
            if count >= subset_size:
                break
                
            img_path = os.path.join(category_path, img_name)
            
            try:
                # Read image
                img = cv2.imread(img_path)
                if img is None:
                    continue # Skip corrupt or invalid images
                
                # Resize
                img_resized = cv2.resize(img, img_size)
                
                # Convert to Grayscale
                img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
                
                # Feature Extraction: HOG
                # orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2) are standard for HOG
                hog_features = hog(img_gray, orientations=9, pixels_per_cell=(8, 8), 
                                   cells_per_block=(2, 2), visualize=False)
                
                features.append(hog_features)
                labels.append(idx) # 0 for Cat, 1 for Dog
                
                # Save a few for the app display
                if count < 5:
                    sample_images[category].append(img_resized)
                
                count += 1
                
            except Exception as e:
                # Handle potential corrupted images in the Kaggle dataset
                continue
                
    return np.array(features), np.array(labels), sample_images

def extract_single_hog(img_path, img_size=(64, 64)):
    """
    Extract HOG features for a single image (useful for inference).
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Could not read image")
    
    img_resized = cv2.resize(img, img_size)
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    hog_features = hog(img_gray, orientations=9, pixels_per_cell=(8, 8), 
                       cells_per_block=(2, 2), visualize=False)
    
    return hog_features.reshape(1, -1), img_resized
