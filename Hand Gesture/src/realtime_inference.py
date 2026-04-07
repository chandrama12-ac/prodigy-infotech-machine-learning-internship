import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import random
import glob

# Configuration
MODEL_PATH = 'models/hand_gesture_model.h5'
IMG_SIZE = 64

# Gesture labels (matching training)
GESTURES = {
    0: 'Palm', 
    1: 'L', 
    2: 'Fist', 
    3: 'Fist Moved', 
    4: 'Thumb', 
    5: 'Index', 
    6: 'OK', 
    7: 'Palm Moved', 
    8: 'C', 
    9: 'Down'
}

def run_demo_mode(model):
    """
    Simulates a feed by picking a random image from the dataset.
    """
    data_path = os.path.join('leapGestRecog', '**', '*.png')
    all_images = glob.glob(data_path, recursive=True)
    
    if not all_images:
        print("Error: No images found in 'leapGestRecog' for demo mode.")
        return

    print(f"Demo Mode: Cycling through {len(all_images)} samples. Press 'q' to quit.")
    
    while True:
        img_path = random.choice(all_images)
        frame = cv2.imread(img_path)
        if frame is None: continue
        
        # Preprocess for prediction
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        normalized = resized.astype('float32') / 255.0
        reshaped = normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)
        
        # Predict
        prediction = model.predict(reshaped, verbose=0)
        class_idx = np.argmax(prediction)
        confidence = prediction[0][class_idx]
        label = GESTURES.get(class_idx, "Unknown")
        
        # Display Result
        display_frame = cv2.resize(frame, (400, 400)) # Larger view for demo
        text = f"DEMO - Predicted: {label} ({confidence*100:.1f}%)"
        cv2.putText(display_frame, text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, "Press 'Space' for NEXT, 'q' to QUIT", (10, 380), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow('Hand Gesture Demo (No Webcam Found)', display_frame)
        
        key = cv2.waitKey(0) & 0xFF  # Wait for key press
        if key == ord('q'):
            break
        # Else continue to next random image

    cv2.destroyAllWindows()

def realtime_display():
    # 1. Load trained model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. Please run training script first.")
        return
        
    model = load_model(MODEL_PATH)
    print("Model loaded successfully. Starting webcam...")
    
    # 2. Access Webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Warning: Could not open webcam. Switching to 'Demo Mode' using dataset samples...")
        run_demo_mode(model)
        return
    
    print("Webcam started. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        
        # 3. Define Region of Interest (ROI) - a square box in the center
        height, width, _ = frame.shape
        top, left, bottom, right = 100, 100, 400, 400
        roi = frame[top:bottom, left:right]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # 4. Preprocess ROI
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        normalized = resized.astype('float32') / 255.0
        reshaped = normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)
        
        # 5. Predict
        prediction = model.predict(reshaped, verbose=0)
        class_idx = np.argmax(prediction)
        confidence = prediction[0][class_idx]
        
        label = GESTURES.get(class_idx, "Unknown")
        
        # 6. Display Result
        text = f"{label} ({confidence*100:.1f}%)"
        cv2.putText(frame, text, (left, top - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow('Hand Gesture Recognition (Press Q to quit)', frame)
        
        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    realtime_display()
