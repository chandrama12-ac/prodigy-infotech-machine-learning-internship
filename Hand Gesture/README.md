# Hand Gesture Recognition System ✋

A complete Machine Learning system to recognize hand gestures using Deep Learning (CNN) and OpenCV.

## 🚀 Features
- **Deep CNN Architecture**: Trained on the Leap GestRecog dataset with over 20,000 images.
- **Data Augmentation**: Robust against shifts, rotations, and zooms.
- **Real-time Recognition**: Standalone OpenCV script for webcam-based prediction.
- **Streamlit GUI**: Interactive dashboard for project overview and single-image testing.
- **Bonus**: Support for MobileNetV2 transfer learning included in code.

## 📂 Project Structure
```text
Hand Gesture/
├── leapGestRecog/             # Dataset directory
├── models/                    # Saved model files
│   ├── hand_gesture_model.h5  # The trained weights
│   └── training_curves.png    # Accuracy & Loss plots
├── src/                       # Source code
│   ├── preprocess.py          # Data loading specialists
│   ├── model.py               # CNN architecture
│   ├── train.py               # Training & Evaluation script
│   ├── realtime_inference.py  # OpenCV webcam implementation
│   └── app.py                 # Streamlit GUI
├── requirements.txt           # Python dependencies
└── README.md                  # Instructions
```

## 🛠️ Setup Instructions

### 1. Install Dependencies
Ensure you have Python installed. Then run:
```bash
pip install -r requirements.txt
```

### 2. Prepare the Dataset
The dataset should be placed in the `leapGestRecog/` folder at the root of the project. (Currently already present in your workspace).

### 3. Training the Model
To train the CNN model from scratch:
```bash
python src/train.py
```
*This will generate `models/hand_gesture_model.h5` and training plots.*

### 4. Real-time Inference
To start the live webcam gesture recognition:
```bash
python src/realtime_inference.py
```
- A green boundary box will appear.
- Place your hand inside the box to see live predictions.
- **Press 'q' to quit.**

### 5. Streamlit GUI
To launch the interactive dashboard:
```bash
streamlit run src/app.py
```

## 📊 Evaluation
After training, the script produces:
1. **Accuracy/Loss Graphs**: Visualize training progress.
2. **Confusion Matrix**: See how well each gesture is recognized.
3. **Classification Report**: Precision, Recall, and F1-score for all 10 classes.

## 📝 Gestures Supported
- Palm
- L
- Fist
- Fist Moved
- Thumb
- Index
- OK
- Palm Moved
- C
- Down

---
Developed with ❤️ by Antigravity AI
