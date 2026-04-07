import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from preprocess import get_train_val_test_splits
from model import build_cnn_model

# Constants
EPOCHS = 20
BATCH_SIZE = 32
MODEL_PATH = 'models/hand_gesture_model.h5'
PLOT_PATH = 'models/training_curves.png'
CM_PATH = 'models/confusion_matrix.png'

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Gesture Labels mapping (extracted from folder structure)
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

def train():
    # 1. Load data
    X_train, X_val, X_test, y_train, y_val, y_test = get_train_val_test_splits()
    
    # 2. Build model
    model = build_cnn_model(input_shape=(64, 64, 1), num_classes=10)
    
    # 3. Data Augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(X_train)
    
    # 4. Callbacks
    early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True)
    
    # 5. Train
    print("\nStarting Training...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=[early_stop, checkpoint],
        verbose=1
    )
    
    # 6. Evaluate on Test set
    print("\nEvaluating on Test Set...")
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    
    # 7. Visualization & Metrics
    plot_results(history)
    generate_metrics(model, X_test, y_test)
    
    # 8. Save Final Model
    model.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

def plot_results(history):
    """
    Plots training vs validation accuracy and loss.
    """
    plt.figure(figsize=(12, 5))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Val Accuracy', color='orange', linewidth=2)
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', color='blue', linewidth=2)
    plt.plot(history.history['val_loss'], label='Val Loss', color='orange', linewidth=2)
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    print(f"Training curves saved to {PLOT_PATH}")
    # plt.show() removed to prevent blocking

def generate_metrics(model, X_test, y_test):
    """
    Generate Confusion Matrix and Classification Report.
    """
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Classification Report
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred_classes, target_names=list(GESTURES.values()))
    print(report)
    
    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=list(GESTURES.values()), 
                yticklabels=list(GESTURES.values()))
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig(CM_PATH)
    print(f"Confusion matrix saved to {CM_PATH}")
    # plt.show() removed to prevent blocking

if __name__ == "__main__":
    train()
