import os
import sys
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.scratch_model import build_scratch_model
from src.data_loader import get_data_generators

def train_scratch(data_dir, epochs=20, num_classes=5):
    """
    Trains the custom CNN model from scratch without any pre-trained weights.
    """
    # 1. Load Data (assuming subset if num_classes is low)
    train_gen, val_gen = get_data_generators(data_dir, target_size=(224, 224))
    
    # Update num_classes based on generator if necessary
    actual_classes = len(train_gen.class_indices)
    
    # 2. Build Scratch Model
    model = build_scratch_model(num_classes=actual_classes)
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001), # Slower learning for scratch models
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 3. Training
    print(f"Starting 'No-AI' Training from scratch on {actual_classes} classes...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs
    )
    
    # 4. Save
    os.makedirs("models", exist_ok=True)
    model.save("models/ScratchModel_best.keras")
    print("✅ Model trained from scratch and saved to models/ScratchModel_best.keras")

if __name__ == "__main__":
    # Example: python src/train_scratch.py --data_dir data/subset --epochs 30
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/subset")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    
    train_scratch(args.data_dir, args.epochs)
