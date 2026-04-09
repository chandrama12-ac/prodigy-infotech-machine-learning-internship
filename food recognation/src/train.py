import os
import sys
import argparse
import tensorflow as tf

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import get_data_generators
from src.model_builder import build_model, fine_tune_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def train(data_dir, model_type='MobileNetV2', epochs=10, batch_size=32):
    """
    Main training function.
    """
    # 1. Load Data
    train_gen, val_gen = get_data_generators(data_dir, batch_size=batch_size)
    num_classes = len(train_gen.class_indices)

    # Ensure models directory exists
    if not os.path.exists('models'):
        os.makedirs('models')

    # 2. Build Model
    model, base_model = build_model(model_type=model_type, num_classes=num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 3. Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint(f'models/{model_type}_best.keras', monitor='val_accuracy', save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

    # 4. Initial Training (Transfer Learning)
    print(f"Starting initial training for {model_type}...")
    history_base = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs // 2,
        callbacks=[early_stop, checkpoint, reduce_lr]
    )

    # 5. Fine-Tuning
    print(f"Starting fine-tuning for {model_type}...")
    model = fine_tune_model(model, base_model, fine_tune_at=100)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), # Lower LR for fine-tuning
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history_fine = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs // 2,
        callbacks=[early_stop, checkpoint, reduce_lr]
    )

    print(f"Training completed. Model saved to models/{model_type}_best.keras")
    return history_base, history_fine

if __name__ == "__main__":
    # Example usage
    # python train.py --data_dir data/food-101/images --model_type MobileNetV2 --epochs 20
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/food-101/images")
    parser.add_argument("--model_type", type=str, default="MobileNetV2")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    train(args.data_dir, args.model_type, args.epochs, args.batch_size)
