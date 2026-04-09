import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import numpy as np
import os
import sys

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import get_data_generators

def build_efficientnet(num_classes=101):
    """
    Builds an EfficientNetB0 based model.
    """
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model

def apply_tta(model, image_array, n_augmentations=5):
    """
    Applies Test Time Augmentation (TTA) to an image.
    Generates multiple augmented versions and averages predictions.
    """
    # Simple TTA: horizontal flip and small shifts
    preds = []
    
    # Original
    preds.append(model.predict(image_array, verbose=0))
    
    # Augmented versions
    for _ in range(n_augmentations - 1):
        # We can implement simple manual augmentation or use a generator
        # For simplicity, let's use a flip and small random noise
        flipped = np.flip(image_array, axis=2)
        preds.append(model.predict(flipped, verbose=0))
        
    avg_preds = np.mean(preds, axis=0)
    return avg_preds

def train_advanced(data_dir, epochs=20):
    """
    Training with EfficientNet and Learning Rate Scheduling.
    """
    train_gen, val_gen = get_data_generators(data_dir)
    num_classes = len(train_gen.class_indices)
    
    # Ensure models directory exists
    if not os.path.exists('models'):
        os.makedirs('models')
        
    model, base_model = build_efficientnet(num_classes)
    
    # Learning Rate Scheduler
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: 1e-3 * 0.9 ** epoch
    )
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("Training EfficientNetB0 (Base Frozen)...")
    model.fit(
        train_gen, 
        validation_data=val_gen, 
        epochs=epochs // 2, 
        callbacks=[lr_schedule]
    )
    
    # Fine-tuning
    base_model.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4), 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    print("Fine-tuning EfficientNetB0...")
    model.fit(
        train_gen, 
        validation_data=val_gen, 
        epochs=epochs // 2, 
        callbacks=[lr_schedule]
    )
    
    model.save("models/EfficientNetB0_best.keras")
    print("Model saved to models/EfficientNetB0_best.keras")

if __name__ == "__main__":
    # Example usage: python src/advanced_train.py --data_dir data/food-101/images
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/food-101/images")
    args = parser.parse_args()
    
    train_advanced(args.data_dir)
