import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import os

def generate_mock_model(save_path="models/MobileNetV2_best.keras"):
    """
    Generates a dummy (randomly initialized) model with the correct Food-101 architecture.
    Useful for testing the UI and pipeline without training on 5GB of data.
    """
    print("Generating Mock Model (MobileNetV2 architecture)...")
    
    # Ensure models directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Standard Food-101 architecture
    base_model = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    predictions = Dense(101, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Save the model
    model.save(save_path)
    print(f"✅ Mock Model saved to {save_path}")
    print("Note: Predictions will be random until the model is trained with real data.")

if __name__ == "__main__":
    generate_mock_model()
