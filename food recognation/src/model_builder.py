import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

def build_model(model_type='MobileNetV2', num_classes=101, input_shape=(224, 224, 3)):
    """
    Builds a transfer learning model with a custom classification head.
    """
    if model_type == 'MobileNetV2':
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_type == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError("Unsupported model type. Choose 'MobileNetV2' or 'ResNet50'.")

    # Freeze the base layers initially
    base_model.trainable = False

    # Create custom head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    return model, base_model

def fine_tune_model(model, base_model, fine_tune_at=100):
    """
    Unfreezes the base model layers from a certain index for fine-tuning.
    """
    base_model.trainable = True
    
    # Freeze all layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
        
    return model
