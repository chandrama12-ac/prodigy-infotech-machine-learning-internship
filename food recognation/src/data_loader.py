import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(data_dir, target_size=(224, 224), batch_size=32):
    """
    Creates train and validation data generators with augmentation.
    Assumes data_dir contains subfolders for each class.
    """
    
    # Data Augmentation & Normalization
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2  # 20% for validation
    )

    # Validation/Test Generator (Only Rescaling)
    valid_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    # Training Generator
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    # Validation Generator
    validation_generator = valid_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    return train_generator, validation_generator

def get_class_labels(generator):
    """Returns a list of class labels from the generator."""
    return list(generator.class_indices.keys())
