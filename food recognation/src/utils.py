import numpy as np
import tensorflow as tf
from PIL import Image

def preprocess_image(image, target_size=(224, 224)):
    """
    Prepares an image for model prediction.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    image = image.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale to [0,1]
    
    return img_array

def get_top_predictions(model, img_array, class_labels, k=5):
    """
    Returns the top K predictions for an image.
    """
    predictions = model.predict(img_array)
    top_indices = predictions[0].argsort()[-k:][::-1]
    
    results = []
    for i in top_indices:
        results.append({
            "label": class_labels[i],
            "confidence": predictions[0][i]
        })
    
    return results
