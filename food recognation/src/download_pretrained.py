import urllib.request
import os

def download_pretrained_model(url, save_path):
    """
    Downloads a pre-trained model from a URL.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print(f"Downloading pre-trained model from Hugging Face...")
    print(f"URL: {url}")
    
    try:
        urllib.request.urlretrieve(url, save_path)
        print(f"✅ Download complete! Model saved to {save_path}")
    except Exception as e:
        print(f"❌ Failed to download model: {e}")

if __name__ == "__main__":
    # URL for a community-trained Food-101 Keras model
    # Source: https://huggingface.co/ml-debi/EfficientNetB0-Food101
    HF_URL = "https://huggingface.co/ml-debi/EfficientNetB0-Food101/resolve/main/tf_model.h5"
    SAVE_PATH = "models/ResNet50_best.keras"  # Map it to one of our app choices
    
    download_pretrained_model(HF_URL, SAVE_PATH)
