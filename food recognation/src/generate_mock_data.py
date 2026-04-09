import os
import numpy as np
import cv2

def generate_mock_data(base_dir="data/subset", num_classes=5, images_per_class=10):
    """
    Generates a mock dataset with small random images.
    Allows testing the training scripts without downloading 5GB of Food-101.
    """
    splits = ["train", "validation"]
    classes = [f"food_class_{i}" for i in range(num_classes)]
    
    print(f"Generating mock dataset in {base_dir}...")
    
    for split in splits:
        for cls in classes:
            path = os.path.join(base_dir, split, cls)
            os.makedirs(path, exist_ok=True)
            
            for i in range(images_per_class):
                # Create a random RGB image
                img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                img_path = os.path.join(path, f"img_{i}.jpg")
                cv2.imwrite(img_path, img)
                
    print(f"✅ Mock dataset created. Total images: {num_classes * images_per_class * 2}")

if __name__ == "__main__":
    generate_mock_data()
