import os
import cv2
from concurrent.futures import ProcessPoolExecutor
from glob import glob

def resize_image(img_path, target_size=(224, 224)):
    """
    Resizes a single image and overwrites it.
    """
    try:
        img = cv2.imread(img_path)
        if img is None:
            return f"Error loading {img_path}"
        
        resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(img_path, resized)
        return None
    except Exception as e:
        return f"Failed {img_path}: {e}"

def preprocess_all_images(base_dir="data/processed"):
    """
    Batch resizes all images in the processed directory using multiple cores.
    """
    all_images = glob(os.path.join(base_dir, "**", "*.jpg"), recursive=True)
    total = len(all_images)
    print(f"Found {total} images to resize. Starting parallel processing...")
    
    # Using 4 workers to avoid crashing some systems, adjust as needed
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(resize_image, all_images))
    
    errors = [r for r in results if r is not None]
    if errors:
        print(f"Encountered {len(errors)} errors during processing.")
        # Optionally print a few errors
        for err in errors[:5]:
            print(err)
    
    print("Preprocessing (resizing) complete.")

if __name__ == "__main__":
    preprocess_all_images()
