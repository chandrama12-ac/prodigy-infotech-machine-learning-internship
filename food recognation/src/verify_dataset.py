import os
import cv2
from glob import glob

def verify_dataset(processed_dir="data/processed"):
    """
    Checks if the dataset structure is correct and images have the right dimensions.
    """
    splits = ["train", "validation", "test"]
    expected_classes = 101
    
    print("--- Verification Report ---")
    
    for split in splits:
        split_path = os.path.join(processed_dir, split)
        if not os.path.exists(split_path):
            print(f"❌ {split} directory missing!")
            continue
            
        classes = os.listdir(split_path)
        print(f"✅ {split} set: {len(classes)} classes found.")
        
        # Check a sample image from the first class
        if len(classes) > 0:
            class_folder = os.path.join(split_path, classes[0])
            images = glob(os.path.join(class_folder, "*.jpg"))
            if images:
                img = cv2.imread(images[0])
                h, w, _ = img.shape
                print(f"   Sample Image: {os.path.basename(images[0])} | Size: {w}x{h}")
                if w == 224 and h == 224:
                    print("   📏 Dimensions: Correct")
                else:
                    print("   📏 Dimensions: WRONG!")
            else:
                print(f"   ⚠️ No images found in {class_folder}")

if __name__ == "__main__":
    verify_dataset()
