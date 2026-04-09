import os
import shutil
from sklearn.model_selection import train_test_split

def organize_dataset(raw_path="data/raw/food-101", output_path="data/processed"):
    """
    Organizes the Food-101 images into train, val, and test folders based on meta files.
    """
    images_dir = os.path.join(raw_path, "images")
    meta_dir = os.path.join(raw_path, "meta")
    
    # Read meta files
    with open(os.path.join(meta_dir, "train.txt"), "r") as f:
        train_list = [line.strip() for line in f.readlines()]
    with open(os.path.join(meta_dir, "test.txt"), "r") as f:
        test_list = [line.strip() for line in f.readlines()]
        
    # Split training into train and validation (80/20)
    train_split, val_split = train_test_split(train_list, test_size=0.2, random_state=42)
    
    splits = {
        "train": train_split,
        "validation": val_split,
        "test": test_list
    }
    
    for split_name, split_files in splits.items():
        print(f"Organizing {split_name} set...")
        for file_rel_path in split_files:
            # Source path (path/to/image) -> add .jpg
            src = os.path.join(images_dir, file_rel_path + ".jpg")
            
            # Destination path: data/processed/{split}/{category}/{image}.jpg
            category = os.path.dirname(file_rel_path)
            dest_dir = os.path.join(output_path, split_name, category)
            
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
                
            dest = os.path.join(dest_dir, os.path.basename(file_rel_path) + ".jpg")
            
            # Use shutil.copy to keep original raw data
            shutil.copy(src, dest)
            
    print("Project data organization complete.")

if __name__ == "__main__":
    organize_dataset()
