import os
import shutil
import random

def create_subset(source_dir, dest_dir, num_classes=5, num_images=100):
    """
    Creates a smaller subset of the dataset for fast 'from scratch' training testing.
    """
    # Ensure destination exists
    os.makedirs(dest_dir, exist_ok=True)
    
    if not os.path.exists(source_dir):
        print(f"⚠️ Source directory {source_dir} not found. Creating empty class folders for structure...")
        # If source is missing, we create empty structure to avoid crashes in downstream scripts
        for i in range(num_classes):
            os.makedirs(os.path.join(dest_dir, f"food_class_{i}"), exist_ok=True)
        return

    # Get all categories
    categories = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    if not categories:
        print(f"⚠️ No category subfolders found in {source_dir}.")
        return
    
    # Select random categories
    selected_categories = random.sample(categories, min(num_classes, len(categories)))
    
    print(f"Selected categories: {selected_categories}")
    
    for category in selected_categories:
        src_cat_dir = os.path.join(source_dir, category)
        dest_cat_dir = os.path.join(dest_dir, category)
        
        os.makedirs(dest_cat_dir, exist_ok=True)
        
        images = os.listdir(src_cat_dir)
        selected_images = random.sample(images, min(num_images, len(images)))
        
        for img in selected_images:
            shutil.copy(os.path.join(src_cat_dir, img), os.path.join(dest_cat_dir, img))
            
    print(f"✅ Subset created at {dest_dir}")

if __name__ == "__main__":
    # Example: Create a small subset for quick training
    create_subset("data/processed/train", "data/subset/train", num_classes=5, num_images=50)
    create_subset("data/processed/validation", "data/subset/validation", num_classes=5, num_images=15)
