import os
from kaggle.api.kaggle_api_extended import KaggleApi

def download_dataset(dataset_id="dansbecker/food-101", download_path="data/raw"):
    """
    Downloads and unzips the Food-101 dataset from Kaggle.
    Requires kaggle.json in ~/.kaggle/
    """
    api = KaggleApi()
    api.authenticate()
    
    if not os.path.exists(download_path):
        os.makedirs(download_path)
        
    print(f"Downloading {dataset_id}...")
    api.dataset_download_files(dataset_id, path=download_path, unzip=True)
    print(f"Download and extraction completed in {download_path}")

if __name__ == "__main__":
    download_dataset()
