# Food Image Recognition and Calorie Estimation System

A deep learning project to classify food images and provide nutritional estimates.

## Project Overview
This project uses **Transfer Learning** with MobileNetV2 and ResNet50 to classify images from the **Food-101** dataset and estimates calorie content using a custom mapping.

## Folder Structure
- `data/`: Place the Food-101 dataset here (structure: `data/food-101/images/<class>/...`)
- `models/`: Stores the best-trained `.keras` models.
- `src/`: Core logic (Data loaders, Model building, Training, Evaluation).
- `app.py`: Streamlit web interface.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Dataset Preparation
Download the Food-101 dataset from Kaggle and place it in `data/food-101/`. Ensure the images are organized into subfolders by class.

### 3. Training the Model
Run the training script to build the model:
```bash
python src/train.py --data_dir data/food-101/images --model_type MobileNetV2 --epochs 20
```
This will save the best model to `models/MobileNetV2_best.keras`.

### 4. Running the App
Launch the Streamlit web app:
```bash
streamlit run app.py
```

## Features
- **Transfer Learning**: Uses pretrained deep networks for high accuracy.
- **Calorie Map**: Real-time nutritional estimation based on class results.
- **Modern UI**: Clean and intuitive Streamlit frontend.
- **Evaluation**: Tools for top-5 accuracy, confusion matrices, and reports.

## Accuracy Improvements Suggestions
1. **Unfreeze more layers**: During fine-tuning, unfreeze more layers of the base model.
2. **More Data Augmentation**: Add Gaussian noise or random brightness/contrast.
3. **Ensemble Modeling**: Average predictions from both MobileNetV2 and ResNet50.

---
Developed by Antigravity AI Assistant
