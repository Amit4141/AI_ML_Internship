"""
Download a sample face mask dataset
Run this if you don't want to manually download from Kaggle
"""
import os
import urllib.request
import zipfile

print("This script helps you download a face mask dataset.")
print("\nOption 1: Download from Kaggle manually")
print("  - Visit: https://www.kaggle.com/datasets/omkargurav/face-mask-dataset")
print("  - Download and extract to 'dataset/' folder")
print("\nOption 2: Use alternative dataset")
print("  - Visit: https://github.com/chandrikadeb7/Face-Mask-Detection")
print("  - Clone or download the dataset folder")
print("\nAfter downloading, your folder structure should be:")
print("dataset/")
print("  ├── with_mask/")
print("  └── without_mask/")

# Create dataset folder if it doesn't exist
os.makedirs('dataset/with_mask', exist_ok=True)
os.makedirs('dataset/without_mask', exist_ok=True)

print("\n✅ Dataset folders created!")
print("Now place your images in the respective folders.")
