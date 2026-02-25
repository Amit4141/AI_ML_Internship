"""
Run this in Google Colab to train the model
Then download mask_detector.h5 and use it locally
"""

# Run these commands in Colab:
# !pip install opencv-python

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Upload your dataset to Colab or mount Google Drive
# For this example, we'll use a sample dataset

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10

def create_model():
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dense(2, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Instructions for Colab:
print("1. Upload this file to Google Colab")
print("2. Upload your dataset folder")
print("3. Run the training")
print("4. Download mask_detector.h5")
print("5. Use it with detect_mask.py on your local machine")
