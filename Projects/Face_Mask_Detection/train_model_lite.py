"""
Lightweight training script using MobileNetV2
Easier to install and faster to train
"""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

try:
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
except ImportError:
    print("TensorFlow not installed properly.")
    print("Try: pip install tensorflow-cpu --user")
    exit(1)

# Configuration
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10
DATA_DIR = 'dataset'

def create_model():
    """Create model using MobileNetV2 transfer learning"""
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(2, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train():
    """Train the mask detection model"""
    if not os.path.exists(DATA_DIR):
        print(f"Error: {DATA_DIR} folder not found!")
        print("Please download dataset and place in dataset/ folder")
        return
    
    # Data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    # Training data
    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )
    
    # Validation data
    val_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )
    
    # Create and train model
    print("Creating model...")
    model = create_model()
    print(model.summary())
    
    print("Training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator
    )
    
    # Save model
    model.save('mask_detector.h5')
    print("✅ Model saved as mask_detector.h5")
    
    return history

if __name__ == '__main__':
    train()
