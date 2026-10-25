# BACKEND CODE - train_model.py

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Dataset paths
TRAIN_DIR = "dataset/train"
VALIDATION_DIR = "dataset/validation"

# Model folder
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# Image settings
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# Data preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

# Load training images
train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

# Load validation images
validation_data = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

print("\nClass mapping:")
print(train_data.class_indices)

# Build CNN model
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(1, activation="sigmoid")
])

# Compile
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Display model information
model.summary()

# Train
history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=10
)

# Save trained model
model_path = os.path.join(MODEL_DIR, "deepfake_model.keras")
model.save(model_path)

print("\n====================================")
print("MODEL TRAINING COMPLETED")
print("Model saved at:", model_path)
print("====================================")
