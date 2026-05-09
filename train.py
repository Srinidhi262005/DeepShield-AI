import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# --- Configuration ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
DATASET_PATH = 'data/faceforensics_processed' # Expected: data/train/real, data/train/fake

def build_model():
    """
    Builds the Xception-based deepfake detection model.
    Using transfer learning with fine-tuning.
    """
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
    
    # Freeze the base model initially
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x) # Binary classification: Real (0) vs Fake (1)

    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer=Adam(learning_rate=0.0001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

def train():
    """
    Main training pipeline.
    """
    # 1. Setup Data Generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )

    # 2. Build Model
    model = build_model()
    print(model.summary())

    # 3. Callbacks
    callbacks = [
        ModelCheckpoint('model.h5', save_best_only=True, monitor='val_loss', mode='min'),
        EarlyStopping(patience=5, restore_best_weights=True)
    ]

    # 4. Training (Stage 1: Top Layers)
    print("Starting training: Phase 1 (Top Layers Only)")
    model.fit(
        train_generator,
        epochs=EPOCHS // 2,
        validation_data=validation_generator,
        callbacks=callbacks
    )

    # 5. Fine-tuning (Stage 2: Unfreeze some layers)
    print("Starting training: Phase 2 (Fine-tuning)")
    model.trainable = True
    # Compile with a lower learning rate for fine-tuning
    model.compile(optimizer=Adam(learning_rate=0.00001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    model.fit(
        train_generator,
        epochs=EPOCHS // 2,
        validation_data=validation_generator,
        callbacks=callbacks
    )

    print("Training complete. Model saved as model.h5")

if __name__ == "__main__":
    # Check for dataset existence before starting
    if os.path.exists(DATASET_PATH):
        train()
    else:
        print(f"Error: Dataset not found at {DATASET_PATH}.")
        print("Please ensure your dataset is organized as: data/faceforensics_processed/{real, fake}")
        # Build model structure anyway to demonstrate
        model = build_model()
        print("Model architecture built successfully.")
