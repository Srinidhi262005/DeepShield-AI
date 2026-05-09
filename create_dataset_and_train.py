"""
DeepShield - Dataset Creation & Model Training Pipeline (v2 - Real Faces)
=========================================================================
This script:
1. Downloads real human face photographs from the LFW (Labeled Faces in the Wild) dataset
2. 'Real' images: Actual human face photographs
3. 'Fake' images: Those same photos with deepfake-like artifacts applied
4. Trains the Xception-based binary classifier
5. Saves the trained model as model.h5

Usage: python create_dataset_and_train.py
"""

import os
import cv2
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
BATCH_SIZE = 16
EPOCHS = 10
NUM_IMAGES_PER_CLASS = 500  # number of real face images to use
DATASET_PATH = 'data/real_faces_dataset'
MODEL_PATH = 'model.h5'


# =====================================================
# STEP 1: Download & Prepare Real Face Dataset
# =====================================================

def download_real_faces(num_images):
    """
    Downloads real face photographs from the LFW (Labeled Faces in the Wild) dataset
    via scikit-learn. Returns a list of BGR images resized to IMG_SIZE.
    """
    from sklearn.datasets import fetch_lfw_people

    print("Downloading LFW (Labeled Faces in the Wild) dataset...")
    print("This may take a few minutes on first download (~200MB)...\n")

    # Download with color=True for RGB images
    lfw = fetch_lfw_people(min_faces_per_person=1, resize=1.0, color=True)
    all_faces = lfw.images  # shape: (N, 125, 94, 3), float64, 0-255 range

    print(f"Downloaded {len(all_faces)} face images from LFW")

    # Shuffle and select a subset
    num_to_select = min(num_images, len(all_faces))
    indices = np.random.permutation(len(all_faces))[:num_to_select]
    selected = all_faces[indices]

    faces = []
    for face_rgb in selected:
        # Convert to uint8
        face_uint8 = np.clip(face_rgb, 0, 255).astype(np.uint8)
        # Convert RGB to BGR for OpenCV compatibility
        face_bgr = cv2.cvtColor(face_uint8, cv2.COLOR_RGB2BGR)
        # Resize to target size with high-quality interpolation
        face_resized = cv2.resize(face_bgr, IMG_SIZE, interpolation=cv2.INTER_LANCZOS4)
        faces.append(face_resized)

    print(f"Prepared {len(faces)} face images at {IMG_SIZE} resolution")
    return faces


def apply_deepfake_artifacts(image):
    """
    Applies multiple subtle artifacts that simulate deepfake generation flaws:
    - Frequency domain noise injection
    - Localized blur (simulating face-swap boundary)
    - Color inconsistency
    - Geometric warping
    - Compression artifacts
    """
    img = image.copy().astype(np.float32)
    h, w = img.shape[:2]

    # 1. Frequency-domain noise (the hallmark of GAN-generated content)
    for c in range(3):
        f = np.fft.fft2(img[:, :, c])
        fshift = np.fft.fftshift(f)
        # Inject periodic noise in mid-frequencies
        rows, cols = fshift.shape
        crow, ccol = rows // 2, cols // 2
        noise_magnitude = np.random.uniform(15, 40)
        for _ in range(np.random.randint(3, 8)):
            r_offset = np.random.randint(-rows // 4, rows // 4)
            c_offset = np.random.randint(-cols // 4, cols // 4)
            fshift[crow + r_offset, ccol + c_offset] += noise_magnitude * np.random.randn()
        f_ishift = np.fft.ifftshift(fshift)
        img[:, :, c] = np.abs(np.fft.ifft2(f_ishift))

    # 2. Localized blur (face-swap boundary artifact)
    mask = np.zeros((h, w), dtype=np.float32)
    cx, cy = w // 2 + np.random.randint(-15, 16), h // 2 + np.random.randint(-15, 16)
    r1 = min(w, h) // 3
    r2 = min(w, h) // 2
    cv2.circle(mask, (cx, cy), r2, 1.0, -1)
    inner = np.zeros_like(mask)
    cv2.circle(inner, (cx, cy), r1, 1.0, -1)
    boundary = mask - inner
    boundary = cv2.GaussianBlur(boundary, (21, 21), 5)
    blurred = cv2.GaussianBlur(img, (7, 7), 2)
    for c in range(3):
        img[:, :, c] = img[:, :, c] * (1 - boundary) + blurred[:, :, c] * boundary

    # 3. Color inconsistency (slight hue/saturation shift in face region)
    hsv = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hue_shift = np.random.uniform(-8, 8)
    sat_shift = np.random.uniform(-20, 20)
    face_mask = inner
    hsv[:, :, 0] = hsv[:, :, 0] + hue_shift * face_mask
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + sat_shift * face_mask, 0, 255)
    img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    # 4. Subtle geometric warping (face alignment imperfection)
    warp_strength = np.random.uniform(2, 6)
    map_x = np.float32(np.tile(np.arange(w), (h, 1)))
    map_y = np.float32(np.tile(np.arange(h).reshape(-1, 1), (1, w)))
    map_x += warp_strength * np.sin(2 * np.pi * map_y / h * np.random.uniform(1, 3))
    map_y += warp_strength * np.sin(2 * np.pi * map_x / w * np.random.uniform(1, 3))
    img = cv2.remap(np.clip(img, 0, 255).astype(np.uint8), map_x, map_y, cv2.INTER_LINEAR).astype(np.float32)

    # 5. JPEG compression artifacts
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), np.random.randint(30, 70)]
    _, encoded = cv2.imencode('.jpg', np.clip(img, 0, 255).astype(np.uint8), encode_param)
    img = cv2.imdecode(encoded, cv2.IMREAD_COLOR).astype(np.float32)

    return np.clip(img, 0, 255).astype(np.uint8)


def create_dataset():
    """
    Creates the training dataset using real LFW face photographs.
    - Real class: actual face photos
    - Fake class: same photos with deepfake artifacts applied
    """
    real_dir = os.path.join(DATASET_PATH, 'real')
    fake_dir = os.path.join(DATASET_PATH, 'fake')
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    # Download real face images
    faces = download_real_faces(NUM_IMAGES_PER_CLASS)

    print(f"\nSaving {len(faces)} real images...")
    for i, face in enumerate(faces):
        # Apply minor natural augmentation to real images (keeps them real)
        augmented = face.copy()
        if np.random.random() > 0.5:
            augmented = cv2.flip(augmented, 1)
        if np.random.random() > 0.7:
            brightness = np.random.uniform(0.9, 1.1)
            augmented = np.clip(augmented * brightness, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(real_dir, f'real_{i:04d}.jpg'), augmented)
        if (i + 1) % 100 == 0:
            print(f"  Real: {i + 1}/{len(faces)}")

    print(f"\nGenerating {len(faces)} fake images (applying deepfake artifacts)...")
    for i, face in enumerate(faces):
        fake_face = apply_deepfake_artifacts(face)
        cv2.imwrite(os.path.join(fake_dir, f'fake_{i:04d}.jpg'), fake_face)
        if (i + 1) % 100 == 0:
            print(f"  Fake: {i + 1}/{len(faces)}")

    print(f"\nDataset created: {len(faces) * 2} total images in '{DATASET_PATH}'")
    print(f"  Real: {len(faces)} actual face photographs")
    print(f"  Fake: {len(faces)} photos with deepfake artifacts")
    return DATASET_PATH


# =====================================================
# STEP 2: Model Architecture (matches train.py)
# =====================================================

def build_model():
    """
    Builds the Xception-based deepfake detection model.
    Architecture matches what app.py and gradcam.py expect:
    - Xception backbone (ImageNet pretrained)
    - GlobalAveragePooling2D
    - Dense(512, relu) + Dropout(0.5)
    - Dense(1, sigmoid) for binary classification
    """
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
    base_model.trainable = False  # Freeze backbone initially

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# =====================================================
# STEP 3: Training Pipeline
# =====================================================

def train_model(dataset_path):
    """Trains the model on the real face dataset."""

    # Data generators with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        brightness_range=[0.9, 1.1],
        validation_split=0.2
    )

    print("\nLoading training data...")
    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training',
        shuffle=True
    )

    print("Loading validation data...")
    validation_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )

    print(f"\nClass mapping: {train_generator.class_indices}")
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {validation_generator.samples}")

    # Build model
    print("\nBuilding Xception model...")
    model = build_model()

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            MODEL_PATH,
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            patience=4,
            restore_best_weights=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        )
    ]

    # Phase 1: Train top layers only (base frozen)
    print("\n" + "=" * 50)
    print("Phase 1: Training classification head (base frozen)")
    print("=" * 50)
    model.fit(
        train_generator,
        epochs=EPOCHS // 2,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )

    # Phase 2: Fine-tune with unfrozen layers
    print("\n" + "=" * 50)
    print("Phase 2: Fine-tuning (unfreezing top layers)")
    print("=" * 50)

    # Unfreeze the last ~20 layers of Xception
    for layer in model.layers:
        layer.trainable = True
    # Re-freeze early layers to preserve low-level features
    for layer in model.layers[:100]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=0.00001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        train_generator,
        epochs=EPOCHS // 2,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )

    # Final save
    model.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    # Validate
    print("\nFinal evaluation:")
    results = model.evaluate(validation_generator, verbose=1)
    print(f"  Loss: {results[0]:.4f}")
    print(f"  Accuracy: {results[1]:.4f}")

    # Verify GradCAM compatibility
    try:
        target_layer = model.get_layer('block14_sepconv2_act')
        print(f"\n✓ GradCAM target layer '{target_layer.name}' found — heatmap generation will work.")
    except ValueError:
        print("\n⚠ GradCAM target layer not found — heatmaps may not generate.")

    return model


# =====================================================
# MAIN
# =====================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  DeepShield — Model Training Pipeline v2 (Real Faces)")
    print("=" * 60)

    # Step 1: Create dataset from real face photographs
    print("\n[Step 1/2] Downloading real faces & creating dataset...\n")
    dataset_path = create_dataset()

    # Step 2: Train the model
    print("\n[Step 2/2] Training the Xception model...\n")
    train_model(dataset_path)

    print("\n" + "=" * 60)
    print("  DONE! model.h5 is ready.")
    print("  Restart the Flask app to use the retrained model.")
    print("=" * 60)
