"""
Script train YOLOv8n cho Coffee Bean Ripeness Detection
Dataset: Lam Dong Coffee Bean Ripeness Dataset
Classes: barely-riped, over-riped, riped, semi-riped, unriped
"""

import os
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np


# ==================== CONFIG ====================
# Paths - Thay đổi cho phù hợp với server
BASE_DIR = Path("/datastore/vit/vit/dpduy123/CoffeeProject")
DATASET_DIR = BASE_DIR / "dataset" / "coffee_dataset"
MODEL_DIR = BASE_DIR / "models"
DATA_YAML = DATASET_DIR / "data.yaml"

# Training params
IMG_SIZE = 640
BATCH_SIZE = 16
EPOCHS = 100
MODEL_NAME = "yolov8n.pt"  # nano model


# ==================== CHECK DATASET ====================
def check_dataset():
    """Kiểm tra dataset và hiển thị thông tin"""
    print("=" * 50)
    print("CHECKING DATASET")
    print("=" * 50)

    # Check paths exist
    train_img_dir = DATASET_DIR / "images" / "train"
    val_img_dir = DATASET_DIR / "images" / "val"
    train_label_dir = DATASET_DIR / "labels" / "train"
    val_label_dir = DATASET_DIR / "labels" / "val"

    for path in [DATASET_DIR, train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
        if path.exists():
            print(f"✓ {path}")
        else:
            print(f"✗ {path} NOT FOUND!")
            return False

    # Count images and labels
    train_images = list(train_img_dir.glob("*.jpg")) + list(train_img_dir.glob("*.png"))
    val_images = list(val_img_dir.glob("*.jpg")) + list(val_img_dir.glob("*.png"))
    train_labels = list(train_label_dir.glob("*.txt"))
    val_labels = list(val_label_dir.glob("*.txt"))

    print(f"\nTrain images: {len(train_images)}")
    print(f"Train labels: {len(train_labels)}")
    print(f"Val images: {len(val_images)}")
    print(f"Val labels: {len(val_labels)}")

    # Check data.yaml
    if DATA_YAML.exists():
        print(f"\n✓ data.yaml found")
        with open(DATA_YAML, 'r') as f:
            print(f.read())
    else:
        print(f"\n✗ data.yaml NOT FOUND!")
        return False

    return True


# ==================== CHECK IMAGE AFTER RESIZE ====================
def check_image_resize(num_samples=5):
    """Kiểm tra ảnh sau khi resize về 640x640"""
    print("\n" + "=" * 50)
    print(f"CHECKING IMAGE RESIZE TO {IMG_SIZE}x{IMG_SIZE}")
    print("=" * 50)

    train_img_dir = DATASET_DIR / "images" / "train"
    images = list(train_img_dir.glob("*.jpg")) + list(train_img_dir.glob("*.png"))

    if not images:
        print("No images found!")
        return

    # Sample random images
    sample_images = np.random.choice(images, min(num_samples, len(images)), replace=False)

    for img_path in sample_images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"✗ Cannot read: {img_path.name}")
            continue

        original_shape = img.shape

        # Resize to 640x640
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        resized_shape = img_resized.shape

        print(f"✓ {img_path.name}: {original_shape} -> {resized_shape}")

    print(f"\nAll images will be resized to {IMG_SIZE}x{IMG_SIZE}x3 during training")


# ==================== TRAIN ====================
def train():
    """Train YOLOv8n model"""
    print("\n" + "=" * 50)
    print("TRAINING YOLOv8n")
    print("=" * 50)

    # Create model directory
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load pretrained model
    model = YOLO(MODEL_NAME)

    # Train
    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=str(MODEL_DIR),
        name="yolov8n_coffee",
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        verbose=True,
        seed=42,
        deterministic=True,
        # GPU settings
        device=0,  # Use first GPU, change to "cpu" if no GPU
        workers=8,
        # Augmentation
        augment=True,
        # Save settings
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
    )

    print("\n" + "=" * 50)
    print("TRAINING COMPLETED!")
    print("=" * 50)
    print(f"Best model saved at: {MODEL_DIR}/yolov8n_coffee/weights/best.pt")
    print(f"Last model saved at: {MODEL_DIR}/yolov8n_coffee/weights/last.pt")

    return results


# ==================== VALIDATE ====================
def validate():
    """Validate trained model"""
    print("\n" + "=" * 50)
    print("VALIDATING MODEL")
    print("=" * 50)

    best_model_path = MODEL_DIR / "yolov8n_coffee" / "weights" / "best.pt"

    if not best_model_path.exists():
        print(f"Model not found at {best_model_path}")
        return

    model = YOLO(str(best_model_path))
    results = model.val(data=str(DATA_YAML))

    return results


# ==================== MAIN ====================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train YOLOv8n for Coffee Bean Detection")
    parser.add_argument("--check", action="store_true", help="Check dataset only")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--validate", action="store_true", help="Validate model")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--img-size", type=int, default=IMG_SIZE, help="Image size")

    args = parser.parse_args()

    # Update config from args
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch
    IMG_SIZE = args.img_size

    if args.check:
        check_dataset()
        check_image_resize()
    elif args.train:
        if check_dataset():
            check_image_resize()
            train()
    elif args.validate:
        validate()
    else:
        # Default: check then train
        if check_dataset():
            check_image_resize()
            train()
