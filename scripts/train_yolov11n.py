"""
Script train YOLOv11n cho Coffee Bean Ripeness Detection
Dataset: Lam Dong Coffee Bean Ripeness Dataset
Classes: barely-riped, over-riped, riped, semi-riped, unriped

YOLOv11n is the latest YOLO version with improved architecture:
- C3K2 blocks (improved from C2f in YOLOv8)
- Better small object detection
- Fewer parameters than YOLOv8n
"""

import os
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np


# ==================== CONFIG ====================
# Paths - Thay đổi cho phù hợp với server
BASE_DIR = Path("/datastore/vit/dpduy123/CoffeeProject")
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"
DATA_YAML = DATASET_DIR / "data.yaml"

# Training params
IMG_SIZE = 640
BATCH_SIZE = 16
EPOCHS = 100
MODEL_NAME = "yolo11n.pt"  # YOLOv11 nano model


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
    train_images = list(train_img_dir.glob("*.jpg")) + list(train_img_dir.glob("*.png")) + list(train_img_dir.glob("*.jpeg"))
    val_images = list(val_img_dir.glob("*.jpg")) + list(val_img_dir.glob("*.png")) + list(val_img_dir.glob("*.jpeg"))
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
        # Create data.yaml if not exists
        create_data_yaml()
        return True

    return True


def create_data_yaml():
    """Create data.yaml file for dataset"""
    print("\nCreating data.yaml...")

    data_yaml_content = f"""# Coffee Bean Ripeness Dataset
path: {DATASET_DIR}
train: images/train
val: images/val

# Classes
names:
  0: barely-riped
  1: over-riped
  2: riped
  3: semi-riped
  4: unriped

nc: 5
"""

    with open(DATA_YAML, 'w') as f:
        f.write(data_yaml_content)

    print(f"✓ Created {DATA_YAML}")


# ==================== CHECK IMAGE AFTER RESIZE ====================
def check_image_resize(num_samples=5):
    """Kiểm tra ảnh sau khi resize về 640x640"""
    print("\n" + "=" * 50)
    print(f"CHECKING IMAGE RESIZE TO {IMG_SIZE}x{IMG_SIZE}")
    print("=" * 50)

    train_img_dir = DATASET_DIR / "images" / "train"
    images = list(train_img_dir.glob("*.jpg")) + list(train_img_dir.glob("*.png")) + list(train_img_dir.glob("*.jpeg"))

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
    """Train YOLOv11n model"""
    print("\n" + "=" * 50)
    print("TRAINING YOLOv11n")
    print("=" * 50)

    # Create model directory
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load pretrained model
    print(f"\nLoading pretrained model: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)

    # Print model info
    print(f"\nModel: {MODEL_NAME}")
    print(f"Parameters: {sum(p.numel() for p in model.model.parameters()):,}")

    # Train
    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=str(MODEL_DIR),
        name="yolov11n_coffee",
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
        # Learning rate
        lr0=0.01,
        lrf=0.01,
        # Warmup
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        # Save settings
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        # Early stopping
        patience=50,
        # Other
        cos_lr=True,  # Cosine learning rate scheduler
        close_mosaic=10,  # Disable mosaic augmentation for last 10 epochs
    )

    print("\n" + "=" * 50)
    print("TRAINING COMPLETED!")
    print("=" * 50)
    print(f"Best model saved at: {MODEL_DIR}/yolov11n_coffee/weights/best.pt")
    print(f"Last model saved at: {MODEL_DIR}/yolov11n_coffee/weights/last.pt")
    print(f"Results saved at: {MODEL_DIR}/yolov11n_coffee/")

    return results


# ==================== VALIDATE ====================
def validate():
    """Validate trained model"""
    print("\n" + "=" * 50)
    print("VALIDATING MODEL")
    print("=" * 50)

    best_model_path = MODEL_DIR / "yolov11n_coffee" / "weights" / "best.pt"

    if not best_model_path.exists():
        print(f"Model not found at {best_model_path}")
        return

    model = YOLO(str(best_model_path))

    print(f"\nModel: {best_model_path}")
    print(f"Dataset: {DATA_YAML}")

    results = model.val(
        data=str(DATA_YAML),
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        conf=0.25,
        iou=0.45,
        device=0,
        verbose=True,
    )

    # Print results
    print("\n" + "=" * 50)
    print("VALIDATION RESULTS")
    print("=" * 50)
    print(f"mAP@0.5: {results.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {results.box.map:.4f}")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall: {results.box.mr:.4f}")

    return results


# ==================== EXPORT ====================
def export_model(format="onnx"):
    """Export trained model to different formats"""
    print("\n" + "=" * 50)
    print(f"EXPORTING MODEL TO {format.upper()}")
    print("=" * 50)

    best_model_path = MODEL_DIR / "yolov11n_coffee" / "weights" / "best.pt"

    if not best_model_path.exists():
        print(f"Model not found at {best_model_path}")
        return

    model = YOLO(str(best_model_path))

    # Export
    model.export(format=format, imgsz=IMG_SIZE)

    print(f"\n✓ Model exported to {format} format")


# ==================== COMPARE WITH YOLOv8n ====================
def compare_models():
    """Compare YOLOv11n vs YOLOv8n architecture"""
    print("\n" + "=" * 50)
    print("YOLOv11n vs YOLOv8n COMPARISON")
    print("=" * 50)

    print("""
    | Feature          | YOLOv8n    | YOLOv11n   |
    |------------------|------------|------------|
    | Parameters       | 3.2M       | 2.6M       |
    | FLOPs            | 8.7G       | 6.5G       |
    | mAP@0.5 (COCO)   | 37.3%      | 39.5%      |
    | Block Type       | C2f        | C3K2       |
    | Speed            | Fast       | Faster     |

    YOLOv11n improvements:
    - C3K2 blocks: Better feature extraction
    - Fewer parameters: Less overfitting on small datasets
    - Better small object detection
    """)


# ==================== MAIN ====================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train YOLOv11n for Coffee Bean Detection")
    parser.add_argument("--check", action="store_true", help="Check dataset only")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--validate", action="store_true", help="Validate model")
    parser.add_argument("--export", type=str, default=None, help="Export model (onnx, torchscript, etc.)")
    parser.add_argument("--compare", action="store_true", help="Compare YOLOv11n vs YOLOv8n")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--img-size", type=int, default=IMG_SIZE, help="Image size")
    parser.add_argument("--device", type=str, default="0", help="Device (0, 1, cpu)")

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
    elif args.export:
        export_model(args.export)
    elif args.compare:
        compare_models()
    else:
        # Default: check then train
        if check_dataset():
            check_image_resize()
            train()
