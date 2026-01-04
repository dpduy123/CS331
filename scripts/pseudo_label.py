"""
Script Pseudo-labeling cho Coffee Bean Dataset
Sử dụng model YOLOv8s đã train để tự động label ảnh mới

Workflow:
1. Load model đã train
2. Predict trên ảnh chưa label
3. Lưu labels với confidence scores
4. Review và fix bằng tool (CVAT, Label Studio, Roboflow)
"""

import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import cv2


# ==================== CONFIG ====================
# Paths - Thay đổi cho phù hợp
BASE_DIR = Path("/datastore/vit/vit/dpduy123/CoffeeProject")

# Model đã train
MODEL_PATH = BASE_DIR / "models" / "yolov8s_coffee" / "weights" / "best.pt"

# Folder chứa ảnh chưa label
UNLABELED_DIR = BASE_DIR / "unlabeled_images"  # Upload "Hình cà phê" vào đây

# Output folder
OUTPUT_DIR = BASE_DIR / "pseudo_labeled"
OUTPUT_IMAGES_DIR = OUTPUT_DIR / "images"
OUTPUT_LABELS_DIR = OUTPUT_DIR / "labels"

# Confidence threshold
CONF_THRESHOLD = 0.25  # Chỉ giữ predictions có confidence >= 25%

# Classes
CLASSES = ["barely-riped", "over-riped", "riped", "semi-riped", "unriped"]


# ==================== PSEUDO LABEL ====================
def pseudo_label():
    """Tự động label ảnh bằng model đã train"""

    print("=" * 50)
    print("PSEUDO LABELING")
    print("=" * 50)

    # Check model exists
    if not MODEL_PATH.exists():
        print(f"✗ Model not found: {MODEL_PATH}")
        print("Hãy train model trước bằng train_yolov8s.py")
        return

    # Check unlabeled folder exists
    if not UNLABELED_DIR.exists():
        print(f"✗ Unlabeled folder not found: {UNLABELED_DIR}")
        print(f"Hãy upload ảnh vào: {UNLABELED_DIR}")
        return

    # Create output directories
    OUTPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_LABELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\nLoading model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    # Get all images
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    images = []
    for ext in image_extensions:
        images.extend(UNLABELED_DIR.glob(f"*{ext}"))
        images.extend(UNLABELED_DIR.glob(f"*{ext.upper()}"))

    print(f"Found {len(images)} images\n")

    if not images:
        print("No images found!")
        return

    # Process each image
    total_boxes = 0
    images_with_detections = 0

    for i, img_path in enumerate(images):
        # Predict
        results = model.predict(
            source=str(img_path),
            conf=CONF_THRESHOLD,
            verbose=False
        )

        result = results[0]
        boxes = result.boxes

        if len(boxes) > 0:
            images_with_detections += 1
            total_boxes += len(boxes)

            # Copy image to output
            # Đổi tên để tránh ký tự đặc biệt
            new_img_name = f"pseudo_{i:04d}.jpg"
            new_img_path = OUTPUT_IMAGES_DIR / new_img_name

            img = cv2.imread(str(img_path))
            cv2.imwrite(str(new_img_path), img)

            # Save labels in YOLO format
            label_name = f"pseudo_{i:04d}.txt"
            label_path = OUTPUT_LABELS_DIR / label_name

            img_h, img_w = img.shape[:2]

            with open(label_path, "w") as f:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])

                    # Get normalized coordinates (YOLO format)
                    x_center, y_center, width, height = box.xywhn[0].tolist()

                    # Write: class x_center y_center width height
                    f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            print(f"✓ {img_path.name} -> {new_img_name} ({len(boxes)} boxes)")
        else:
            print(f"✗ {img_path.name} (no detections)")

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Total images processed: {len(images)}")
    print(f"Images with detections: {images_with_detections}")
    print(f"Total bounding boxes: {total_boxes}")
    print(f"\nOutput saved to: {OUTPUT_DIR}")
    print(f"  - Images: {OUTPUT_IMAGES_DIR}")
    print(f"  - Labels: {OUTPUT_LABELS_DIR}")
    print("\nNext steps:")
    print("1. Review labels using CVAT/Label Studio/Roboflow")
    print("2. Fix incorrect predictions")
    print("3. Merge with original dataset")
    print("4. Train again with more data")


# ==================== MERGE DATASETS ====================
def merge_datasets():
    """Gộp pseudo-labeled data với original dataset"""

    print("=" * 50)
    print("MERGING DATASETS")
    print("=" * 50)

    # Original dataset
    ORIGINAL_DIR = BASE_DIR / "dataset" / "coffee_dataset"
    MERGED_DIR = BASE_DIR / "dataset" / "coffee_dataset_merged"

    # Create merged directories
    (MERGED_DIR / "images" / "train").mkdir(parents=True, exist_ok=True)
    (MERGED_DIR / "images" / "val").mkdir(parents=True, exist_ok=True)
    (MERGED_DIR / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (MERGED_DIR / "labels" / "val").mkdir(parents=True, exist_ok=True)

    # Copy original data
    print("\nCopying original data...")
    for split in ["train", "val"]:
        # Images
        src_img = ORIGINAL_DIR / "images" / split
        dst_img = MERGED_DIR / "images" / split
        for img in src_img.glob("*"):
            shutil.copy(img, dst_img / img.name)

        # Labels
        src_lbl = ORIGINAL_DIR / "labels" / split
        dst_lbl = MERGED_DIR / "labels" / split
        for lbl in src_lbl.glob("*"):
            shutil.copy(lbl, dst_lbl / lbl.name)

    # Copy pseudo-labeled data (to train set)
    print("Copying pseudo-labeled data...")
    for img in OUTPUT_IMAGES_DIR.glob("*"):
        shutil.copy(img, MERGED_DIR / "images" / "train" / img.name)

    for lbl in OUTPUT_LABELS_DIR.glob("*"):
        shutil.copy(lbl, MERGED_DIR / "labels" / "train" / lbl.name)

    # Create data.yaml
    data_yaml_content = f"""path: {MERGED_DIR}
train: images/train
val: images/val
nc: 5
names:
- barely-riped
- over-riped
- riped
- semi-riped
- unriped
"""

    with open(MERGED_DIR / "data.yaml", "w") as f:
        f.write(data_yaml_content)

    # Count
    train_images = len(list((MERGED_DIR / "images" / "train").glob("*")))
    val_images = len(list((MERGED_DIR / "images" / "val").glob("*")))

    print("\n" + "=" * 50)
    print("MERGE COMPLETED")
    print("=" * 50)
    print(f"Merged dataset: {MERGED_DIR}")
    print(f"Train images: {train_images}")
    print(f"Val images: {val_images}")
    print(f"\nTo train with merged data:")
    print(f"  Update DATA_YAML in train script to: {MERGED_DIR}/data.yaml")


# ==================== VISUALIZE ====================
def visualize(num_samples=5):
    """Hiển thị một số ảnh đã pseudo-label để kiểm tra"""

    print("=" * 50)
    print("VISUALIZING PSEUDO LABELS")
    print("=" * 50)

    import random

    images = list(OUTPUT_IMAGES_DIR.glob("*.jpg"))
    if not images:
        print("No pseudo-labeled images found!")
        return

    samples = random.sample(images, min(num_samples, len(images)))

    VIS_DIR = OUTPUT_DIR / "visualize"
    VIS_DIR.mkdir(exist_ok=True)

    colors = [
        (255, 0, 0),    # barely-riped - Blue
        (0, 0, 255),    # over-riped - Red
        (0, 255, 0),    # riped - Green
        (255, 255, 0),  # semi-riped - Cyan
        (255, 0, 255),  # unriped - Magenta
    ]

    for img_path in samples:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        # Read labels
        label_path = OUTPUT_LABELS_DIR / f"{img_path.stem}.txt"
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    cls_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:])

                    # Convert to pixel coordinates
                    x1 = int((x_center - width/2) * w)
                    y1 = int((y_center - height/2) * h)
                    x2 = int((x_center + width/2) * w)
                    y2 = int((y_center + height/2) * h)

                    # Draw box
                    color = colors[cls_id]
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, CLASSES[cls_id], (x1, y1-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Save
        cv2.imwrite(str(VIS_DIR / img_path.name), img)
        print(f"✓ Saved: {VIS_DIR / img_path.name}")

    print(f"\nVisualized images saved to: {VIS_DIR}")


# ==================== MAIN ====================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pseudo-labeling for Coffee Bean Dataset")
    parser.add_argument("--label", action="store_true", help="Run pseudo-labeling")
    parser.add_argument("--merge", action="store_true", help="Merge datasets")
    parser.add_argument("--visualize", action="store_true", help="Visualize pseudo labels")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD, help="Confidence threshold")

    args = parser.parse_args()

    CONF_THRESHOLD = args.conf

    if args.label:
        pseudo_label()
    elif args.merge:
        merge_datasets()
    elif args.visualize:
        visualize()
    else:
        # Default: run pseudo-labeling
        pseudo_label()
