"""
Visualize YOLO annotations with bounding boxes on images
Perfect for use in Google Colab or Jupyter Notebooks
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import random

# Class definitions matching your dataset
CLASSES = {
    0: "barely-riped",
    1: "over-riped",
    2: "riped",
    3: "semi-riped",
    4: "unriped"
}

# Colors for each class (BGR format for OpenCV)
COLORS = {
    0: (71, 99, 255),    # Orange/Red for barely-riped
    1: (44, 44, 44),     # Dark gray for over-riped
    2: (157, 166, 196),  # Brown/Pink for riped
    3: (61, 217, 255),   # Yellow for semi-riped
    4: (144, 238, 144)   # Light green for unriped
}

# RGB colors for matplotlib
COLORS_RGB = {
    0: (255, 99, 71),    # Tomato for barely-riped
    1: (44, 44, 44),     # Dark gray for over-riped
    2: (196, 166, 157),  # Brown/Pink for riped
    3: (255, 217, 61),   # Yellow for semi-riped
    4: (144, 238, 144)   # Light green for unriped
}


def load_yolo_annotation(label_path):
    """Load YOLO format annotations from file"""
    annotations = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                annotations.append({
                    'class_id': class_id,
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height
                })
    return annotations


def yolo_to_corners(x_center, y_center, width, height, img_width, img_height):
    """Convert YOLO format to corner coordinates"""
    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    width_px = width * img_width
    height_px = height * img_height

    x1 = int(x_center_px - width_px / 2)
    y1 = int(y_center_px - height_px / 2)
    x2 = int(x_center_px + width_px / 2)
    y2 = int(y_center_px + height_px / 2)

    return x1, y1, x2, y2


def draw_bbox_cv2(image, annotations, show_labels=True, thickness=2):
    """Draw bounding boxes using OpenCV"""
    img = image.copy()
    img_height, img_width = img.shape[:2]

    for ann in annotations:
        class_id = ann['class_id']
        x1, y1, x2, y2 = yolo_to_corners(
            ann['x_center'], ann['y_center'],
            ann['width'], ann['height'],
            img_width, img_height
        )

        # Draw rectangle
        color = COLORS.get(class_id, (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # Draw label
        if show_labels:
            label = CLASSES.get(class_id, f"Class {class_id}")
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # Draw background for text
            cv2.rectangle(img,
                         (x1, y1 - label_size[1] - 4),
                         (x1 + label_size[0], y1),
                         color, -1)

            # Draw text
            cv2.putText(img, label, (x1, y1 - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img


def visualize_image(image_path, label_path, figsize=(15, 10), show_labels=True):
    """Visualize a single image with annotations using matplotlib"""
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return

    # Convert BGR to RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Load annotations
    if Path(label_path).exists():
        annotations = load_yolo_annotation(label_path)
    else:
        print(f"Warning: No label file found at {label_path}")
        annotations = []

    # Draw bounding boxes
    img_with_boxes = draw_bbox_cv2(img, annotations, show_labels=show_labels)
    img_with_boxes_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)

    # Plot
    plt.figure(figsize=figsize)
    plt.imshow(img_with_boxes_rgb)
    plt.axis('off')
    plt.title(f"{Path(image_path).name} - {len(annotations)} beans detected",
              fontsize=14, pad=10)
    plt.tight_layout()
    plt.show()

    # Print statistics
    class_counts = {}
    for ann in annotations:
        class_id = ann['class_id']
        class_name = CLASSES.get(class_id, f"Class {class_id}")
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    print(f"\n📊 Bean counts:")
    for class_name, count in sorted(class_counts.items()):
        print(f"   {class_name}: {count}")
    print(f"   Total: {len(annotations)}")


def visualize_grid(image_dir, label_dir, num_images=4, cols=2, figsize=(15, 15),
                   show_labels=True, random_sample=True):
    """Visualize multiple images in a grid"""
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)

    # Get all image files
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))

    if random_sample:
        image_files = random.sample(image_files, min(num_images, len(image_files)))
    else:
        image_files = image_files[:num_images]

    rows = (len(image_files) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)

    for idx, image_path in enumerate(image_files):
        row = idx // cols
        col = idx % cols
        ax = axes[row][col]

        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            continue

        # Load annotations
        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            annotations = load_yolo_annotation(label_path)
        else:
            annotations = []

        # Draw bounding boxes
        img_with_boxes = draw_bbox_cv2(img, annotations, show_labels=show_labels)
        img_with_boxes_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)

        # Display
        ax.imshow(img_with_boxes_rgb)
        ax.axis('off')
        ax.set_title(f"{image_path.name}\n{len(annotations)} beans", fontsize=10)

    # Hide empty subplots
    for idx in range(len(image_files), rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row][col].axis('off')

    plt.tight_layout()
    plt.show()


def show_class_legend():
    """Display color legend for classes"""
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('off')

    y_pos = 0.8
    for class_id in sorted(CLASSES.keys()):
        class_name = CLASSES[class_id]
        color = tuple(c/255 for c in COLORS_RGB[class_id])

        # Draw colored box
        rect = plt.Rectangle((0.1, y_pos - 0.08), 0.1, 0.15,
                             facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)

        # Add text
        ax.text(0.25, y_pos, class_name, fontsize=14, va='center')

        y_pos -= 0.2

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.title("Coffee Bean Ripeness Classes", fontsize=16, pad=20)
    plt.tight_layout()
    plt.show()


# Example usage functions for Google Colab
def colab_example_single_image():
    """Example: Visualize a single image in Colab"""
    print("Example usage for single image:")
    print()
    print("# Visualize one image")
    print("visualize_image(")
    print("    image_path='kaggle_dataset_round1/images/train/image_001.jpg',")
    print("    label_path='kaggle_dataset_round1/labels/train/image_001.txt',")
    print("    figsize=(15, 10),")
    print("    show_labels=True")
    print(")")


def colab_example_grid():
    """Example: Visualize multiple images in grid"""
    print("Example usage for image grid:")
    print()
    print("# Visualize 6 random images in 2x3 grid")
    print("visualize_grid(")
    print("    image_dir='kaggle_dataset_round1/images/train',")
    print("    label_dir='kaggle_dataset_round1/labels/train',")
    print("    num_images=6,")
    print("    cols=3,")
    print("    figsize=(20, 12),")
    print("    show_labels=True,")
    print("    random_sample=True")
    print(")")


if __name__ == "__main__":
    print("Coffee Bean Annotation Visualizer")
    print("=" * 50)
    print()
    print("Available functions:")
    print("  1. visualize_image() - View single image with bboxes")
    print("  2. visualize_grid() - View multiple images in grid")
    print("  3. show_class_legend() - Display color legend")
    print()
    print("Run in Python/Jupyter/Colab:")
    print()
    colab_example_single_image()
    print()
    colab_example_grid()
