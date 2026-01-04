"""
SAM (Segment Anything Model) for Coffee Bean Segmentation
Ket hop voi YOLO bbox de segment tung hat ca phe
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# KAGGLE CODE - Copy vao notebook
# ============================================================

KAGGLE_SAM_CODE = '''
# ============================================================
# SAM + YOLO Coffee Bean Segmentation
# ============================================================

# Cell 1: Install SAM
!pip install segment-anything -q
!wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
print("SAM downloaded!")

# Cell 2: Import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO
import torch

# Cell 3: Load models
# Load YOLO (model da train)
yolo_model = YOLO('/kaggle/working/runs/coffee_round1/weights/best.pt')

# Load SAM
sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
sam.to("cuda" if torch.cuda.is_available() else "cpu")
sam_predictor = SamPredictor(sam)
print("Models loaded!")

# Cell 4: Helper functions
def show_mask(mask, ax, color=None, alpha=0.5):
    """Hien thi mask len anh"""
    if color is None:
        color = np.array([30/255, 144/255, 255/255, alpha])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax, label=None, color='green'):
    """Ve bbox"""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor='none', lw=2))
    if label:
        ax.text(x0, y0 - 5, label, color=color, fontsize=8, backgroundcolor='white')

# Mau cho tung class
CLASS_COLORS = {
    'barely-riped': np.array([255, 99, 71, 128]) / 255,    # Orange
    'over-riped': np.array([44, 44, 44, 128]) / 255,       # Dark
    'riped': np.array([196, 166, 157, 128]) / 255,         # Brown
    'semi-riped': np.array([255, 217, 61, 128]) / 255,     # Yellow
    'unriped': np.array([144, 238, 144, 128]) / 255        # Green
}

# Cell 5: Segment mot anh
import os
import random

DATASET_PATH = "/kaggle/input/lamdongcoffeebeanripenessdataset"
img_dir = f"{DATASET_PATH}/images/val"

# Chon 1 anh
images = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
img_name = random.choice(images)
img_path = f"{img_dir}/{img_name}"

# Load image
image = cv2.imread(img_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# YOLO detect
yolo_results = yolo_model.predict(img_path, conf=0.25, verbose=False)
boxes = yolo_results[0].boxes

print(f"Image: {img_name}")
print(f"YOLO detected: {len(boxes)} beans")

# SAM segment
sam_predictor.set_image(image_rgb)

# Segment tung bean
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Anh goc voi YOLO bbox
axes[0].imshow(image_rgb)
axes[0].set_title(f"YOLO Detection: {len(boxes)} beans")
axes[0].axis('off')

for box in boxes:
    xyxy = box.xyxy[0].cpu().numpy()
    cls_name = yolo_model.names[int(box.cls[0])]
    show_box(xyxy, axes[0], label=cls_name, color='lime')

# Anh voi SAM masks
axes[1].imshow(image_rgb)
axes[1].set_title("SAM Segmentation")
axes[1].axis('off')

for box in boxes:
    xyxy = box.xyxy[0].cpu().numpy()
    cls_name = yolo_model.names[int(box.cls[0])]

    # SAM segment voi bbox prompt
    masks, scores, _ = sam_predictor.predict(
        box=xyxy,
        multimask_output=False
    )

    # Hien thi mask
    color = CLASS_COLORS.get(cls_name, np.array([0.5, 0.5, 0.5, 0.5]))
    show_mask(masks[0], axes[1], color=color)

plt.tight_layout()
plt.show()

# Cell 6: Segment nhieu anh
sample_images = random.sample(images, min(4, len(images)))

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for idx, img_name in enumerate(sample_images):
    img_path = f"{img_dir}/{img_name}"
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # YOLO detect
    yolo_results = yolo_model.predict(img_path, conf=0.25, verbose=False)
    boxes = yolo_results[0].boxes

    # Row 1: YOLO bbox
    axes[0][idx].imshow(image_rgb)
    axes[0][idx].set_title(f"YOLO: {len(boxes)} beans", fontsize=9)
    axes[0][idx].axis('off')
    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        show_box(xyxy, axes[0][idx], color='lime')

    # Row 2: SAM masks
    sam_predictor.set_image(image_rgb)
    axes[1][idx].imshow(image_rgb)
    axes[1][idx].set_title("SAM Segmentation", fontsize=9)
    axes[1][idx].axis('off')

    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        cls_name = yolo_model.names[int(box.cls[0])]
        masks, _, _ = sam_predictor.predict(box=xyxy, multimask_output=False)
        color = CLASS_COLORS.get(cls_name, np.array([0.5, 0.5, 0.5, 0.5]))
        show_mask(masks[0], axes[1][idx], color=color)

plt.tight_layout()
plt.suptitle("YOLO Detection vs SAM Segmentation", fontsize=14, y=1.02)
plt.show()

# Cell 7: Thong ke dien tich tung class
img_path = f"{img_dir}/{random.choice(images)}"
image = cv2.imread(img_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

yolo_results = yolo_model.predict(img_path, conf=0.25, verbose=False)
boxes = yolo_results[0].boxes

sam_predictor.set_image(image_rgb)

class_areas = {}
total_pixels = image.shape[0] * image.shape[1]

for box in boxes:
    xyxy = box.xyxy[0].cpu().numpy()
    cls_name = yolo_model.names[int(box.cls[0])]

    masks, _, _ = sam_predictor.predict(box=xyxy, multimask_output=False)
    area = masks[0].sum()

    if cls_name not in class_areas:
        class_areas[cls_name] = 0
    class_areas[cls_name] += area

print("\\nDien tich tung loai (pixels):")
print("=" * 40)
for cls_name, area in sorted(class_areas.items(), key=lambda x: -x[1]):
    pct = area / total_pixels * 100
    print(f"  {cls_name}: {area:,} px ({pct:.2f}%)")
'''


def print_kaggle_code():
    """In code de copy vao Kaggle"""
    print("=" * 60)
    print("COPY CODE NAY VAO KAGGLE NOTEBOOK")
    print("=" * 60)
    print(KAGGLE_SAM_CODE)


if __name__ == "__main__":
    print("SAM Segmentation Script")
    print("=" * 60)
    print("\nScript nay cung cap code de chay SAM tren Kaggle.")
    print("Copy code ben duoi vao Kaggle notebook de su dung.\n")
    print_kaggle_code()
