"""
Script kiem tra toan bo anh trong dataset co load duoc khong
Dung de debug loi OpenCV truoc khi train YOLO
Co the plot anh ra de xem
"""

import os
import cv2
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def check_single_image(img_path):
    """Kiem tra 1 anh co load duoc khong"""
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return False, "cv2.imread tra ve None"
        if img.shape[0] == 0 or img.shape[1] == 0:
            return False, f"Anh co kich thuoc 0: {img.shape}"
        return True, f"OK - Shape: {img.shape}"
    except Exception as e:
        return False, f"Exception: {str(e)}"


def check_all_images(image_dir):
    """Kiem tra toan bo anh trong thu muc"""
    image_dir = Path(image_dir)

    if not image_dir.exists():
        print(f"ERROR: Thu muc khong ton tai: {image_dir}")
        return

    # Tim tat ca file anh
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    for ext in extensions:
        image_files.extend(image_dir.glob(ext))
        image_files.extend(image_dir.glob(ext.upper()))

    if not image_files:
        print(f"WARNING: Khong tim thay anh trong {image_dir}")
        return

    print(f"Kiem tra {len(image_files)} anh trong {image_dir}")
    print("=" * 60)

    ok_count = 0
    fail_count = 0
    failed_images = []

    for img_path in image_files:
        success, message = check_single_image(img_path)
        if success:
            ok_count += 1
        else:
            fail_count += 1
            failed_images.append((img_path.name, message))
            print(f"FAILED: {img_path.name} - {message}")

    print("=" * 60)
    print(f"Ket qua: {ok_count}/{len(image_files)} anh OK")

    if fail_count > 0:
        print(f"\nCo {fail_count} anh bi loi:")
        for name, msg in failed_images:
            print(f"  - {name}: {msg}")
    else:
        print("\nTat ca anh deu load duoc!")

    return ok_count, fail_count, failed_images


def check_dataset(dataset_path):
    """Kiem tra toan bo dataset (train va val)"""
    dataset_path = Path(dataset_path)

    print("=" * 60)
    print(f"KIEM TRA DATASET: {dataset_path}")
    print("=" * 60)

    # Check train
    train_path = dataset_path / "images" / "train"
    if train_path.exists():
        print("\n[TRAIN SET]")
        check_all_images(train_path)
    else:
        print(f"\nWARNING: Khong tim thay {train_path}")

    # Check val
    val_path = dataset_path / "images" / "val"
    if val_path.exists():
        print("\n[VALIDATION SET]")
        check_all_images(val_path)
    else:
        print(f"\nWARNING: Khong tim thay {val_path}")


def check_with_resize(image_dir):
    """Kiem tra anh voi cv2.resize (giong nhu YOLO lam)"""
    image_dir = Path(image_dir)

    extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    for ext in extensions:
        image_files.extend(image_dir.glob(ext))

    print(f"\nKiem tra {len(image_files)} anh voi cv2.resize (640x640)...")
    print("=" * 60)

    fail_count = 0
    for img_path in image_files:
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"FAILED: {img_path.name} - Khong load duoc")
                fail_count += 1
                continue

            # Thu resize nhu YOLO
            resized = cv2.resize(img, (640, 640))
            if resized is None:
                print(f"FAILED: {img_path.name} - Resize that bai")
                fail_count += 1
        except Exception as e:
            print(f"FAILED: {img_path.name} - {str(e)}")
            fail_count += 1

    if fail_count == 0:
        print("Tat ca anh deu resize duoc!")
    else:
        print(f"\nCo {fail_count} anh bi loi khi resize")


def plot_sample_images(image_dir, num_images=6, cols=3, figsize=(15, 10)):
    """Plot mot so anh mau tu dataset"""
    image_dir = Path(image_dir)

    extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    for ext in extensions:
        image_files.extend(image_dir.glob(ext))

    if not image_files:
        print(f"Khong tim thay anh trong {image_dir}")
        return

    # Lay ngau nhien num_images anh
    import random
    sample_images = random.sample(image_files, min(num_images, len(image_files)))

    rows = (len(sample_images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    for idx, img_path in enumerate(sample_images):
        row = idx // cols
        col = idx % cols
        ax = axes[row][col]

        img = cv2.imread(str(img_path))
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            ax.set_title(f"{img_path.name}\n{img.shape}", fontsize=8)
        else:
            ax.set_title(f"{img_path.name}\nFAILED TO LOAD", fontsize=8, color='red')
        ax.axis('off')

    # An cac subplot thua
    for idx in range(len(sample_images), rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row][col].axis('off')

    plt.tight_layout()
    plt.suptitle(f"Sample Images from {image_dir.name}", fontsize=12, y=1.02)
    plt.show()


def plot_with_resize_comparison(image_dir, num_images=3, figsize=(15, 8)):
    """Plot anh goc va anh sau khi resize de so sanh"""
    image_dir = Path(image_dir)

    extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    for ext in extensions:
        image_files.extend(image_dir.glob(ext))

    if not image_files:
        print(f"Khong tim thay anh trong {image_dir}")
        return

    import random
    sample_images = random.sample(image_files, min(num_images, len(image_files)))

    fig, axes = plt.subplots(num_images, 2, figsize=figsize)

    if num_images == 1:
        axes = [axes]

    for idx, img_path in enumerate(sample_images):
        img = cv2.imread(str(img_path))

        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Anh goc
            axes[idx][0].imshow(img_rgb)
            axes[idx][0].set_title(f"Original: {img_path.name}\nShape: {img.shape}", fontsize=9)
            axes[idx][0].axis('off')

            # Anh resize
            try:
                resized = cv2.resize(img, (640, 640))
                resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                axes[idx][1].imshow(resized_rgb)
                axes[idx][1].set_title(f"Resized: 640x640\nOK!", fontsize=9, color='green')
            except Exception as e:
                axes[idx][1].set_title(f"Resize FAILED!\n{str(e)[:30]}", fontsize=9, color='red')
            axes[idx][1].axis('off')
        else:
            axes[idx][0].set_title(f"FAILED TO LOAD\n{img_path.name}", fontsize=9, color='red')
            axes[idx][0].axis('off')
            axes[idx][1].axis('off')

    plt.tight_layout()
    plt.suptitle("Original vs Resized (640x640)", fontsize=12, y=1.02)
    plt.show()


# ============================================================
# KAGGLE VERSION - Copy phan nay vao Kaggle notebook
# ============================================================

KAGGLE_CODE = '''
# Copy code nay vao Kaggle notebook de kiem tra:

import os
import cv2

def check_dataset_kaggle(dataset_path):
    """Kiem tra dataset tren Kaggle"""
    train_path = f"{dataset_path}/images/train"
    val_path = f"{dataset_path}/images/val"

    print(f"Checking dataset: {dataset_path}")
    print("=" * 60)

    for split_name, split_path in [("TRAIN", train_path), ("VAL", val_path)]:
        if not os.path.exists(split_path):
            print(f"{split_name}: Path not found - {split_path}")
            continue

        images = [f for f in os.listdir(split_path) if f.endswith(('.jpg', '.png'))]
        print(f"\\n[{split_name}] Checking {len(images)} images...")

        ok = 0
        failed = []

        for img_name in images:
            img_path = f"{split_path}/{img_name}"
            img = cv2.imread(img_path)

            if img is None:
                failed.append(img_name)
            else:
                # Thu resize
                try:
                    resized = cv2.resize(img, (640, 640))
                    ok += 1
                except Exception as e:
                    failed.append(f"{img_name} (resize error: {e})")

        print(f"   OK: {ok}/{len(images)}")
        if failed:
            print(f"   FAILED: {len(failed)}")
            for f in failed[:5]:  # Chi hien 5 cai dau
                print(f"      - {f}")

# SU DUNG:
DATASET_PATH = "/kaggle/input/lamdongcoffeebeanripenessdataset"
check_dataset_kaggle(DATASET_PATH)
'''


if __name__ == "__main__":
    print("=" * 60)
    print("COFFEE BEAN IMAGE CHECKER")
    print("=" * 60)

    # Mac dinh kiem tra kaggle_dataset_round1
    default_path = Path(__file__).parent.parent / "kaggle_dataset_round1"

    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    elif default_path.exists():
        dataset_path = default_path
    else:
        print("Su dung: python check_images.py <duong_dan_dataset>")
        print("\nHoac copy code Kaggle ben duoi vao notebook:\n")
        print(KAGGLE_CODE)
        sys.exit(0)

    check_dataset(dataset_path)

    # Check voi resize
    train_path = Path(dataset_path) / "images" / "train"
    if train_path.exists():
        check_with_resize(train_path)

    print("\n" + "=" * 60)
    print("CODE CHO KAGGLE NOTEBOOK:")
    print("=" * 60)
    print(KAGGLE_CODE)
