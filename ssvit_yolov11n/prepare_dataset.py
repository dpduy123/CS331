"""
Prepare dataset for SSViT-YOLOv11n training

Reorganizes the kaggle_dataset_label folder to the expected structure:
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
"""

import shutil
from pathlib import Path
import argparse


def prepare_dataset(source_dir: str, output_dir: str):
    """
    Reorganize dataset from kaggle format to SSViT training format

    Args:
        source_dir: Path to kaggle_dataset_label folder
        output_dir: Path to output dataset folder
    """
    source = Path(source_dir)
    output = Path(output_dir)

    # Create output structure
    for split in ['train', 'val']:
        (output / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Copy files
    for split in ['train', 'val']:
        # Copy images
        src_images = source / 'images' / split
        dst_images = output / 'images' / split

        if src_images.exists():
            for img in src_images.glob('*'):
                if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    shutil.copy2(img, dst_images / img.name)
                    print(f"  Copied image: {img.name}")

        # Copy labels
        src_labels = source / 'labels' / split
        dst_labels = output / 'labels' / split

        if src_labels.exists():
            for lbl in src_labels.glob('*.txt'):
                shutil.copy2(lbl, dst_labels / lbl.name)
                print(f"  Copied label: {lbl.name}")

    # Print summary
    print("\n" + "=" * 50)
    print("Dataset prepared successfully!")
    print("=" * 50)

    for split in ['train', 'val']:
        n_images = len(list((output / 'images' / split).glob('*')))
        n_labels = len(list((output / 'labels' / split).glob('*.txt')))
        print(f"  {split}: {n_images} images, {n_labels} labels")

    print(f"\nDataset location: {output}")
    print("\nTo train, run:")
    print(f"  python train.py --data-path {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for SSViT-YOLOv11n")
    parser.add_argument('--source', type=str,
                        default='../kaggle_dataset_label',
                        help='Source dataset folder (kaggle format)')
    parser.add_argument('--output', type=str,
                        default='./dataset',
                        help='Output dataset folder')

    args = parser.parse_args()

    print("Preparing dataset...")
    print(f"  Source: {args.source}")
    print(f"  Output: {args.output}")
    print()

    prepare_dataset(args.source, args.output)
