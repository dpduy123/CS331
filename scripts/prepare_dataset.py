"""
Prepare dataset for YOLOv11 training
Splits data into train/val/test sets

Usage:
    python scripts/prepare_dataset.py --images "Hình cà phê" --labels labels --output dataset

    # With custom split ratio
    python scripts/prepare_dataset.py --images images --labels labels --output dataset --train 0.8 --val 0.15 --test 0.05
"""

import argparse
import shutil
import random
from pathlib import Path
from collections import defaultdict


def prepare_dataset(
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.15,
    test_ratio: float = 0.05,
    seed: int = 42,
    copy_files: bool = True
):
    """
    Prepare dataset by splitting into train/val/test

    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing YOLO format labels (.txt)
        output_dir: Output directory for split dataset
        train_ratio: Ratio for training set (default: 0.8)
        val_ratio: Ratio for validation set (default: 0.15)
        test_ratio: Ratio for test set (default: 0.05)
        seed: Random seed for reproducibility
        copy_files: If True, copy files; if False, create symlinks
    """

    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    output_path = Path(output_dir)

    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        print(f"WARNING: Ratios sum to {total_ratio}, normalizing...")
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio

    print(f"Split ratios: train={train_ratio:.2f}, val={val_ratio:.2f}, test={test_ratio:.2f}")

    # Find all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    all_images = []

    for ext in image_extensions:
        all_images.extend(images_path.glob(f'*{ext}'))
        all_images.extend(images_path.glob(f'*{ext.upper()}'))

    print(f"Found {len(all_images)} images in {images_path}")

    if len(all_images) == 0:
        print("ERROR: No images found!")
        return

    # Match images with labels
    matched_pairs = []
    missing_labels = []

    for img_path in all_images:
        label_path = labels_path / f"{img_path.stem}.txt"

        if label_path.exists():
            matched_pairs.append((img_path, label_path))
        else:
            missing_labels.append(img_path)

    print(f"Matched pairs: {len(matched_pairs)}")
    print(f"Missing labels: {len(missing_labels)}")

    if missing_labels:
        print("\nImages without labels:")
        for img in missing_labels[:10]:
            print(f"  - {img.name}")
        if len(missing_labels) > 10:
            print(f"  ... and {len(missing_labels) - 10} more")

    if len(matched_pairs) == 0:
        print("ERROR: No matched image-label pairs found!")
        return

    # Shuffle and split
    random.seed(seed)
    random.shuffle(matched_pairs)

    n_total = len(matched_pairs)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    train_pairs = matched_pairs[:n_train]
    val_pairs = matched_pairs[n_train:n_train + n_val]
    test_pairs = matched_pairs[n_train + n_val:]

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_pairs)}")
    print(f"  Val: {len(val_pairs)}")
    print(f"  Test: {len(test_pairs)}")

    # Create output directories
    splits = {
        'train': train_pairs,
        'val': val_pairs,
        'test': test_pairs
    }

    for split_name, pairs in splits.items():
        if len(pairs) == 0:
            continue

        images_out = output_path / split_name / 'images'
        labels_out = output_path / split_name / 'labels'

        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        for img_path, label_path in pairs:
            img_dest = images_out / img_path.name
            label_dest = labels_out / label_path.name

            if copy_files:
                shutil.copy2(img_path, img_dest)
                shutil.copy2(label_path, label_dest)
            else:
                # Create symlinks
                if not img_dest.exists():
                    img_dest.symlink_to(img_path.absolute())
                if not label_dest.exists():
                    label_dest.symlink_to(label_path.absolute())

        print(f"Created {split_name}: {len(pairs)} samples")

    # Analyze class distribution
    print("\nClass distribution:")
    analyze_labels(output_path)

    # Create dataset.yaml
    create_yaml(output_path)

    print(f"\nDataset prepared at: {output_path}")
    print(f"Dataset config: {output_path / 'dataset.yaml'}")


def analyze_labels(dataset_path: Path):
    """Analyze class distribution in dataset"""

    class_names = ['barely-riped', 'over-riped', 'riped', 'semi-riped', 'unriped']

    for split in ['train', 'val', 'test']:
        labels_dir = dataset_path / split / 'labels'

        if not labels_dir.exists():
            continue

        class_counts = defaultdict(int)
        total_objects = 0

        for label_file in labels_dir.glob('*.txt'):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls_id = int(parts[0])
                        class_counts[cls_id] += 1
                        total_objects += 1

        print(f"\n  {split.upper()}:")
        print(f"    Total objects: {total_objects}")
        for cls_id, count in sorted(class_counts.items()):
            cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
            pct = count / total_objects * 100 if total_objects > 0 else 0
            print(f"    {cls_id}: {cls_name}: {count} ({pct:.1f}%)")


def create_yaml(dataset_path: Path):
    """Create dataset.yaml file"""

    yaml_content = f"""# Coffee Bean Ripeness Detection Dataset
# Auto-generated by prepare_dataset.py

path: {dataset_path.absolute()}
train: train/images
val: val/images
test: test/images

nc: 5

names:
  0: barely-riped
  1: over-riped
  2: riped
  3: semi-riped
  4: unriped
"""

    yaml_path = dataset_path / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset for YOLOv11 training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python scripts/prepare_dataset.py --images "Hình cà phê" --labels labels --output dataset

    # Custom split ratio
    python scripts/prepare_dataset.py --images images --labels labels --output dataset \\
        --train 0.7 --val 0.2 --test 0.1

    # Use symlinks instead of copying
    python scripts/prepare_dataset.py --images images --labels labels --output dataset --symlink
        """
    )

    parser.add_argument('--images', type=str, required=True,
                        help='Directory containing images')
    parser.add_argument('--labels', type=str, required=True,
                        help='Directory containing YOLO format labels')
    parser.add_argument('--output', type=str, default='dataset',
                        help='Output directory (default: dataset)')

    parser.add_argument('--train', type=float, default=0.8,
                        help='Training set ratio (default: 0.8)')
    parser.add_argument('--val', type=float, default=0.15,
                        help='Validation set ratio (default: 0.15)')
    parser.add_argument('--test', type=float, default=0.05,
                        help='Test set ratio (default: 0.05)')

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--symlink', action='store_true',
                        help='Create symlinks instead of copying files')

    args = parser.parse_args()

    prepare_dataset(
        images_dir=args.images,
        labels_dir=args.labels,
        output_dir=args.output,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed,
        copy_files=not args.symlink
    )


if __name__ == "__main__":
    main()
