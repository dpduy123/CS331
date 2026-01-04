"""
Prepare dataset for Active Learning from Label Studio export
Uses existing YOLO-format export from Label Studio
"""

import argparse
import shutil
import yaml
from pathlib import Path


def prepare_dataset_from_ls_export(export_dir, output_dir='data_round1', train_split=0.8):
    """
    Prepare YOLO dataset from Label Studio YOLO export

    Args:
        export_dir: Label Studio export directory (project-1-at-2025...)
        output_dir: Output directory for training
        train_split: Train/val split ratio
    """

    export_dir = Path(export_dir)
    output_dir = Path(output_dir)

    # Create output structure
    train_img_dir = output_dir / 'images' / 'train'
    val_img_dir = output_dir / 'images' / 'val'
    train_lbl_dir = output_dir / 'labels' / 'train'
    val_lbl_dir = output_dir / 'labels' / 'val'

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Get all images
    src_img_dir = export_dir / 'images'
    src_lbl_dir = export_dir / 'labels'

    images = sorted(list(src_img_dir.glob('*.jpg')) + list(src_img_dir.glob('*.png')))

    print(f"\n📊 Found {len(images)} annotated images")

    # Split train/val
    split_idx = int(len(images) * train_split)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    print(f"📦 Train: {len(train_images)} images")
    print(f"📦 Val: {len(val_images)} images")

    # Copy files
    print(f"\n📁 Copying files to {output_dir}...")

    for img in train_images:
        shutil.copy2(img, train_img_dir / img.name)
        lbl = src_lbl_dir / f"{img.stem}.txt"
        if lbl.exists():
            shutil.copy2(lbl, train_lbl_dir / lbl.name)

    for img in val_images:
        shutil.copy2(img, val_img_dir / img.name)
        lbl = src_lbl_dir / f"{img.stem}.txt"
        if lbl.exists():
            shutil.copy2(lbl, val_lbl_dir / lbl.name)

    # Read class names from export
    classes_file = export_dir / 'classes.txt'
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f if line.strip()]

    print(f"\n🏷️  Classes: {classes}")

    # Create data.yaml
    data_yaml = {
        'path': str(output_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(classes),
        'names': classes
    }

    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    print(f"\n✅ Dataset prepared!")
    print(f"📄 Config: {yaml_path}")
    print(f"\n{'='*60}")
    print(f"🎓 Ready to train! Run:")
    print(f"")
    print(f"  python scripts/train_yolo.py \\")
    print(f"    --data ./{output_dir} \\")
    print(f"    --model n \\")
    print(f"    --epochs 100 \\")
    print(f"    --batch 16 \\")
    print(f"    --name coffee_round1")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare dataset from Label Studio YOLO export'
    )

    parser.add_argument('--export', type=str, required=True,
                       help='Label Studio export directory (project-1-at-2025...)')
    parser.add_argument('--output', type=str, default='data_round1',
                       help='Output directory (default: data_round1)')
    parser.add_argument('--train-split', type=float, default=0.8,
                       help='Train/val split ratio (default: 0.8)')

    args = parser.parse_args()

    prepare_dataset_from_ls_export(
        export_dir=args.export,
        output_dir=args.output,
        train_split=args.train_split
    )


if __name__ == '__main__':
    main()
