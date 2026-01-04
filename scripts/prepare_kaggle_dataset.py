"""
Prepare dataset for Kaggle upload
Creates a clean dataset package ready for Kaggle
"""

import argparse
import shutil
import yaml
from pathlib import Path
import zipfile


def prepare_kaggle_dataset(export_dir, output_dir='kaggle_dataset', train_split=0.8):
    """
    Prepare dataset for Kaggle from Label Studio YOLO export

    Creates:
    - Organized YOLO dataset structure
    - data.yaml configuration
    - README.md with instructions
    - Everything zipped for upload

    Args:
        export_dir: Label Studio export directory
        output_dir: Output directory
        train_split: Train/val split ratio
    """

    export_dir = Path(export_dir)
    output_dir = Path(output_dir)

    # Clean output directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # Create YOLO structure
    train_img_dir = output_dir / 'images' / 'train'
    val_img_dir = output_dir / 'images' / 'val'
    train_lbl_dir = output_dir / 'labels' / 'train'
    val_lbl_dir = output_dir / 'labels' / 'val'

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n📦 Preparing dataset for Kaggle upload...")
    print(f"   Export: {export_dir}")
    print(f"   Output: {output_dir}\n")

    # Get source directories
    src_img_dir = export_dir / 'images'
    src_lbl_dir = export_dir / 'labels'

    if not src_img_dir.exists():
        print(f"❌ Images directory not found: {src_img_dir}")
        return False

    # Get all images
    images = sorted(list(src_img_dir.glob('*.jpg')) +
                   list(src_img_dir.glob('*.png')) +
                   list(src_img_dir.glob('*.JPG')) +
                   list(src_img_dir.glob('*.PNG')))

    if len(images) == 0:
        print(f"❌ No images found in {src_img_dir}")
        return False

    print(f"✓ Found {len(images)} annotated images")

    # Split train/val
    split_idx = int(len(images) * train_split)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    print(f"✓ Train split: {len(train_images)} images")
    print(f"✓ Val split: {len(val_images)} images")

    # Copy files with progress
    print(f"\n📁 Copying files...")

    train_labels_count = 0
    for img in train_images:
        shutil.copy2(img, train_img_dir / img.name)
        lbl = src_lbl_dir / f"{img.stem}.txt"
        if lbl.exists():
            shutil.copy2(lbl, train_lbl_dir / lbl.name)
            train_labels_count += 1

    val_labels_count = 0
    for img in val_images:
        shutil.copy2(img, val_img_dir / img.name)
        lbl = src_lbl_dir / f"{img.stem}.txt"
        if lbl.exists():
            shutil.copy2(lbl, val_lbl_dir / lbl.name)
            val_labels_count += 1

    print(f"✓ Copied {len(train_images)} train images ({train_labels_count} labels)")
    print(f"✓ Copied {len(val_images)} val images ({val_labels_count} labels)")

    # Read class names
    classes_file = export_dir / 'classes.txt'
    if classes_file.exists():
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f if line.strip()]
    else:
        # Default classes
        classes = ['barely-riped', 'over-riped', 'riped', 'semi-riped', 'unriped']

    print(f"\n🏷️  Classes ({len(classes)}):")
    for i, cls in enumerate(classes):
        print(f"   {i}: {cls}")

    # Create data.yaml with placeholder path
    # Users will need to update this path in Kaggle to match their dataset name
    data_yaml = {
        'path': '.',  # Relative path - will be in same directory as data.yaml
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(classes),
        'names': classes
    }

    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    print(f"\n✓ Created data.yaml (path: '.' - relative to dataset root)")

    # Create README for Kaggle
    readme_content = f"""# Coffee Bean Ripeness Dataset

## Dataset Description

This dataset contains annotated coffee bean images for ripeness classification.

**Classes ({len(classes)}):**
"""
    for i, cls in enumerate(classes):
        readme_content += f"- **{i}: {cls}**\n"

    readme_content += f"""
## Dataset Statistics

- **Total images:** {len(images)}
- **Training images:** {len(train_images)}
- **Validation images:** {len(val_images)}
- **Total annotations:** {train_labels_count + val_labels_count}

## Dataset Structure

```
dataset/
├── data.yaml           # YOLOv8 configuration
├── images/
│   ├── train/         # Training images
│   └── val/           # Validation images
└── labels/
    ├── train/         # Training labels (YOLO format)
    └── val/           # Validation labels (YOLO format)
```

## Label Format

Labels are in YOLO format (one .txt file per image):
```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are normalized (0-1).

## Usage

### Training YOLOv8 on Kaggle

```python
# Install ultralytics
!pip install ultralytics

# Import
from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.pt')

# Train
results = model.train(
    data='/kaggle/input/coffee-bean-dataset/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0  # Use GPU
)

# Validate
metrics = model.val()

# Save model
model.export(format='onnx')
```

## License

Research and educational use.
"""

    readme_path = output_dir / 'README.md'
    with open(readme_path, 'w') as f:
        f.write(readme_content)

    print(f"✓ Created README.md")

    # Create Kaggle notebook template
    notebook_content = f"""# Coffee Bean Detection - YOLOv8 Training

## 1. Install Dependencies

# Kaggle already has: torch, opencv, numpy, pandas, matplotlib, sklearn, yaml
# Only need to install: ultralytics (YOLOv8)

!pip install ultralytics -q

print("✅ Installation complete!")

## 2. Import Libraries

from ultralytics import YOLO
import torch
import cv2
import numpy as np
import yaml
import os
from pathlib import Path

print("✅ All imports successful!")
print(f"PyTorch: {{torch.__version__}}")
print(f"OpenCV: {{cv2.__version__}}")
print(f"NumPy: {{np.__version__}}")

## 3. Verify Dataset & Get Path

# List all datasets
print("\\n📁 Available datasets:")
!ls /kaggle/input/

# IMPORTANT: Replace <YOUR-DATASET-NAME> with your actual dataset folder name
DATASET_NAME = '<YOUR-DATASET-NAME>'  # e.g., 'lamdongcoffeebeanripenessdataset'
DATASET_PATH = f'/kaggle/input/{{DATASET_NAME}}'

print(f"\\nUsing dataset: {{DATASET_PATH}}")
!ls $DATASET_PATH

## 4. Fix data.yaml Path

# Read original data.yaml
with open(f'{{DATASET_PATH}}/data.yaml', 'r') as f:
    data_config = yaml.safe_load(f)

# Update path to point to dataset location
data_config['path'] = DATASET_PATH

# Save to working directory
with open('/kaggle/working/data.yaml', 'w') as f:
    yaml.dump(data_config, f, sort_keys=False)

print("\\n✅ Updated data.yaml:")
print(f"   path: {{data_config['path']}}")
print(f"   train: {{data_config['train']}}")
print(f"   val: {{data_config['val']}}")
print(f"   classes: {{data_config['nc']}}")

# Verify paths exist
train_path = os.path.join(data_config['path'], data_config['train'])
val_path = os.path.join(data_config['path'], data_config['val'])

if os.path.exists(train_path) and os.path.exists(val_path):
    print(f"\\n✅ Paths verified successfully!")
else:
    print(f"\\n❌ Error: Paths not found!")
    print(f"   Train: {{train_path}}")
    print(f"   Val: {{val_path}}")

## 5. Train Model

# Load pretrained YOLOv8n
model = YOLO('yolov8n.pt')

# Train - use FIXED data.yaml from working directory
results = model.train(
    data='/kaggle/working/data.yaml',  # ← Use fixed data.yaml
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,  # Use GPU
    patience=20,
    save=True,
    project='/kaggle/working/runs',
    name='coffee_bean_round1',
    # Data augmentation
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10.0,
    translate=0.1,
    scale=0.5,
    flipud=0.5,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1
)

## Validate Model

metrics = model.val()
print(f"mAP50: {{metrics.box.map50:.3f}}")
print(f"mAP50-95: {{metrics.box.map:.3f}}")

## Save Model

# Best model saved at:
# /kaggle/working/runs/coffee_bean_v1/weights/best.pt

# Copy to output for download
import shutil
shutil.copy2('/kaggle/working/runs/coffee_bean_v1/weights/best.pt',
             '/kaggle/working/best.pt')

print("✅ Model saved to /kaggle/working/best.pt")
"""

    notebook_path = output_dir / 'kaggle_train_notebook.txt'
    with open(notebook_path, 'w') as f:
        f.write(notebook_content)

    print(f"✓ Created kaggle_train_notebook.txt")

    # Create zip file
    print(f"\n📦 Creating zip file...")
    zip_path = Path(f"{output_dir}.zip")

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in output_dir.rglob('*'):
            if file.is_file():
                arcname = file.relative_to(output_dir)
                zipf.write(file, arcname)

    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"✓ Created {zip_path.name} ({zip_size_mb:.1f} MB)")

    # Print summary
    print(f"\n{'='*70}")
    print(f"✅ Kaggle Dataset Ready!")
    print(f"{'='*70}")
    print(f"\n📊 Dataset Summary:")
    print(f"   - Images: {len(images)} ({len(train_images)} train, {len(val_images)} val)")
    print(f"   - Labels: {train_labels_count + val_labels_count}")
    print(f"   - Classes: {len(classes)}")
    print(f"   - Output: {output_dir}/")
    print(f"   - Zip: {zip_path} ({zip_size_mb:.1f} MB)")
    print(f"\n📝 Next Steps:")
    print(f"   1. Upload {zip_path.name} to Kaggle Datasets:")
    print(f"      https://www.kaggle.com/datasets")
    print(f"")
    print(f"   2. Create new dataset → Upload → Select {zip_path.name}")
    print(f"")
    print(f"   3. Create Kaggle Notebook:")
    print(f"      - Copy code from kaggle_train_notebook.txt")
    print(f"      - Add your dataset as data source")
    print(f"      - Run with GPU accelerator")
    print(f"")
    print(f"   4. Download trained model:")
    print(f"      - /kaggle/working/best.pt")
    print(f"")
    print(f"💡 See KAGGLE_GUIDE.md for detailed instructions!")
    print(f"{'='*70}\n")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Prepare dataset for Kaggle upload'
    )

    parser.add_argument('--export', type=str, required=True,
                       help='Label Studio export directory')
    parser.add_argument('--output', type=str, default='kaggle_dataset',
                       help='Output directory (default: kaggle_dataset)')
    parser.add_argument('--train-split', type=float, default=0.8,
                       help='Train/val split ratio (default: 0.8)')

    args = parser.parse_args()

    prepare_kaggle_dataset(
        export_dir=args.export,
        output_dir=args.output,
        train_split=args.train_split
    )


if __name__ == '__main__':
    main()
