"""
Run one round of Active Learning
- Train model on annotated images
- Predict on remaining unannotated images
- Generate Label Studio import file
"""

import argparse
import subprocess
from pathlib import Path
import shutil


def get_annotated_images(export_dir):
    """Get list of already annotated images from Label Studio export"""
    export_dir = Path(export_dir)
    img_dir = export_dir / 'images'

    if not img_dir.exists():
        return set()

    annotated = set()
    for img in img_dir.glob('*.*'):
        if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            # Extract original filename (Label Studio adds prefix like "020a77e7-")
            # Format: <uuid>-<original_name>
            parts = img.name.split('-', 1)
            if len(parts) == 2:
                original_name = parts[1]
                annotated.add(original_name)
            else:
                annotated.add(img.name)

    return annotated


def get_all_images(image_dir):
    """Get all images in directory"""
    image_dir = Path(image_dir)
    extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']

    all_images = set()
    for ext in extensions:
        for img in image_dir.glob(f'*{ext}'):
            all_images.add(img.name)

    return all_images


def run_round(round_num, export_dir, image_dir, epochs=100, batch=16, model_size='n'):
    """
    Run one round of active learning

    Args:
        round_num: Round number
        export_dir: Label Studio export directory
        image_dir: Directory with all images
        epochs: Training epochs
        batch: Batch size
        model_size: YOLO model size (n, s, m, l, x)
    """

    print(f"\n{'='*70}")
    print(f"🔄 ACTIVE LEARNING ROUND {round_num}")
    print(f"{'='*70}\n")

    # Step 1: Check current status
    annotated = get_annotated_images(export_dir)
    all_images = get_all_images(image_dir)
    unlabeled = all_images - annotated

    print(f"📊 Current Status:")
    print(f"  - Total images: {len(all_images)}")
    print(f"  - Annotated: {len(annotated)}")
    print(f"  - Remaining: {len(unlabeled)}")

    if len(annotated) == 0:
        print(f"\n❌ No annotated images found in {export_dir}")
        print(f"   Please annotate some images first!")
        return False

    if len(unlabeled) == 0:
        print(f"\n✅ All images annotated! Done!")
        return False

    # Step 2: Prepare dataset
    print(f"\n📦 Preparing dataset from Label Studio export...")

    data_dir = f'data_round{round_num}'

    subprocess.run([
        'python', 'scripts/prepare_active_learning.py',
        '--export', export_dir,
        '--output', data_dir
    ], check=True)

    # Step 3: Train model
    print(f"\n🎓 Training YOLOv8 on {len(annotated)} annotated images...")
    print(f"   Model: yolov8{model_size}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch: {batch}\n")

    model_name = f'coffee_round{round_num}'

    subprocess.run([
        'python', 'scripts/train_yolo.py',
        '--data', f'./{data_dir}',
        '--model', model_size,
        '--epochs', str(epochs),
        '--batch', str(batch),
        '--name', model_name
    ], check=True)

    model_path = f'runs/detect/{model_name}/weights/best.pt'

    if not Path(model_path).exists():
        print(f"\n❌ Model training failed! Model not found at {model_path}")
        return False

    # Step 4: Predict on unlabeled images
    print(f"\n🔮 Running predictions on {len(unlabeled)} unlabeled images...")

    output_dir = f'predictions_round{round_num}'

    subprocess.run([
        'python', 'scripts/predict_and_import.py',
        '--model', model_path,
        '--images', image_dir,
        '--conf', '0.20',  # Lower confidence to catch more beans
        '--output', output_dir
    ], check=True)

    # Step 5: Success message
    print(f"\n{'='*70}")
    print(f"✅ Round {round_num} Complete!")
    print(f"{'='*70}")
    print(f"\n📝 Next steps:")
    print(f"  1. Import {output_dir}/label_studio_tasks.json to Label Studio:")
    print(f"     - Open Label Studio (http://localhost:8080)")
    print(f"     - Go to your project")
    print(f"     - Click 'Import'")
    print(f"     - Upload {output_dir}/label_studio_tasks.json")
    print(f"")
    print(f"  2. Review and correct predictions:")
    print(f"     - Model predictions shown as light-colored boxes")
    print(f"     - Accept good predictions ✓")
    print(f"     - Delete false positives ✗")
    print(f"     - Add missed beans")
    print(f"     - Correct misclassified labels")
    print(f"")
    print(f"  3. Annotate more images (aim for 20-50 more)")
    print(f"")
    print(f"  4. Export all annotations:")
    print(f"     - Label Studio → Export → YOLO")
    print(f"     - Save to new folder")
    print(f"")
    print(f"  5. Run Round {round_num + 1}:")
    print(f"     python scripts/run_active_round.py \\")
    print(f"       --round {round_num + 1} \\")
    print(f"       --export <new_export_folder> \\")
    print(f"       --images \"Hình cà phê\"")
    print(f"")
    print(f"💡 Model will improve with each round!")
    print(f"{'='*70}\n")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Run one round of Active Learning'
    )

    parser.add_argument('--round', type=int, default=1,
                       help='Round number (default: 1)')
    parser.add_argument('--export', type=str, required=True,
                       help='Label Studio export directory (project-1-at-2025...)')
    parser.add_argument('--images', type=str, required=True,
                       help='Directory containing all images')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Training epochs (default: 100)')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--model', type=str, default='n',
                       help='YOLO model size: n, s, m, l, x (default: n)')

    args = parser.parse_args()

    run_round(
        round_num=args.round,
        export_dir=args.export,
        image_dir=args.images,
        epochs=args.epochs,
        batch=args.batch,
        model_size=args.model
    )


if __name__ == '__main__':
    main()
