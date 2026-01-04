"""
Active Learning Workflow for Coffee Bean Annotation
Iteratively train model and pre-annotate unlabeled images
"""

import argparse
import json
from pathlib import Path
import subprocess
import shutil


def get_annotated_images(export_file):
    """Get list of already annotated images"""
    with open(export_file, 'r') as f:
        data = json.load(f)

    annotated = set()
    for task in data:
        if 'annotations' in task and task['annotations']:
            # Get image name from task
            if 'file_upload' in task['data']:
                img_name = task['data']['file_upload'].split('/')[-1]
            elif 'image' in task['data']:
                img_name = task['data']['image'].split('=')[-1]
            else:
                continue
            annotated.add(img_name)

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


def select_next_batch(unlabeled_images, batch_size=50, strategy='random'):
    """
    Select next batch of images to annotate

    Strategies:
    - random: Random selection
    - diverse: Try to select diverse images (placeholder)
    - uncertain: Based on model confidence (requires predictions)
    """
    import random

    unlabeled_list = list(unlabeled_images)

    if strategy == 'random':
        random.shuffle(unlabeled_list)
        return unlabeled_list[:batch_size]

    # TODO: Implement uncertainty-based sampling
    return unlabeled_list[:batch_size]


def run_active_learning_round(
    round_num,
    export_file,
    image_dir,
    batch_size=50,
    epochs=50
):
    """Run one round of active learning"""

    print(f"\n{'='*60}")
    print(f"🔄 ACTIVE LEARNING ROUND {round_num}")
    print(f"{'='*60}\n")

    # Step 1: Check current status
    annotated = get_annotated_images(export_file)
    all_images = get_all_images(image_dir)
    unlabeled = all_images - annotated

    print(f"📊 Current Status:")
    print(f"  - Total images: {len(all_images)}")
    print(f"  - Annotated: {len(annotated)}")
    print(f"  - Remaining: {len(unlabeled)}")

    if len(unlabeled) == 0:
        print(f"\n✅ All images annotated! Done!")
        return False

    # Step 2: Train model on current annotations
    print(f"\n🎓 Training model on {len(annotated)} annotated images...")

    model_name = f'active_round{round_num}'

    # Convert to YOLO
    subprocess.run([
        'python', 'scripts/convert_ls_to_yolo.py',
        '--export', export_file,
        '--images', image_dir,
        '--output', f'data_round{round_num}'
    ])

    # Train
    subprocess.run([
        'python', 'scripts/train_yolo.py',
        '--data', f'./data_round{round_num}',
        '--model', 'n',
        '--epochs', str(epochs),
        '--batch', '16',
        '--name', model_name
    ])

    model_path = f'runs/detect/{model_name}/weights/best.pt'

    # Step 3: Select next batch
    next_batch = select_next_batch(unlabeled, batch_size)

    print(f"\n🎯 Selected {len(next_batch)} images for next batch")

    # Create temp directory with only next batch
    temp_dir = Path(f'temp_batch_round{round_num}')
    temp_dir.mkdir(exist_ok=True)

    for img_name in next_batch:
        src = Path(image_dir) / img_name
        dst = temp_dir / img_name
        shutil.copy2(src, dst)

    # Step 4: Predict on next batch
    print(f"\n🔮 Predicting on next batch...")

    subprocess.run([
        'python', 'scripts/predict_and_import.py',
        '--model', model_path,
        '--images', str(temp_dir),
        '--conf', '0.15',
        '--output', f'predictions_round{round_num}'
    ])

    # Clean up temp dir
    shutil.rmtree(temp_dir)

    print(f"\n{'='*60}")
    print(f"✅ Round {round_num} Complete!")
    print(f"{'='*60}")
    print(f"\n📝 Next steps:")
    print(f"  1. Import predictions_round{round_num}/label_studio_tasks.json")
    print(f"     to Label Studio")
    print(f"  2. Review and correct {len(next_batch)} images")
    print(f"  3. Export all annotations")
    print(f"  4. Run round {round_num + 1}")
    print(f"\n💡 Tip: Model will get better each round!")
    print(f"{'='*60}\n")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Active Learning for Coffee Bean Annotation'
    )

    parser.add_argument('--export', type=str, required=True,
                        help='Label Studio export file (updated after each round)')
    parser.add_argument('--images', type=str, required=True,
                        help='Directory containing all images')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Images to annotate per round (default: 50)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs per round (default: 50)')
    parser.add_argument('--max-rounds', type=int, default=10,
                        help='Maximum rounds (default: 10)')
    parser.add_argument('--round', type=int, default=None,
                        help='Specific round to run (default: auto-detect)')

    args = parser.parse_args()

    # Auto-detect round if not specified
    if args.round is None:
        # Find highest round number
        existing_rounds = list(Path('.').glob('data_round*'))
        if existing_rounds:
            round_nums = [int(p.name.replace('data_round', ''))
                         for p in existing_rounds]
            start_round = max(round_nums) + 1
        else:
            start_round = 1
    else:
        start_round = args.round

    print(f"\n🚀 Starting Active Learning from Round {start_round}")

    # Run rounds
    for round_num in range(start_round, start_round + args.max_rounds):
        success = run_active_learning_round(
            round_num=round_num,
            export_file=args.export,
            image_dir=args.images,
            batch_size=args.batch_size,
            epochs=args.epochs
        )

        if not success:
            break

        # Wait for user to review before next round
        if round_num < start_round + args.max_rounds - 1:
            print(f"\n⏸️  Pausing before next round...")
            print(f"   Please review annotations and export updated JSON")
            input(f"   Press Enter when ready for Round {round_num + 1}...")


if __name__ == '__main__':
    main()
