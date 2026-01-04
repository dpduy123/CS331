"""
YOLOv8 Training Script for Docker
Usage: python train.py [--model yolov8n] [--epochs 100] [--batch 4] [--device cpu]
"""

import argparse
import os
import yaml
import torch
from ultralytics import YOLO


def get_device():
    """Detect best available device"""
    if torch.cuda.is_available():
        return 0, torch.cuda.get_device_name(0)
    else:
        return 'cpu', 'CPU'


def setup_data_yaml(data_path, output_path='/app/data.yaml'):
    """Create data.yaml for training"""
    data_config = {
        'path': data_path,
        'train': 'images/train',
        'val': 'images/val',
        'nc': 5,
        'names': ['barely-riped', 'over-riped', 'riped', 'semi-riped', 'unriped']
    }

    with open(output_path, 'w') as f:
        yaml.dump(data_config, f, sort_keys=False)

    print(f"Created {output_path}")
    return output_path


def train(args):
    """Run training"""
    print("=" * 60)
    print("COFFEE BEAN YOLOV8 TRAINING")
    print("=" * 60)

    # Detect device
    device, device_name = get_device()
    if args.device:
        device = args.device
        device_name = args.device

    print(f"Device: {device_name}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Image size: {args.imgsz}")
    print("=" * 60)

    # Setup data.yaml
    data_yaml = setup_data_yaml(args.data)

    # Verify dataset
    train_path = os.path.join(args.data, 'images/train')
    val_path = os.path.join(args.data, 'images/val')

    if os.path.exists(train_path):
        train_count = len([f for f in os.listdir(train_path) if f.endswith(('.jpg', '.png'))])
        print(f"Train images: {train_count}")
    else:
        print(f"ERROR: Train path not found: {train_path}")
        return

    if os.path.exists(val_path):
        val_count = len([f for f in os.listdir(val_path) if f.endswith(('.jpg', '.png'))])
        print(f"Val images: {val_count}")
    else:
        print(f"ERROR: Val path not found: {val_path}")
        return

    print("=" * 60)

    # Load model
    model_file = f"{args.model}.pt" if not args.model.endswith('.pt') else args.model
    model = YOLO(model_file)

    # Train config
    train_config = {
        'data': data_yaml,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'patience': args.patience,
        'project': args.project,
        'name': args.name,
        'device': device,
        'workers': 0,
        'exist_ok': True,
    }

    # Disable AMP for CPU or if specified
    if device == 'cpu' or args.no_amp:
        train_config['amp'] = False

    # Train
    print("\nStarting training...")
    results = model.train(**train_config)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)

    # Validate
    print("\nRunning validation...")
    metrics = model.val()

    print(f"\nResults:")
    print(f"  mAP@0.5: {metrics.box.map50:.3f}")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.3f}")

    # Save path
    best_path = os.path.join(args.project, args.name, 'weights/best.pt')
    print(f"\nBest model saved to: {best_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Coffee Bean Training')

    parser.add_argument('--model', type=str, default='yolov8n',
                        help='Model to use (yolov8n, yolov8s, yolov8m, etc.)')
    parser.add_argument('--data', type=str, default='/app/data',
                        help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cpu, 0, 1, etc.)')
    parser.add_argument('--project', type=str, default='/app/runs',
                        help='Project directory')
    parser.add_argument('--name', type=str, default='coffee_train',
                        help='Run name')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable automatic mixed precision')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
