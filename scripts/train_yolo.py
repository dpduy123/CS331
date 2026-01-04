"""
YOLOv8 Training Script for Coffee Bean Detection
Trains a model on annotated data exported from Label Studio
"""

import os
import yaml
import argparse
from pathlib import Path
from ultralytics import YOLO
import torch

# Class mapping - matches your Label Studio labels
CLASSES = {
    0: "riped",
    1: "unriped",
    2: "semi-riped",
    3: "over-riped",
    4: "barely-riped"
}


def create_dataset_yaml(data_dir, output_path='coffee_beans.yaml'):
    """
    Create YOLO dataset configuration file

    Args:
        data_dir: Path to dataset directory
        output_path: Output path for YAML file
    """
    data_dir = Path(data_dir).resolve()

    config = {
        'path': str(data_dir),
        'train': 'train/images',
        'val': 'val/images',
        'nc': len(CLASSES),
        'names': CLASSES
    }

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"✓ Dataset config created: {output_path}")
    return output_path


def train_model(
    data_yaml,
    model_size='n',
    epochs=100,
    imgsz=640,
    batch=16,
    name='coffee_bean_detection',
    pretrained=True,
    device=None
):
    """
    Train YOLOv8 model

    Args:
        data_yaml: Path to dataset YAML file
        model_size: Model size (n, s, m, l, x)
        epochs: Number of training epochs
        imgsz: Input image size
        batch: Batch size
        name: Experiment name
        pretrained: Use pretrained weights
        device: Device to train on (None for auto-select)
    """
    # Auto-select device
    if device is None:
        device = 'mps' if torch.backends.mps.is_available() else \
                 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n{'='*60}")
    print(f"🚀 Starting YOLOv8 Training")
    print(f"{'='*60}")
    print(f"Model: YOLOv8{model_size}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Image Size: {imgsz}")
    print(f"Batch Size: {batch}")
    print(f"Dataset: {data_yaml}")
    print(f"{'='*60}\n")

    # Initialize model
    model_name = f'yolov8{model_size}.pt' if pretrained else f'yolov8{model_size}.yaml'
    model = YOLO(model_name)

    # Train
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name=name,
        device=device,
        patience=20,  # Early stopping patience
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        val=True,
        plots=True,
        verbose=True,
        # Data augmentation
        hsv_h=0.015,  # HSV-Hue augmentation
        hsv_s=0.7,    # HSV-Saturation augmentation
        hsv_v=0.4,    # HSV-Value augmentation
        degrees=10,   # Rotation
        translate=0.1,  # Translation
        scale=0.5,    # Scale
        flipud=0.5,   # Vertical flip probability
        fliplr=0.5,   # Horizontal flip probability
        mosaic=1.0,   # Mosaic augmentation
        mixup=0.1,    # Mixup augmentation
    )

    print(f"\n{'='*60}")
    print(f"✓ Training Complete!")
    print(f"{'='*60}")
    print(f"Best model saved to: runs/detect/{name}/weights/best.pt")
    print(f"Results saved to: runs/detect/{name}")
    print(f"{'='*60}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 for coffee bean detection')

    parser.add_argument('--data', type=str, default='./data',
                        help='Path to dataset directory')
    parser.add_argument('--model', type=str, default='n',
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='Model size (n=nano, s=small, m=medium, l=large, x=xlarge)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Input image size')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--name', type=str, default='coffee_bean_detection',
                        help='Experiment name')
    parser.add_argument('--no-pretrained', action='store_true',
                        help='Train from scratch (no pretrained weights)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to train on (cpu, cuda, mps)')

    args = parser.parse_args()

    # Create dataset YAML
    yaml_path = create_dataset_yaml(args.data)

    # Train model
    train_model(
        data_yaml=yaml_path,
        model_size=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        pretrained=not args.no_pretrained,
        device=args.device
    )


if __name__ == '__main__':
    main()
