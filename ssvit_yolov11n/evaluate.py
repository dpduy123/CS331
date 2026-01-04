"""
Evaluation Script for SSViT-YOLOv11n
Coffee Bean Ripeness Detection

This script evaluates a trained SSViT-YOLOv11n model on test/val dataset.
Outputs:
- mAP@0.5, mAP@0.5:0.95
- Precision, Recall, F1
- Per-class AP
- Confusion Matrix
- Sample predictions visualization

Usage:
    python evaluate.py --weights runs/ssvit_yolov11n/best.pt --data-path dataset --split val
"""

import os
import sys
import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import argparse
import json
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from ssvit_yolov11 import SSViTYOLOv11n, create_ssvit_yolov11n
from metrics import DetectionMetrics, ConfusionMatrix, xywh2xyxy
from loss import bbox_iou
from train import CoffeeDataset, collate_fn, decode_predictions

# Class names for coffee bean dataset
CLASS_NAMES = ['barely-riped', 'over-riped', 'riped', 'semi-riped', 'unriped']

# Colors for visualization (BGR format)
COLORS = {
    'barely-riped': (0, 165, 255),   # Orange
    'over-riped': (0, 0, 139),       # Dark Red
    'riped': (0, 128, 0),            # Green
    'semi-riped': (0, 255, 255),     # Yellow
    'unriped': (0, 255, 0),          # Lime
}


def load_model(weights_path, num_classes=5, device='cuda'):
    """Load trained model from checkpoint"""
    print(f"\nLoading model from {weights_path}...")

    # Create model
    model = create_ssvit_yolov11n(num_classes=num_classes)

    # Load weights
    checkpoint = torch.load(weights_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'best_map50' in checkpoint:
            print(f"  Best mAP@0.5: {checkpoint['best_map50']:.4f}")
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model


@torch.no_grad()
def evaluate(model, dataloader, device, conf_threshold=0.25, iou_threshold=0.45):
    """Run evaluation and compute all metrics"""

    metrics_calculator = DetectionMetrics(num_classes=5, class_names=CLASS_NAMES)
    confusion_matrix = ConfusionMatrix(num_classes=5, conf_threshold=conf_threshold)

    all_predictions = []
    all_targets = []

    print("\nRunning evaluation...")
    pbar = tqdm(dataloader, desc="Evaluating")

    for imgs, targets, paths in pbar:
        imgs = imgs.to(device)

        # Forward pass
        predictions = model(imgs)

        # Decode predictions
        detections = decode_predictions(
            predictions,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )

        # Process each image in batch
        for bi, dets in enumerate(detections):
            batch_targets = targets[targets[:, 0] == bi]

            if dets is not None:
                dets_np = dets.cpu().numpy()
                all_predictions.append(dets_np)
            else:
                all_predictions.append(np.array([]))

            if len(batch_targets) > 0:
                all_targets.append(batch_targets[:, 1:].cpu().numpy())
            else:
                all_targets.append(np.array([]))

            # Update metrics
            if dets is not None:
                dets_cpu = dets.cpu()
            else:
                dets_cpu = None

            metrics_calculator.process_batch(dets_cpu, batch_targets.cpu())

            # Update confusion matrix
            if dets is not None:
                confusion_matrix.process_batch(dets.cpu().numpy(), batch_targets[:, 1:].cpu().numpy())

    # Compute final metrics
    results = metrics_calculator.compute()

    return results, confusion_matrix, all_predictions, all_targets


def print_results(results, class_names):
    """Print evaluation results in a nice format"""
    print("\n" + "=" * 80)
    print("                           EVALUATION RESULTS")
    print("=" * 80)

    # Overall metrics
    print("\n📊 Overall Metrics:")
    print("-" * 40)
    print(f"  {'Metric':<20} {'Value':>10}")
    print("-" * 40)
    print(f"  {'Precision':<20} {results['precision']:>10.4f}")
    print(f"  {'Recall':<20} {results['recall']:>10.4f}")
    print(f"  {'F1-Score':<20} {results['f1']:>10.4f}")
    print(f"  {'mAP@0.5':<20} {results['map50']:>10.4f}")
    print(f"  {'mAP@0.5:0.95':<20} {results['map50_95']:>10.4f}")

    # Per-class metrics
    print("\n📈 Per-Class AP@0.5:")
    print("-" * 50)
    print(f"  {'Class':<20} {'AP@0.5':>10} {'Precision':>10} {'Recall':>10}")
    print("-" * 50)

    ap_per_class = results.get('ap_per_class', np.zeros(len(class_names)))
    p_per_class = results.get('p_per_class', np.zeros(len(class_names)))
    r_per_class = results.get('r_per_class', np.zeros(len(class_names)))

    for i, name in enumerate(class_names):
        ap = ap_per_class[i] if i < len(ap_per_class) else 0
        p = p_per_class[i] if i < len(p_per_class) else 0
        r = r_per_class[i] if i < len(r_per_class) else 0
        print(f"  {name:<20} {ap:>10.4f} {p:>10.4f} {r:>10.4f}")

    print("-" * 50)
    print(f"  {'Mean':<20} {np.mean(ap_per_class):>10.4f} {np.mean(p_per_class):>10.4f} {np.mean(r_per_class):>10.4f}")
    print("=" * 80)


def visualize_predictions(model, dataset, device, output_dir, num_samples=10, conf_threshold=0.25):
    """Visualize model predictions on sample images"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving {num_samples} sample predictions to {output_dir}...")

    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    for idx in tqdm(indices, desc="Visualizing"):
        img, targets, img_path = dataset[idx]

        # Run inference
        img_tensor = img.unsqueeze(0).to(device)
        predictions = model(img_tensor)
        detections = decode_predictions(predictions, conf_threshold=conf_threshold)

        # Convert image to numpy for visualization
        img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Draw ground truth (green boxes)
        if len(targets) > 0:
            for t in targets:
                cls_id = int(t[0])
                x, y, w, h = t[1:5]

                # Convert to pixel coordinates
                x1 = int((x - w/2) * 640)
                y1 = int((y - h/2) * 640)
                x2 = int((x + w/2) * 640)
                y2 = int((y + h/2) * 640)

                cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_np, f"GT: {CLASS_NAMES[cls_id]}", (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Draw predictions (colored by class)
        if detections[0] is not None:
            for det in detections[0]:
                x1, y1, x2, y2, conf, cls_id = det.cpu().numpy()
                cls_id = int(cls_id)

                color = COLORS.get(CLASS_NAMES[cls_id], (255, 255, 255))

                cv2.rectangle(img_np, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"{CLASS_NAMES[cls_id]}: {conf:.2f}"
                cv2.putText(img_np, label, (int(x1), int(y2)+15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Save image
        save_path = output_dir / f"pred_{idx}.jpg"
        cv2.imwrite(str(save_path), img_np)

    print(f"  Saved {num_samples} visualizations to {output_dir}")


def save_results(results, confusion_matrix, output_dir, class_names):
    """Save evaluation results to files"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics to JSON
    metrics_dict = {
        'precision': float(results['precision']),
        'recall': float(results['recall']),
        'f1': float(results['f1']),
        'map50': float(results['map50']),
        'map50_95': float(results['map50_95']),
        'per_class': {}
    }

    ap_per_class = results.get('ap_per_class', np.zeros(len(class_names)))
    p_per_class = results.get('p_per_class', np.zeros(len(class_names)))
    r_per_class = results.get('r_per_class', np.zeros(len(class_names)))

    for i, name in enumerate(class_names):
        metrics_dict['per_class'][name] = {
            'ap50': float(ap_per_class[i]) if i < len(ap_per_class) else 0,
            'precision': float(p_per_class[i]) if i < len(p_per_class) else 0,
            'recall': float(r_per_class[i]) if i < len(r_per_class) else 0,
        }

    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics_dict, f, indent=2)

    print(f"\n  Metrics saved to {output_dir / 'metrics.json'}")

    # Save confusion matrix
    confusion_matrix.plot(class_names, save_path=output_dir / 'confusion_matrix.png')
    print(f"  Confusion matrix saved to {output_dir / 'confusion_matrix.png'}")

    # Save confusion matrix as CSV
    np.savetxt(output_dir / 'confusion_matrix.csv', confusion_matrix.get_matrix(),
               delimiter=',', fmt='%.0f')
    print(f"  Confusion matrix CSV saved to {output_dir / 'confusion_matrix.csv'}")


def main(args):
    """Main evaluation function"""

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load model
    model = load_model(args.weights, num_classes=args.num_classes, device=device)

    # Create dataset
    print(f"\nLoading {args.split} dataset from {args.data_path}...")
    dataset = CoffeeDataset(
        args.data_path,
        split=args.split,
        img_size=args.img_size,
        augment=False
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Run evaluation
    results, confusion_matrix, predictions, targets = evaluate(
        model, dataloader, device,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold
    )

    # Print results
    print_results(results, CLASS_NAMES)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f'eval_{timestamp}'

    # Save results
    save_results(results, confusion_matrix, output_dir, CLASS_NAMES)

    # Visualize predictions
    if args.visualize:
        visualize_predictions(
            model, dataset, device,
            output_dir / 'visualizations',
            num_samples=args.num_vis,
            conf_threshold=args.conf_threshold
        )

    print(f"\n✅ Evaluation complete! Results saved to {output_dir}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SSViT-YOLOv11n")

    # Required arguments
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to model weights (best.pt or last.pt)')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to dataset')

    # Optional arguments
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--output-dir', type=str, default='./runs/eval',
                        help='Output directory for results')
    parser.add_argument('--num-classes', type=int, default=5,
                        help='Number of classes')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Input image size')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--conf-threshold', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--iou-threshold', type=float, default=0.45,
                        help='NMS IoU threshold')
    parser.add_argument('--visualize', action='store_true',
                        help='Save visualization of predictions')
    parser.add_argument('--num-vis', type=int, default=20,
                        help='Number of images to visualize')

    args = parser.parse_args()

    main(args)
