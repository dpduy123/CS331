"""
Debug script to check if model is predicting correctly
"""

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ssvit_yolov11 import create_ssvit_yolov11n
from train import CoffeeDataset, decode_predictions

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    weights_path = Path(__file__).parent / 'runs' / 'ssvit_yolov11n' / 'best.pt'
    print(f"\nLoading model from {weights_path}...")

    model = create_ssvit_yolov11n(num_classes=5)
    checkpoint = torch.load(weights_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"  Best mAP@0.5: {checkpoint.get('best_map50', 'unknown')}")
        if 'metrics' in checkpoint:
            print(f"  Metrics: {checkpoint['metrics']}")
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    # Load one image
    data_path = Path(__file__).parent.parent / 'kaggle_dataset_label'
    print(f"\nLoading dataset from {data_path}...")

    dataset = CoffeeDataset(data_path, split='val', img_size=640, augment=False)
    print(f"  Found {len(dataset)} images")

    if len(dataset) == 0:
        print("ERROR: No images found!")
        return

    # Get one sample
    img, targets, img_path = dataset[0]
    print(f"\nSample image: {img_path}")
    print(f"  Image shape: {img.shape}")
    print(f"  Targets: {targets}")

    # Run inference
    img_tensor = img.unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(img_tensor)

    print(f"\nRaw predictions:")
    for i, pred in enumerate(predictions):
        print(f"  P{i+3}: shape={pred.shape}, min={pred.min():.4f}, max={pred.max():.4f}")

        # Check objectness scores
        obj_scores = torch.sigmoid(pred[0, 4, :, :])
        print(f"       Objectness: min={obj_scores.min():.4f}, max={obj_scores.max():.4f}, mean={obj_scores.mean():.4f}")

        # Check class scores
        cls_scores = torch.sigmoid(pred[0, 5:, :, :])
        print(f"       Class scores: min={cls_scores.min():.4f}, max={cls_scores.max():.4f}")

    # Try different confidence thresholds
    print("\nDetections at different thresholds:")
    for conf in [0.001, 0.01, 0.05, 0.1, 0.25, 0.5]:
        detections = decode_predictions(predictions, conf_threshold=conf)
        num_dets = len(detections[0]) if detections[0] is not None else 0
        print(f"  conf={conf}: {num_dets} detections")

        if detections[0] is not None and len(detections[0]) > 0:
            print(f"    Top 5 detections:")
            for det in detections[0][:5]:
                x1, y1, x2, y2, conf_score, cls = det.cpu().numpy()
                print(f"      class={int(cls)}, conf={conf_score:.4f}, box=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")


if __name__ == "__main__":
    main()
