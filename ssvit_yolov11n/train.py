"""
Training Script for SSViT-YOLOv11n
Coffee Bean Ripeness Detection

This script trains the SSViT-YOLOv11n model on coffee bean dataset.
Features:
- Full metrics (mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1)
- Confusion matrix and per-class AP
- Training curves (CSV + TensorBoard)
- Early stopping and learning rate scheduling
- Output format similar to Ultralytics

Usage:
    python train.py --data-path /path/to/dataset --epochs 100
"""

import os
import sys
import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import argparse
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from ssvit_yolov11 import SSViTYOLOv11n, create_ssvit_yolov11n
from loss import YOLOv8Loss, bbox_iou
from metrics import DetectionMetrics, ConfusionMatrix, EarlyStopping, xywh2xyxy
from logger import TrainingLogger


# Class names for coffee bean dataset
CLASS_NAMES = ['barely-riped', 'over-riped', 'riped', 'semi-riped', 'unriped']


class CoffeeDataset(Dataset):
    """
    Coffee Bean Detection Dataset

    Expects YOLO format:
    - images/train/*.jpg
    - labels/train/*.txt (class x_center y_center width height)
    """

    def __init__(self, data_path, split='train', img_size=640, augment=True):
        self.data_path = Path(data_path)
        self.split = split
        self.img_size = img_size
        self.augment = augment and split == 'train'

        # Find images
        self.img_dir = self.data_path / 'images' / split
        self.label_dir = self.data_path / 'labels' / split

        self.images = list(self.img_dir.glob('*.jpg')) + \
                      list(self.img_dir.glob('*.jpeg')) + \
                      list(self.img_dir.glob('*.png'))

        print(f"  Found {len(self.images)} {split} images in {self.img_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load labels
        label_path = self.label_dir / (img_path.stem + '.txt')
        labels = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls, x, y, w, h = map(float, parts[:5])
                        labels.append([cls, x, y, w, h])

        labels = np.array(labels) if labels else np.zeros((0, 5))

        # Resize image
        h0, w0 = img.shape[:2]
        r = self.img_size / max(h0, w0)
        if r != 1:
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)))

        # Pad to square
        h, w = img.shape[:2]
        dh, dw = self.img_size - h, self.img_size - w
        top, bottom = dh // 2, dh - dh // 2
        left, right = dw // 2, dw - dw // 2
        img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                  cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # Augmentation
        if self.augment:
            img, labels = self.apply_augmentation(img, labels)

        # Convert to tensor (make contiguous copy)
        img = np.ascontiguousarray(img.transpose(2, 0, 1))
        img = torch.from_numpy(img).float() / 255.0

        # Convert labels to tensor
        targets = torch.from_numpy(labels).float()

        return img, targets, str(img_path)

    def apply_augmentation(self, img, labels):
        """Apply data augmentation similar to Ultralytics"""
        # Random horizontal flip
        if np.random.random() < 0.5:
            img = img[:, ::-1, :].copy()
            if len(labels) > 0:
                labels[:, 1] = 1 - labels[:, 1]

        # Random vertical flip
        if np.random.random() < 0.5:
            img = img[::-1, :, :].copy()
            if len(labels) > 0:
                labels[:, 2] = 1 - labels[:, 2]

        # Random brightness/contrast
        if np.random.random() < 0.5:
            alpha = 0.8 + np.random.random() * 0.4
            beta = -20 + np.random.random() * 40
            img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)

        # HSV augmentation
        if np.random.random() < 0.5:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)

            # Hue shift
            hsv[:, :, 0] = (hsv[:, :, 0] + np.random.uniform(-10, 10)) % 180

            # Saturation scale
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * np.random.uniform(0.8, 1.2), 0, 255)

            # Value scale
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * np.random.uniform(0.8, 1.2), 0, 255)

            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # Random rotation (small angle)
        if np.random.random() < 0.3:
            angle = np.random.uniform(-10, 10)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderValue=(114, 114, 114))

        return img, labels


def collate_fn(batch):
    """Custom collate function for variable number of labels"""
    imgs, targets, paths = zip(*batch)

    # Stack images
    imgs = torch.stack(imgs, 0)

    # Process targets (add batch index)
    new_targets = []
    for i, t in enumerate(targets):
        if len(t) > 0:
            batch_idx = torch.full((len(t), 1), i)
            new_targets.append(torch.cat([batch_idx, t], dim=1))

    if new_targets:
        targets = torch.cat(new_targets, 0)
    else:
        targets = torch.zeros((0, 6))

    return imgs, targets, paths


def decode_predictions(predictions, conf_threshold=0.25, iou_threshold=0.45, img_size=640):
    """
    Decode model predictions to bounding boxes

    Args:
        predictions: tuple of (p3, p4, p5) outputs
        conf_threshold: Confidence threshold
        iou_threshold: NMS IoU threshold

    Returns:
        List of detections per image [x1, y1, x2, y2, conf, class]
    """
    strides = [8, 16, 32]
    all_detections = []

    batch_size = predictions[0].shape[0]

    for bi in range(batch_size):
        detections = []

        for si, pred in enumerate(predictions):
            stride = strides[si]
            _, C, H, W = pred.shape

            # [C, H, W] -> [H, W, C]
            p = pred[bi].permute(1, 2, 0)

            # Create grid
            yv, xv = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
            grid = torch.stack([xv, yv], dim=-1).float().to(pred.device)

            # Decode boxes
            xy = (torch.sigmoid(p[..., :2]) + grid) * stride
            wh = torch.exp(p[..., 2:4].clamp(max=10)) * stride

            # Objectness and class scores
            obj = torch.sigmoid(p[..., 4])
            cls = torch.sigmoid(p[..., 5:])

            # Combined confidence
            conf = obj.unsqueeze(-1) * cls

            # Flatten
            xy = xy.reshape(-1, 2)
            wh = wh.reshape(-1, 2)
            conf = conf.reshape(-1, conf.shape[-1])

            # Filter by confidence
            max_conf, max_cls = conf.max(dim=1)
            mask = max_conf > conf_threshold

            if mask.sum() > 0:
                xy = xy[mask]
                wh = wh[mask]
                max_conf = max_conf[mask]
                max_cls = max_cls[mask]

                # Convert to xyxy
                x1y1 = xy - wh / 2
                x2y2 = xy + wh / 2

                # Combine
                dets = torch.cat([x1y1, x2y2, max_conf.unsqueeze(1), max_cls.float().unsqueeze(1)], dim=1)
                detections.append(dets)

        if detections:
            detections = torch.cat(detections, dim=0)

            # Apply NMS using torchvision if available, otherwise simple NMS
            try:
                from torchvision.ops import batched_nms
                boxes = detections[:, :4]
                scores = detections[:, 4]
                classes = detections[:, 5]
                keep_indices = batched_nms(boxes, scores, classes.long(), iou_threshold)
                detections = detections[keep_indices]
            except ImportError:
                # Simple NMS fallback
                keep_indices = []
                order = detections[:, 4].argsort(descending=True)
                detections = detections[order]

                while len(detections) > 0:
                    keep_indices.append(0)
                    if len(detections) == 1:
                        break

                    # Calculate IoU with remaining boxes
                    current_box = detections[0:1, :4]
                    other_boxes = detections[1:, :4]

                    # Simple IoU calculation
                    x1 = torch.max(current_box[:, 0], other_boxes[:, 0])
                    y1 = torch.max(current_box[:, 1], other_boxes[:, 1])
                    x2 = torch.min(current_box[:, 2], other_boxes[:, 2])
                    y2 = torch.min(current_box[:, 3], other_boxes[:, 3])

                    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
                    area1 = (current_box[:, 2] - current_box[:, 0]) * (current_box[:, 3] - current_box[:, 1])
                    area2 = (other_boxes[:, 2] - other_boxes[:, 0]) * (other_boxes[:, 3] - other_boxes[:, 1])
                    union = area1 + area2 - inter
                    ious = inter / (union + 1e-7)

                    # Keep boxes with low IoU
                    mask = ious.squeeze() < iou_threshold
                    if mask.dim() == 0:
                        mask = mask.unsqueeze(0)
                    detections = detections[1:][mask]

                if len(keep_indices) == 0:
                    detections = None
        else:
            detections = None

        all_detections.append(detections)

    return all_detections


def train_one_epoch(model, dataloader, optimizer, loss_fn, device, epoch, logger):
    """Train for one epoch"""
    model.train()

    total_loss = {'box': 0, 'cls': 0, 'obj': 0, 'total': 0}
    num_batches = len(dataloader)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=True)

    for batch_idx, (imgs, targets, paths) in enumerate(pbar):
        imgs = imgs.to(device)
        targets = targets.to(device)

        # Forward pass
        predictions = model(imgs)

        # Compute loss
        loss, loss_dict = loss_fn(predictions, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        optimizer.step()

        # Accumulate losses
        for k in total_loss:
            total_loss[k] += loss_dict.get(k, 0)

        # Update progress bar
        pbar.set_postfix({
            'box': f"{loss_dict['box']:.4f}",
            'cls': f"{loss_dict['cls']:.4f}",
            'obj': f"{loss_dict['obj']:.4f}"
        })

    # Average losses
    for k in total_loss:
        total_loss[k] /= num_batches

    return total_loss


@torch.no_grad()
def validate(model, dataloader, loss_fn, metrics_calculator, device, conf_threshold=0.25):
    """Validate model and compute metrics"""
    model.eval()

    total_loss = {'box': 0, 'cls': 0, 'obj': 0, 'total': 0}
    num_batches = len(dataloader)

    metrics_calculator.reset()

    pbar = tqdm(dataloader, desc="Validating", leave=False)

    for imgs, targets, paths in pbar:
        imgs = imgs.to(device)
        targets = targets.to(device)

        # Forward pass
        predictions = model(imgs)

        # Compute loss
        loss, loss_dict = loss_fn(predictions, targets)

        for k in total_loss:
            total_loss[k] += loss_dict.get(k, 0)

        # Decode predictions for metrics
        detections = decode_predictions(predictions, conf_threshold=conf_threshold)

        # Process metrics
        for bi, dets in enumerate(detections):
            # Get targets for this batch item
            batch_targets = targets[targets[:, 0] == bi]

            if dets is not None:
                dets = dets.cpu()
            if len(batch_targets) > 0:
                batch_targets = batch_targets.cpu()

            metrics_calculator.process_batch(dets, batch_targets)

    # Average losses
    for k in total_loss:
        total_loss[k] /= max(num_batches, 1)

    # Compute final metrics
    metrics = metrics_calculator.compute()

    return total_loss, metrics


def main(args):
    """Main training function"""

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f'train_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = output_dir / 'weights'
    weights_dir.mkdir(exist_ok=True)

    # Initialize logger
    logger = TrainingLogger(
        save_dir=output_dir,
        class_names=CLASS_NAMES,
        use_tensorboard=True
    )

    # Create model
    model = create_ssvit_yolov11n(num_classes=args.num_classes)
    model = model.to(device)

    # Print training info
    logger.print_training_start(
        model_name="SSViT-YOLOv11n",
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_params=model.count_parameters()
    )

    # Create datasets
    print("\nLoading datasets...")
    train_dataset = CoffeeDataset(args.data_path, split='train', img_size=args.img_size, augment=True)
    val_dataset = CoffeeDataset(args.data_path, split='val', img_size=args.img_size, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val: {len(val_dataset)} images")

    # Loss function
    loss_fn = YOLOv8Loss(num_classes=args.num_classes)

    # Metrics calculator
    metrics_calculator = DetectionMetrics(num_classes=args.num_classes, class_names=CLASS_NAMES)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0005)

    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        # Warmup for first 3 epochs
        if epoch < 3:
            return (epoch + 1) / 3
        # Cosine decay
        return 0.5 * (1 + np.cos(np.pi * (epoch - 3) / (args.epochs - 3)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience)

    # Training loop
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)

    start_time = time.time()
    best_map50 = 0.0

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch, logger
        )

        # Validate
        val_loss, metrics = validate(
            model, val_loader, loss_fn, metrics_calculator, device
        )

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Log epoch
        logger.log_epoch(epoch, train_loss, val_loss, metrics, current_lr)

        # Update scheduler
        scheduler.step()

        # Save best model
        if metrics['map50'] > best_map50:
            best_map50 = metrics['map50']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'best_map50': best_map50,
            }, weights_dir / 'best.pt')
            print(f"  ✓ Saved best model (mAP@0.5: {best_map50:.4f})")

        # Save last model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
        }, weights_dir / 'last.pt')

        # Check early stopping
        if early_stopping(metrics['map50']):
            print(f"\n  Early stopping triggered at epoch {epoch}")
            break

    # Training complete
    total_time = time.time() - start_time
    logger.print_training_end(epoch, total_time)

    # Save confusion matrix
    confusion_matrix = ConfusionMatrix(num_classes=args.num_classes)
    # Run final validation for confusion matrix
    model.eval()
    with torch.no_grad():
        for imgs, targets, paths in val_loader:
            imgs = imgs.to(device)
            predictions = model(imgs)
            detections = decode_predictions(predictions)

            for bi, dets in enumerate(detections):
                batch_targets = targets[targets[:, 0] == bi]
                if dets is not None:
                    confusion_matrix.process_batch(dets.cpu().numpy(), batch_targets[:, 1:].numpy())

    confusion_matrix.plot(CLASS_NAMES, save_path=output_dir / 'confusion_matrix.png')
    print(f"\n  Confusion matrix saved to {output_dir / 'confusion_matrix.png'}")

    logger.close()

    return best_map50


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SSViT-YOLOv11n")

    # Data arguments
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to dataset (with images/ and labels/ folders)')
    parser.add_argument('--output-dir', type=str, default='./runs/ssvit',
                        help='Output directory for models and logs')

    # Model arguments
    parser.add_argument('--num-classes', type=int, default=5,
                        help='Number of classes')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Input image size')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience')

    args = parser.parse_args()

    main(args)
