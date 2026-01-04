"""
Training Logger for SSViT-YOLOv11n
Supports CSV logging, TensorBoard, and console output similar to Ultralytics
"""

import os
import csv
import json
from pathlib import Path
from datetime import datetime
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("TensorBoard not available. Install with: pip install tensorboard")


class TrainingLogger:
    """
    Comprehensive training logger with multiple output formats
    """

    def __init__(self, save_dir, class_names=None, use_tensorboard=True):
        """
        Args:
            save_dir: Directory to save logs
            class_names: List of class names
            use_tensorboard: Whether to use TensorBoard
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.class_names = class_names or []
        self.num_classes = len(self.class_names)

        # CSV logger
        self.csv_path = self.save_dir / 'results.csv'
        self.csv_file = None
        self.csv_writer = None
        self._init_csv()

        # TensorBoard
        self.tb_writer = None
        if use_tensorboard and TENSORBOARD_AVAILABLE:
            self.tb_writer = SummaryWriter(log_dir=str(self.save_dir / 'tensorboard'))

        # Training info
        self.training_info = {
            'start_time': datetime.now().isoformat(),
            'class_names': self.class_names,
        }

        # Best metrics tracking
        self.best_map50 = 0.0
        self.best_epoch = 0

    def _init_csv(self):
        """Initialize CSV file with headers"""
        headers = [
            'epoch',
            'train/box_loss',
            'train/cls_loss',
            'train/obj_loss',
            'train/total_loss',
            'val/box_loss',
            'val/cls_loss',
            'val/obj_loss',
            'val/total_loss',
            'metrics/precision',
            'metrics/recall',
            'metrics/mAP50',
            'metrics/mAP50-95',
            'metrics/f1',
            'lr'
        ]

        # Add per-class AP headers
        for name in self.class_names:
            headers.append(f'metrics/AP50_{name}')

        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=headers)
        self.csv_writer.writeheader()
        self.csv_file.flush()

    def log_epoch(self, epoch, train_loss, val_loss, metrics, lr):
        """
        Log epoch results

        Args:
            epoch: Current epoch number
            train_loss: Dict with train losses
            val_loss: Dict with val losses (or None)
            metrics: Dict with evaluation metrics
            lr: Current learning rate
        """
        # Console output (Ultralytics style)
        self._print_epoch(epoch, train_loss, val_loss, metrics, lr)

        # CSV logging
        row = {
            'epoch': epoch,
            'train/box_loss': train_loss.get('box', 0),
            'train/cls_loss': train_loss.get('cls', 0),
            'train/obj_loss': train_loss.get('obj', 0),
            'train/total_loss': train_loss.get('total', 0),
            'val/box_loss': val_loss.get('box', 0) if val_loss else 0,
            'val/cls_loss': val_loss.get('cls', 0) if val_loss else 0,
            'val/obj_loss': val_loss.get('obj', 0) if val_loss else 0,
            'val/total_loss': val_loss.get('total', 0) if val_loss else 0,
            'metrics/precision': metrics.get('precision', 0),
            'metrics/recall': metrics.get('recall', 0),
            'metrics/mAP50': metrics.get('map50', 0),
            'metrics/mAP50-95': metrics.get('map50_95', 0),
            'metrics/f1': metrics.get('f1', 0),
            'lr': lr
        }

        # Add per-class AP
        ap_per_class = metrics.get('ap_per_class', np.zeros(self.num_classes))
        for i, name in enumerate(self.class_names):
            row[f'metrics/AP50_{name}'] = ap_per_class[i] if i < len(ap_per_class) else 0

        self.csv_writer.writerow(row)
        self.csv_file.flush()

        # TensorBoard logging
        if self.tb_writer:
            self.tb_writer.add_scalar('train/box_loss', train_loss.get('box', 0), epoch)
            self.tb_writer.add_scalar('train/cls_loss', train_loss.get('cls', 0), epoch)
            self.tb_writer.add_scalar('train/obj_loss', train_loss.get('obj', 0), epoch)
            self.tb_writer.add_scalar('train/total_loss', train_loss.get('total', 0), epoch)

            if val_loss:
                self.tb_writer.add_scalar('val/box_loss', val_loss.get('box', 0), epoch)
                self.tb_writer.add_scalar('val/cls_loss', val_loss.get('cls', 0), epoch)
                self.tb_writer.add_scalar('val/obj_loss', val_loss.get('obj', 0), epoch)
                self.tb_writer.add_scalar('val/total_loss', val_loss.get('total', 0), epoch)

            self.tb_writer.add_scalar('metrics/precision', metrics.get('precision', 0), epoch)
            self.tb_writer.add_scalar('metrics/recall', metrics.get('recall', 0), epoch)
            self.tb_writer.add_scalar('metrics/mAP50', metrics.get('map50', 0), epoch)
            self.tb_writer.add_scalar('metrics/mAP50-95', metrics.get('map50_95', 0), epoch)
            self.tb_writer.add_scalar('metrics/f1', metrics.get('f1', 0), epoch)
            self.tb_writer.add_scalar('lr', lr, epoch)

        # Track best
        if metrics.get('map50', 0) > self.best_map50:
            self.best_map50 = metrics.get('map50', 0)
            self.best_epoch = epoch

    def _print_epoch(self, epoch, train_loss, val_loss, metrics, lr):
        """Print epoch results in Ultralytics format"""
        # Box separator
        print()

        # Loss line
        box_loss = train_loss.get('box', 0)
        cls_loss = train_loss.get('cls', 0)
        obj_loss = train_loss.get('obj', 0)

        print(f"      Epoch    GPU_mem   box_loss   cls_loss   obj_loss  Instances       Size")
        print(f"      {epoch:>4}          -    {box_loss:>7.4f}    {cls_loss:>7.4f}    {obj_loss:>7.4f}          -        640")

        # Metrics line
        if metrics.get('map50', 0) > 0:
            print()
            print(f"                 Class     Images  Instances          P          R      mAP50   mAP50-95")
            print(f"                   all          -          -    {metrics.get('precision', 0):>7.3f}    {metrics.get('recall', 0):>7.3f}    {metrics.get('map50', 0):>7.3f}    {metrics.get('map50_95', 0):>7.3f}")

            # Per-class metrics
            ap_per_class = metrics.get('ap_per_class', [])
            p_per_class = metrics.get('p_per_class', [])
            r_per_class = metrics.get('r_per_class', [])

            for i, name in enumerate(self.class_names):
                if i < len(ap_per_class):
                    p = p_per_class[i] if i < len(p_per_class) else 0
                    r = r_per_class[i] if i < len(r_per_class) else 0
                    ap = ap_per_class[i]
                    print(f"          {name:>12}          -          -    {p:>7.3f}    {r:>7.3f}    {ap:>7.3f}          -")

    def log_train_batch(self, batch_idx, total_batches, loss_dict, lr):
        """Log training batch progress"""
        progress = (batch_idx + 1) / total_batches * 100
        box_loss = loss_dict.get('box', 0)
        cls_loss = loss_dict.get('cls', 0)
        obj_loss = loss_dict.get('obj', 0)

        print(f"\r      [{batch_idx+1}/{total_batches}] {progress:>5.1f}%   "
              f"box: {box_loss:.4f}   cls: {cls_loss:.4f}   obj: {obj_loss:.4f}   "
              f"lr: {lr:.6f}", end='')

    def print_training_start(self, model_name, num_epochs, batch_size, img_size, num_params):
        """Print training start info"""
        print()
        print("=" * 80)
        print(f"                         SSViT-YOLOv11n Training")
        print("=" * 80)
        print()
        print(f"Model:         {model_name}")
        print(f"Parameters:    {num_params:,}")
        print(f"Epochs:        {num_epochs}")
        print(f"Batch size:    {batch_size}")
        print(f"Image size:    {img_size}")
        print(f"Classes:       {len(self.class_names)} ({', '.join(self.class_names)})")
        print(f"Save dir:      {self.save_dir}")
        print()
        print("-" * 80)

    def print_training_end(self, total_epochs, total_time):
        """Print training end info"""
        print()
        print("=" * 80)
        print(f"                         Training Complete")
        print("=" * 80)
        print()
        print(f"Total epochs:      {total_epochs}")
        print(f"Total time:        {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"Best mAP@0.5:      {self.best_map50:.4f} (epoch {self.best_epoch})")
        print(f"Results saved to:  {self.save_dir}")
        print()

        # Save final summary
        self.training_info['end_time'] = datetime.now().isoformat()
        self.training_info['total_epochs'] = total_epochs
        self.training_info['total_time_seconds'] = total_time
        self.training_info['best_map50'] = self.best_map50
        self.training_info['best_epoch'] = self.best_epoch

        with open(self.save_dir / 'training_info.json', 'w') as f:
            json.dump(self.training_info, f, indent=2)

    def close(self):
        """Close all loggers"""
        if self.csv_file:
            self.csv_file.close()
        if self.tb_writer:
            self.tb_writer.close()


class ProgressBar:
    """Simple progress bar for training"""

    def __init__(self, total, desc="", width=50):
        self.total = total
        self.desc = desc
        self.width = width
        self.current = 0

    def update(self, n=1):
        self.current += n
        self._display()

    def _display(self):
        percent = self.current / self.total
        filled = int(self.width * percent)
        bar = '█' * filled + '░' * (self.width - filled)
        print(f'\r{self.desc}: |{bar}| {self.current}/{self.total} [{percent*100:.1f}%]', end='')

    def close(self):
        print()


if __name__ == "__main__":
    # Test logger
    print("Testing Training Logger...")

    class_names = ['barely-riped', 'over-riped', 'riped', 'semi-riped', 'unriped']
    logger = TrainingLogger('./test_logs', class_names=class_names, use_tensorboard=False)

    # Simulate training start
    logger.print_training_start(
        model_name="SSViT-YOLOv11n",
        num_epochs=100,
        batch_size=16,
        img_size=640,
        num_params=2160000
    )

    # Simulate some epochs
    import random
    for epoch in range(1, 4):
        train_loss = {
            'box': random.uniform(0.01, 0.1),
            'cls': random.uniform(0.01, 0.1),
            'obj': random.uniform(0.01, 0.1),
            'total': random.uniform(0.03, 0.3)
        }

        val_loss = {
            'box': random.uniform(0.01, 0.1),
            'cls': random.uniform(0.01, 0.1),
            'obj': random.uniform(0.01, 0.1),
            'total': random.uniform(0.03, 0.3)
        }

        metrics = {
            'precision': random.uniform(0.3, 0.8),
            'recall': random.uniform(0.3, 0.8),
            'map50': random.uniform(0.2, 0.7),
            'map50_95': random.uniform(0.1, 0.5),
            'f1': random.uniform(0.3, 0.7),
            'ap_per_class': np.random.uniform(0.2, 0.8, 5),
            'p_per_class': np.random.uniform(0.3, 0.8, 5),
            'r_per_class': np.random.uniform(0.3, 0.8, 5),
        }

        logger.log_epoch(epoch, train_loss, val_loss, metrics, lr=0.001)

    # Simulate training end
    logger.print_training_end(total_epochs=3, total_time=120.5)

    logger.close()

    # Cleanup
    import shutil
    shutil.rmtree('./test_logs')
    print("\nLogger test complete!")
