"""
Metrics for Object Detection Evaluation
Implements mAP, Precision, Recall, F1, Confusion Matrix
Compatible with YOLO-style evaluation
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json


def box_iou(box1, box2):
    """
    Calculate IoU between two sets of boxes

    Args:
        box1: [N, 4] in xyxy format
        box2: [M, 4] in xyxy format

    Returns:
        iou: [N, M] IoU matrix
    """
    def box_area(box):
        return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])

    area1 = box_area(box1)
    area2 = box_area(box2)

    # Intersection
    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N, M, 2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - inter

    return inter / (union + 1e-7)


def xywh2xyxy(x):
    """Convert [x_center, y_center, w, h] to [x1, y1, x2, y2]"""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
    return y


def xyxy2xywh(x):
    """Convert [x1, y1, x2, y2] to [x_center, y_center, w, h]"""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x_center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y_center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def compute_ap(recall, precision):
    """
    Compute Average Precision using all-point interpolation

    Args:
        recall: Recall values
        precision: Precision values

    Returns:
        ap: Average Precision
        mpre: Interpolated precision
        mrec: Interpolated recall
    """
    # Append sentinel values
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Make precision monotonically decreasing
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Calculate area under curve
    x = np.linspace(0, 1, 101)  # 101-point interpolation
    ap = np.trapz(np.interp(x, mrec, mpre), x)

    return ap, mpre, mrec


class ConfusionMatrix:
    """
    Confusion Matrix for object detection
    """

    def __init__(self, num_classes, conf_threshold=0.25, iou_threshold=0.5):
        self.num_classes = num_classes
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.matrix = np.zeros((num_classes + 1, num_classes + 1))  # +1 for background

    def process_batch(self, detections, labels):
        """
        Process a batch of detections and labels

        Args:
            detections: [N, 6] (x1, y1, x2, y2, conf, class)
            labels: [M, 5] (class, x_center, y_center, w, h) normalized
        """
        if detections is None or len(detections) == 0:
            # All labels are false negatives
            for label in labels:
                self.matrix[int(label[0]), self.num_classes] += 1
            return

        # Filter by confidence
        detections = detections[detections[:, 4] >= self.conf_threshold]

        if len(labels) == 0:
            # All detections are false positives
            for det in detections:
                self.matrix[self.num_classes, int(det[5])] += 1
            return

        # Convert labels to xyxy format (assuming 640x640 image)
        gt_boxes = xywh2xyxy(labels[:, 1:5] * 640)
        gt_classes = labels[:, 0].astype(int)

        det_boxes = detections[:, :4]
        det_classes = detections[:, 5].astype(int)

        # Calculate IoU
        iou = box_iou(torch.tensor(gt_boxes), torch.tensor(det_boxes)).numpy()

        # Match detections to ground truth
        matched_gt = set()
        matched_det = set()

        # Sort detections by confidence
        det_order = np.argsort(-detections[:, 4])

        for det_idx in det_order:
            det_class = det_classes[det_idx]

            # Find best matching ground truth
            best_iou = 0
            best_gt = -1

            for gt_idx in range(len(gt_boxes)):
                if gt_idx in matched_gt:
                    continue
                if gt_classes[gt_idx] != det_class:
                    continue
                if iou[gt_idx, det_idx] > best_iou:
                    best_iou = iou[gt_idx, det_idx]
                    best_gt = gt_idx

            if best_iou >= self.iou_threshold:
                # True positive
                self.matrix[gt_classes[best_gt], det_class] += 1
                matched_gt.add(best_gt)
                matched_det.add(det_idx)
            else:
                # False positive
                self.matrix[self.num_classes, det_class] += 1

        # Count false negatives
        for gt_idx in range(len(gt_boxes)):
            if gt_idx not in matched_gt:
                self.matrix[gt_classes[gt_idx], self.num_classes] += 1

    def get_matrix(self):
        return self.matrix

    def plot(self, class_names, save_path=None):
        """Plot confusion matrix"""
        fig, ax = plt.subplots(figsize=(10, 8))

        names = list(class_names) + ['background']

        im = ax.imshow(self.matrix, cmap='Blues')

        ax.set_xticks(np.arange(len(names)))
        ax.set_yticks(np.arange(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_yticklabels(names)

        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')

        # Add text annotations
        for i in range(len(names)):
            for j in range(len(names)):
                text = ax.text(j, i, int(self.matrix[i, j]),
                              ha='center', va='center', color='black' if self.matrix[i, j] < self.matrix.max()/2 else 'white')

        plt.colorbar(im)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            plt.close()
        else:
            plt.show()


class DetectionMetrics:
    """
    Complete detection metrics calculator
    Computes mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1
    """

    def __init__(self, num_classes, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names or [f'class_{i}' for i in range(num_classes)]
        self.reset()

    def reset(self):
        """Reset all statistics"""
        self.stats = []  # List of (correct, conf, pred_cls, target_cls)
        self.confusion_matrix = ConfusionMatrix(self.num_classes)

    def process_batch(self, predictions, targets, iou_thresholds=None):
        """
        Process a batch of predictions and targets

        Args:
            predictions: [N, 6] (x1, y1, x2, y2, conf, class) in pixel coords
            targets: [M, 6] (batch_idx, class, x, y, w, h) normalized
        """
        if iou_thresholds is None:
            iou_thresholds = torch.linspace(0.5, 0.95, 10)

        if predictions is None or len(predictions) == 0:
            if len(targets) > 0:
                self.stats.append((
                    torch.zeros(0, len(iou_thresholds), dtype=torch.bool),
                    torch.zeros(0),
                    torch.zeros(0),
                    targets[:, 1].cpu()
                ))
            return

        if len(targets) == 0:
            self.stats.append((
                torch.zeros(len(predictions), len(iou_thresholds), dtype=torch.bool),
                predictions[:, 4].cpu(),
                predictions[:, 5].cpu(),
                torch.zeros(0)
            ))
            return

        # Convert targets to xyxy format (assuming 640x640)
        tbox = xywh2xyxy(targets[:, 2:6] * 640)
        tcls = targets[:, 1]

        # Calculate IoU
        iou = box_iou(tbox, predictions[:, :4])

        # Assign predictions to targets
        correct = torch.zeros(len(predictions), len(iou_thresholds), dtype=torch.bool)

        for i, threshold in enumerate(iou_thresholds):
            matches = torch.where((iou >= threshold) & (tcls[:, None] == predictions[:, 5]))

            if matches[0].shape[0]:
                # Get unique matches
                matches = torch.cat((torch.stack(matches, 1), iou[matches[0], matches[1]][:, None]), 1)

                if matches.shape[0] > 1:
                    matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1].cpu(), return_index=True)[1]]
                    matches = matches[np.unique(matches[:, 0].cpu(), return_index=True)[1]]

                correct[matches[:, 1].long(), i] = True

        self.stats.append((correct.cpu(), predictions[:, 4].cpu(), predictions[:, 5].cpu(), tcls.cpu()))

    def compute(self):
        """
        Compute all metrics

        Returns:
            dict with keys: precision, recall, f1, map50, map50_95, ap_per_class
        """
        if not self.stats:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'map50': 0.0,
                'map50_95': 0.0,
                'ap_per_class': np.zeros(self.num_classes)
            }

        # Concatenate statistics
        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*self.stats)]

        if len(stats) and stats[0].any():
            correct, conf, pred_cls, target_cls = stats

            # Sort by confidence
            i = np.argsort(-conf)
            correct, conf, pred_cls = correct[i], conf[i], pred_cls[i]

            # Find unique classes
            unique_classes = np.unique(target_cls)

            # Compute metrics per class
            ap = np.zeros((self.num_classes, correct.shape[1]))
            p = np.zeros(self.num_classes)
            r = np.zeros(self.num_classes)

            for ci, c in enumerate(range(self.num_classes)):
                i = pred_cls == c
                n_l = (target_cls == c).sum()  # Number of labels
                n_p = i.sum()  # Number of predictions

                if n_p == 0 or n_l == 0:
                    continue

                # Cumulative false positives and true positives
                fpc = (1 - correct[i]).cumsum(0)
                tpc = correct[i].cumsum(0)

                # Recall
                recall = tpc / (n_l + 1e-16)

                # Precision
                precision = tpc / (tpc + fpc)

                # AP per IoU threshold
                for j in range(correct.shape[1]):
                    ap[ci, j], _, _ = compute_ap(recall[:, j], precision[:, j])

                # P, R at IoU=0.5 (index 0)
                p[ci] = precision[-1, 0] if len(precision) else 0
                r[ci] = recall[-1, 0] if len(recall) else 0

            # Mean metrics
            mp = p.mean()
            mr = r.mean()
            map50 = ap[:, 0].mean()
            map50_95 = ap.mean()
            f1 = 2 * mp * mr / (mp + mr + 1e-16)

            return {
                'precision': mp,
                'recall': mr,
                'f1': f1,
                'map50': map50,
                'map50_95': map50_95,
                'ap_per_class': ap[:, 0],  # AP@0.5 per class
                'p_per_class': p,
                'r_per_class': r
            }

        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'map50': 0.0,
            'map50_95': 0.0,
            'ap_per_class': np.zeros(self.num_classes)
        }

    def print_results(self, results=None):
        """Print results in Ultralytics format"""
        if results is None:
            results = self.compute()

        # Header
        print(f"\n{'Class':<15}{'Images':>8}{'Instances':>10}{'P':>10}{'R':>10}{'mAP50':>10}{'mAP50-95':>10}")
        print("-" * 73)

        # All classes
        print(f"{'all':<15}{'-':>8}{'-':>10}{results['precision']:>10.3f}{results['recall']:>10.3f}{results['map50']:>10.3f}{results['map50_95']:>10.3f}")

        # Per class
        if 'ap_per_class' in results:
            for i, name in enumerate(self.class_names):
                p = results.get('p_per_class', np.zeros(self.num_classes))[i]
                r = results.get('r_per_class', np.zeros(self.num_classes))[i]
                ap50 = results['ap_per_class'][i]
                print(f"{name:<15}{'-':>8}{'-':>10}{p:>10.3f}{r:>10.3f}{ap50:>10.3f}{'-':>10}")


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""

    def __init__(self, patience=50, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        """
        Args:
            score: Metric to monitor (higher is better, e.g., mAP)

        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop


if __name__ == "__main__":
    # Test metrics
    print("Testing Detection Metrics...")

    class_names = ['barely-riped', 'over-riped', 'riped', 'semi-riped', 'unriped']
    metrics = DetectionMetrics(num_classes=5, class_names=class_names)

    # Simulate some predictions and targets
    # predictions: [x1, y1, x2, y2, conf, class]
    preds = torch.tensor([
        [100, 100, 200, 200, 0.9, 2],  # riped, high conf
        [300, 300, 400, 400, 0.8, 0],  # barely-riped
        [150, 150, 250, 250, 0.6, 2],  # riped, lower conf
    ]).float()

    # targets: [batch_idx, class, x_center, y_center, w, h] normalized
    targets = torch.tensor([
        [0, 2, 0.234, 0.234, 0.156, 0.156],  # riped
        [0, 0, 0.547, 0.547, 0.156, 0.156],  # barely-riped
    ]).float()

    metrics.process_batch(preds, targets)
    results = metrics.compute()
    metrics.print_results(results)
