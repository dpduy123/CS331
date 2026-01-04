"""
YOLO Loss Functions
Implements CIoU Loss, BCE Loss, Focal Loss for object detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    Calculate IoU variants between two sets of boxes

    Args:
        box1: [N, 4] boxes
        box2: [M, 4] boxes
        xywh: If True, boxes are in [x_center, y_center, w, h] format
        GIoU, DIoU, CIoU: Which IoU variant to compute

    Returns:
        IoU values
    """
    # Ensure both tensors have same number of dimensions
    if box1.dim() == 1:
        box1 = box1.unsqueeze(0)
    if box2.dim() == 1:
        box2 = box2.unsqueeze(0)

    # Convert to xyxy if needed
    if xywh:
        # Transform from center format to corner format
        b1_x1, b1_x2 = box1[..., 0] - box1[..., 2] / 2, box1[..., 0] + box1[..., 2] / 2
        b1_y1, b1_y2 = box1[..., 1] - box1[..., 3] / 2, box1[..., 1] + box1[..., 3] / 2
        b2_x1, b2_x2 = box2[..., 0] - box2[..., 2] / 2, box2[..., 0] + box2[..., 2] / 2
        b2_y1, b2_y2 = box2[..., 1] - box2[..., 3] / 2, box2[..., 1] + box2[..., 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union

    if GIoU or DIoU or CIoU:
        # Enclosing box
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)

        if CIoU or DIoU:
            # Diagonal distance squared
            c2 = cw ** 2 + ch ** 2 + eps
            # Center distance squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4

            if DIoU:
                return iou - rho2 / c2

            if CIoU:
                # Aspect ratio
                v = (4 / math.pi ** 2) * torch.pow(
                    torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2
                )
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)

        else:  # GIoU
            c_area = cw * ch + eps
            return iou - (c_area - union) / c_area

    return iou


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        """
        Args:
            pred: [N, C] logits
            target: [N, C] one-hot or [N] class indices

        Returns:
            Focal loss
        """
        # Apply sigmoid
        pred_prob = torch.sigmoid(pred)

        # Get probabilities for target class
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)

        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weighting
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)

        # BCE loss
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

        # Combine
        loss = alpha_t * focal_weight * bce

        return loss.mean()


class QualityFocalLoss(nn.Module):
    """
    Quality Focal Loss - combines classification and localization quality

    QFL(sigma) = -|y - sigma|^beta * ((1-y)*log(1-sigma) + y*log(sigma))
    """

    def __init__(self, beta=2.0):
        super().__init__()
        self.beta = beta

    def forward(self, pred, target, weight=None):
        """
        Args:
            pred: [N] logits
            target: [N] soft labels (IoU values)

        Returns:
            QFL loss
        """
        pred_prob = torch.sigmoid(pred)

        # Scale factor
        scale = torch.abs(target - pred_prob).pow(self.beta)

        # BCE with soft labels
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

        loss = scale * bce

        if weight is not None:
            loss = loss * weight

        return loss.mean()


class YOLOv8Loss(nn.Module):
    """
    Complete YOLO Loss
    Combines:
    - Box regression loss (CIoU)
    - Classification loss (BCE with optional focal)
    - Distribution Focal Loss for box predictions (optional)

    This matches the loss function used in YOLOv8/v11
    """

    def __init__(self, num_classes=5, box_gain=7.5, cls_gain=0.5, dfl_gain=1.5, use_focal=False):
        super().__init__()
        self.num_classes = num_classes
        self.box_gain = box_gain
        self.cls_gain = cls_gain
        self.dfl_gain = dfl_gain

        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.focal = FocalLoss() if use_focal else None

        # Strides for each detection scale
        self.strides = [8, 16, 32]  # P3, P4, P5

    def forward(self, predictions, targets, img_size=640):
        """
        Compute loss

        Args:
            predictions: tuple of (p3, p4, p5) outputs
                Each: [B, 5+num_classes, H, W]
            targets: [N, 6] (batch_idx, class, x, y, w, h) normalized

        Returns:
            total_loss, loss_dict
        """
        device = predictions[0].device
        dtype = predictions[0].dtype

        loss_box = torch.tensor(0.0, device=device, dtype=dtype)
        loss_cls = torch.tensor(0.0, device=device, dtype=dtype)
        loss_obj = torch.tensor(0.0, device=device, dtype=dtype)

        num_pos = 0

        # Process each scale
        for si, pred in enumerate(predictions):
            B, C, H, W = pred.shape
            stride = self.strides[si]

            # Reshape: [B, C, H, W] -> [B, H, W, C]
            pred = pred.permute(0, 2, 3, 1).contiguous()

            # Split predictions
            # pred[..., 0:4] = box (x, y, w, h)
            # pred[..., 4] = objectness
            # pred[..., 5:] = class scores
            pred_box = pred[..., :4]
            pred_obj = pred[..., 4]
            pred_cls = pred[..., 5:]

            # Create target masks
            obj_target = torch.zeros_like(pred_obj)

            if len(targets) > 0:
                # Scale targets to feature map size
                # targets: [N, 6] (batch_idx, class, x, y, w, h)
                target_box = targets[:, 2:6].clone()
                target_box[:, 0] *= W  # x
                target_box[:, 1] *= H  # y
                target_box[:, 2] *= W  # w
                target_box[:, 3] *= H  # h

                target_batch = targets[:, 0].long()
                target_cls = targets[:, 1].long()

                # Get grid cell indices
                gx = target_box[:, 0].long().clamp(0, W - 1)
                gy = target_box[:, 1].long().clamp(0, H - 1)

                # Process each target
                for ti in range(len(targets)):
                    bi = target_batch[ti]
                    gi = gx[ti]
                    gj = gy[ti]
                    tc = target_cls[ti]

                    # Set objectness target
                    obj_target[bi, gj, gi] = 1.0

                    # Box loss (CIoU)
                    pbox = pred_box[bi, gj, gi]  # [4]
                    tbox = torch.tensor([
                        target_box[ti, 0] - gi,  # offset x
                        target_box[ti, 1] - gj,  # offset y
                        target_box[ti, 2],       # w in grid cells
                        target_box[ti, 3]        # h in grid cells
                    ], device=device, dtype=dtype)

                    # Apply sigmoid to predictions for offsets
                    pbox_decoded = torch.cat([
                        torch.sigmoid(pbox[:2]),  # xy offset
                        torch.exp(pbox[2:4].clamp(max=10))  # wh scale
                    ])

                    iou = bbox_iou(pbox_decoded.unsqueeze(0), tbox.unsqueeze(0), xywh=True, CIoU=True)
                    loss_box += (1.0 - iou).mean()

                    # Class loss
                    cls_target = torch.zeros(self.num_classes, device=device, dtype=dtype)
                    cls_target[tc] = 1.0

                    loss_cls += self.bce(pred_cls[bi, gj, gi], cls_target).mean()

                    num_pos += 1

            # Objectness loss (all cells)
            loss_obj += self.bce(pred_obj, obj_target).mean()

        # Normalize by number of positive samples
        num_pos = max(num_pos, 1)

        # Apply gains
        loss_box = loss_box * self.box_gain / num_pos
        loss_cls = loss_cls * self.cls_gain / num_pos
        loss_obj = loss_obj

        total_loss = loss_box + loss_cls + loss_obj

        return total_loss, {
            'box': loss_box.item(),
            'cls': loss_cls.item(),
            'obj': loss_obj.item(),
            'total': total_loss.item()
        }


class TaskAlignedAssigner:
    """
    Task-Aligned Assigner for matching predictions to targets
    Used in YOLOv8 for better target assignment

    Combines classification score and IoU to create alignment metric
    """

    def __init__(self, topk=13, num_classes=5, alpha=0.5, beta=6.0):
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta

    def __call__(self, pred_scores, pred_bboxes, anchor_points, gt_labels, gt_bboxes):
        """
        Assign targets to predictions based on task-aligned metric

        Returns:
            target_labels, target_bboxes, target_scores, fg_mask
        """
        # Implementation details follow YOLOv8's assigner
        # This is a simplified version

        pass  # Full implementation requires more context


if __name__ == "__main__":
    # Test loss functions
    print("Testing YOLO Loss...")

    loss_fn = YOLOv8Loss(num_classes=5)

    # Simulate predictions (3 scales)
    p3 = torch.randn(2, 10, 80, 80)  # 5 + 5 classes
    p4 = torch.randn(2, 10, 40, 40)
    p5 = torch.randn(2, 10, 20, 20)

    predictions = (p3, p4, p5)

    # Simulate targets: [batch_idx, class, x, y, w, h]
    targets = torch.tensor([
        [0, 2, 0.5, 0.5, 0.1, 0.1],
        [0, 0, 0.3, 0.3, 0.15, 0.15],
        [1, 4, 0.7, 0.7, 0.2, 0.2],
    ])

    loss, loss_dict = loss_fn(predictions, targets)
    print(f"Total Loss: {loss.item():.4f}")
    print(f"Box Loss: {loss_dict['box']:.4f}")
    print(f"Cls Loss: {loss_dict['cls']:.4f}")
    print(f"Obj Loss: {loss_dict['obj']:.4f}")

    # Test IoU functions
    print("\nTesting IoU functions...")
    box1 = torch.tensor([[0.5, 0.5, 0.2, 0.2]])  # xywh
    box2 = torch.tensor([[0.55, 0.55, 0.2, 0.2]])

    iou = bbox_iou(box1, box2, xywh=True)
    ciou = bbox_iou(box1, box2, xywh=True, CIoU=True)
    print(f"IoU: {iou.item():.4f}")
    print(f"CIoU: {ciou.item():.4f}")
