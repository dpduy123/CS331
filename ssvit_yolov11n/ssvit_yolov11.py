"""
SSViT-YOLOv11n - Complete Model Implementation
From paper: SSViT-YOLOv11: fusing lightweight YOLO & ViT for coffee fruit maturity detection

This implements the full SSViT-YOLOv11n architecture:
- Backbone: YOLOv11n with AKC3K2 modules
- Neck: PANet with SSViT on C5
- Head: Detection head with MSCA attention

Performance (from paper):
- Precision: 81.1%
- Recall: 77.4%
- mAP@0.5: 84.54%
- FPS: 23
- Parameters: 2.16M
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from akconv import AKConvLight
from akc3k2 import Conv, AKC3K2, AKC3K2_False, C3K2
from ssvit import SSViT, SSViTLight
from msca import MSCA_v2, MSCAHead


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast"""

    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class Upsample(nn.Module):
    """Upsample module"""

    def __init__(self, scale_factor=2, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class Concat(nn.Module):
    """Concatenate tensors along dimension"""

    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cat(x, self.dim)


class SSViTYOLOv11nBackbone(nn.Module):
    """
    YOLOv11n Backbone with AKC3K2

    Architecture (following paper):
    - Conv stem
    - AKC3K2 blocks instead of C3K2
    - Outputs C3, C4, C5 feature maps

    For 640x640 input:
    - C3: 80x80, 64 channels
    - C4: 40x40, 128 channels
    - C5: 20x20, 512 channels
    """

    def __init__(self):
        super().__init__()

        # Stem: 640 -> 320 -> 160
        self.stem = nn.Sequential(
            Conv(3, 16, 3, 2),   # /2: 320x320
            Conv(16, 32, 3, 2),  # /4: 160x160
        )

        # Stage 1: 160 -> 80 (P3/8)
        self.down1 = Conv(32, 64, 3, 2)  # /8: 80x80
        self.stage1 = AKC3K2_False(64, 64, n=1)

        # Stage 2: 80 -> 40 (P4/16)
        self.down2 = Conv(64, 128, 3, 2)  # /16: 40x40
        self.stage2 = AKC3K2_False(128, 128, n=1)

        # Stage 3: 40 -> 20 (P5/32)
        self.down3 = Conv(128, 256, 3, 2)  # /32: 20x20
        self.stage3 = nn.Sequential(
            AKC3K2_False(256, 256, n=1),
            AKC3K2_False(256, 512, n=1),
            SPPF(512, 512),
        )

    def forward(self, x):
        # Stem: 640 -> 160
        x = self.stem(x)

        # Stage 1 - C3: 160 -> 80
        x = self.down1(x)
        c3 = self.stage1(x)  # 80x80, 64ch

        # Stage 2 - C4: 80 -> 40
        x = self.down2(c3)
        c4 = self.stage2(x)  # 40x40, 128ch

        # Stage 3 - C5: 40 -> 20
        x = self.down3(c4)
        c5 = self.stage3(x)  # 20x20, 512ch

        return c3, c4, c5


class SSViTYOLOv11nNeck(nn.Module):
    """
    PANet Neck with SSViT on C5

    Architecture:
    - SSViT applied only to C5 (single scale ViT)
    - Top-down path: C5 -> C4 -> C3
    - Bottom-up path: C3 -> C4 -> C5

    Input from backbone:
    - C3: 80x80, 64 channels
    - C4: 40x40, 128 channels
    - C5: 20x20, 512 channels
    """

    def __init__(self, use_ssvit=True):
        super().__init__()

        self.use_ssvit = use_ssvit

        # SSViT on C5 (512 channels)
        if use_ssvit:
            self.ssvit = SSViTLight(in_channels=512, embed_dim=128, num_heads=2)

        # Top-down path
        self.upsample = Upsample(scale_factor=2)

        # C5 -> C4 fusion: upsample C5 (512->128ch) and concat with C4 (128ch)
        self.td_conv1 = Conv(512, 128, 1, 1)  # Reduce C5 channels to match C4
        self.td_c3k2_1 = AKC3K2_False(128 + 128, 128, n=1)  # 128 from C5 + 128 from C4 = 256 -> 128

        # C4 -> C3 fusion: upsample (128->64ch) and concat with C3 (64ch)
        self.td_conv2 = Conv(128, 64, 1, 1)  # Reduce to match C3 channels
        self.td_c3k2_2 = AKC3K2_False(64 + 64, 64, n=1)  # 64 + 64 = 128 -> 64

        # Bottom-up path
        # P3 -> P4 fusion
        self.bu_conv1 = Conv(64, 64, 3, 2)  # Downsample P3
        self.bu_c3k2_1 = AKC3K2_False(64 + 128, 128, n=1)  # 64 from P3 + 128 from P4 = 192 -> 128

        # P4 -> P5 fusion
        self.bu_conv2 = Conv(128, 128, 3, 2)  # Downsample P4
        self.bu_c3k2_2 = AKC3K2_False(128 + 512, 256, n=1)  # 128 from P4 + 512 from C5 = 640 -> 256

    def forward(self, features):
        c3, c4, c5 = features
        # c3: [B, 64, 80, 80]
        # c4: [B, 128, 40, 40]
        # c5: [B, 512, 20, 20]

        # Apply SSViT to C5
        if self.use_ssvit:
            c5 = self.ssvit(c5)

        # Top-down path
        # C5 -> C4: upsample 20x20 -> 40x40
        p5_small = self.td_conv1(c5)  # [B, 128, 20, 20]
        p5_up = self.upsample(p5_small)  # [B, 128, 40, 40]
        p4 = self.td_c3k2_1(torch.cat([p5_up, c4], 1))  # [B, 256, 40, 40] -> [B, 128, 40, 40]

        # C4 -> C3: upsample 40x40 -> 80x80
        p4_small = self.td_conv2(p4)  # [B, 64, 40, 40]
        p4_up = self.upsample(p4_small)  # [B, 64, 80, 80]
        p3 = self.td_c3k2_2(torch.cat([p4_up, c3], 1))  # [B, 128, 80, 80] -> [B, 64, 80, 80]

        # Bottom-up path
        # P3 -> P4: downsample 80x80 -> 40x40
        p3_down = self.bu_conv1(p3)  # [B, 64, 40, 40]
        n4 = self.bu_c3k2_1(torch.cat([p3_down, p4], 1))  # [B, 192, 40, 40] -> [B, 128, 40, 40]

        # P4 -> P5: downsample 40x40 -> 20x20
        n4_down = self.bu_conv2(n4)  # [B, 128, 20, 20]
        n5 = self.bu_c3k2_2(torch.cat([n4_down, c5], 1))  # [B, 640, 20, 20] -> [B, 256, 20, 20]

        return p3, n4, n5


class SSViTYOLOv11nHead(nn.Module):
    """
    Detection Head with MSCA

    Outputs detections at 3 scales (P3, P4, P5)
    Each scale has MSCA attention before detection

    Input from neck:
    - P3: 80x80, 64 channels
    - P4: 40x40, 128 channels
    - P5: 20x20, 256 channels
    """

    def __init__(self, num_classes=5, use_msca=True):
        super().__init__()

        self.num_classes = num_classes
        self.use_msca = use_msca

        # MSCA modules for each scale (matching neck output channels)
        if use_msca:
            self.msca_p3 = MSCA_v2(64)   # P3: 64 channels
            self.msca_p4 = MSCA_v2(128)  # P4: 128 channels
            self.msca_p5 = MSCA_v2(256)  # P5: 256 channels

        # Detection heads
        # Output: (x, y, w, h, objectness, class_probs)
        self.detect_p3 = nn.Conv2d(64, 5 + num_classes, 1)
        self.detect_p4 = nn.Conv2d(128, 5 + num_classes, 1)
        self.detect_p5 = nn.Conv2d(256, 5 + num_classes, 1)

    def forward(self, features):
        p3, p4, p5 = features
        # p3: [B, 64, 80, 80]
        # p4: [B, 128, 40, 40]
        # p5: [B, 256, 20, 20]

        # Apply MSCA
        if self.use_msca:
            p3 = self.msca_p3(p3)
            p4 = self.msca_p4(p4)
            p5 = self.msca_p5(p5)

        # Detection outputs
        out_p3 = self.detect_p3(p3)  # [B, 5+num_classes, 80, 80]
        out_p4 = self.detect_p4(p4)  # [B, 5+num_classes, 40, 40]
        out_p5 = self.detect_p5(p5)  # [B, 5+num_classes, 20, 20]

        return out_p3, out_p4, out_p5


class SSViTYOLOv11n(nn.Module):
    """
    Complete SSViT-YOLOv11n Model

    Combines:
    - Backbone with AKC3K2
    - Neck with SSViT
    - Head with MSCA

    Usage:
        model = SSViTYOLOv11n(num_classes=5)
        x = torch.randn(1, 3, 640, 640)
        outputs = model(x)  # Returns detections at 3 scales
    """

    def __init__(self, num_classes=5, use_ssvit=True, use_msca=True):
        super().__init__()

        self.num_classes = num_classes

        self.backbone = SSViTYOLOv11nBackbone()
        self.neck = SSViTYOLOv11nNeck(use_ssvit=use_ssvit)
        self.head = SSViTYOLOv11nHead(num_classes=num_classes, use_msca=use_msca)

    def forward(self, x):
        # Backbone
        features = self.backbone(x)

        # Neck with SSViT
        features = self.neck(features)

        # Head with MSCA
        outputs = self.head(features)

        return outputs

    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_ssvit_yolov11n(num_classes=5, pretrained=False):
    """
    Factory function to create SSViT-YOLOv11n

    Args:
        num_classes: Number of detection classes
        pretrained: Load pretrained weights (not implemented yet)

    Returns:
        SSViT-YOLOv11n model
    """
    model = SSViTYOLOv11n(num_classes=num_classes)

    if pretrained:
        # TODO: Load pretrained weights when available
        print("Warning: Pretrained weights not available yet")

    return model


if __name__ == "__main__":
    # Test complete model
    print("Testing SSViT-YOLOv11n...")
    print("=" * 60)

    # Create model
    model = SSViTYOLOv11n(num_classes=5)

    # Test input
    x = torch.randn(1, 3, 640, 640)

    # Forward pass
    outputs = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shapes:")
    for i, out in enumerate(outputs):
        print(f"  P{i+3}: {out.shape}")

    # Count parameters
    total_params = model.count_parameters()
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Parameters (M): {total_params / 1e6:.2f}M")

    # Compare with paper: 2.16M parameters
    print(f"\nPaper reports: 2.16M parameters")
    print(f"Our implementation: {total_params / 1e6:.2f}M parameters")

    # Breakdown by component
    print(f"\nParameter breakdown:")
    print(f"  Backbone: {sum(p.numel() for p in model.backbone.parameters()):,}")
    print(f"  Neck: {sum(p.numel() for p in model.neck.parameters()):,}")
    print(f"  Head: {sum(p.numel() for p in model.head.parameters()):,}")

    # Test different configurations
    print("\n" + "=" * 60)
    print("Testing different configurations:")

    configs = [
        {"use_ssvit": True, "use_msca": True, "name": "Full SSViT-YOLOv11n"},
        {"use_ssvit": False, "use_msca": True, "name": "YOLOv11n + MSCA only"},
        {"use_ssvit": True, "use_msca": False, "name": "YOLOv11n + SSViT only"},
        {"use_ssvit": False, "use_msca": False, "name": "Base YOLOv11n (AKC3K2)"},
    ]

    for cfg in configs:
        model = SSViTYOLOv11n(
            num_classes=5,
            use_ssvit=cfg["use_ssvit"],
            use_msca=cfg["use_msca"]
        )
        params = model.count_parameters()
        print(f"  {cfg['name']}: {params:,} ({params/1e6:.2f}M)")
