"""
MSCA - Multi-Scale Convolutional Attention
From paper: SSViT-YOLOv11: fusing lightweight YOLO & ViT for coffee fruit maturity detection

MSCA is added to the detection head to improve detection accuracy for objects
at different scales (especially important for coffee beans of varying sizes).

Key features:
- Multi-scale feature extraction with different kernel sizes
- Channel attention mechanism
- Spatial attention mechanism
- Improves detection of objects at different scales
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention Module
    Learns "what" to focus on by modeling inter-channel relationships
    """

    def __init__(self, channels, reduction=16):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Average pooling path
        avg_out = self.mlp(self.avg_pool(x))

        # Max pooling path
        max_out = self.mlp(self.max_pool(x))

        # Combine
        out = self.sigmoid(avg_out + max_out)

        return x * out


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module
    Learns "where" to focus by modeling inter-spatial relationships
    """

    def __init__(self, kernel_size=7):
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Compute attention map
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        concat = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(concat))

        return x * attn


class MultiScaleConv(nn.Module):
    """
    Multi-Scale Convolution Block
    Extracts features at multiple scales using parallel convolutions
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Branch 1: 1x1 conv (point-wise, for channel mixing)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.SiLU()
        )

        # Branch 2: 3x3 conv (local features)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.SiLU()
        )

        # Branch 3: 5x5 conv (medium-range features) - implemented as two 3x3
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.SiLU(),
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.SiLU()
        )

        # Branch 4: 7x7 conv (larger context) - implemented with dilation
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.SiLU()
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        return torch.cat([b1, b2, b3, b4], dim=1)


class MSCA(nn.Module):
    """
    Multi-Scale Convolutional Attention (MSCA)

    Combines multi-scale feature extraction with attention mechanisms
    for improved detection of objects at different scales.

    Architecture:
    1. Multi-scale convolution for features at different scales
    2. Channel attention to focus on important channels
    3. Spatial attention to focus on important spatial locations
    """

    def __init__(self, in_channels, reduction=16):
        super().__init__()

        # Multi-scale feature extraction
        self.multi_scale = MultiScaleConv(in_channels, in_channels)

        # Channel attention
        self.channel_attn = ChannelAttention(in_channels, reduction)

        # Spatial attention
        self.spatial_attn = SpatialAttention(kernel_size=7)

    def forward(self, x):
        residual = x

        # Multi-scale features
        x = self.multi_scale(x)

        # Channel attention
        x = self.channel_attn(x)

        # Spatial attention
        x = self.spatial_attn(x)

        # Residual connection
        x = x + residual

        return x


class MSCA_v2(nn.Module):
    """
    MSCA Version 2 - Lighter version with depthwise separable convolutions
    Better for smaller models like YOLOv11n
    """

    def __init__(self, in_channels, reduction=8):
        super().__init__()

        # Multi-scale depthwise convolutions
        self.dw_conv1 = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3, padding=1, groups=in_channels, bias=False
        )
        self.dw_conv2 = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=5, padding=2, groups=in_channels, bias=False
        )
        self.dw_conv3 = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=7, padding=3, groups=in_channels, bias=False
        )

        self.bn = nn.BatchNorm2d(in_channels * 3)

        # Channel attention with SE-style
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels * 3, in_channels * 3 // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 3 // reduction, in_channels * 3, 1),
            nn.Sigmoid()
        )

        # Projection back to original channels
        self.proj = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1, bias=False)

        self.act = nn.SiLU()

    def forward(self, x):
        residual = x

        # Multi-scale depthwise convolutions
        x1 = self.dw_conv1(x)
        x2 = self.dw_conv2(x)
        x3 = self.dw_conv3(x)

        # Concatenate
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.bn(x)

        # Channel attention
        attn = self.se(x)
        x = x * attn

        # Project back
        x = self.proj(x)
        x = self.act(x)

        # Residual
        x = x + residual

        return x


class MSCAHead(nn.Module):
    """
    MSCA-enhanced Detection Head

    Replaces standard detection head with MSCA attention for better
    multi-scale detection performance.

    Used at each detection scale (P3, P4, P5) in YOLO head.
    """

    def __init__(self, in_channels, num_classes, num_anchors=1):
        super().__init__()

        # MSCA attention
        self.msca = MSCA_v2(in_channels)

        # Detection conv
        self.detect_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU()
        )

        # Output: (x, y, w, h, obj, classes) per anchor
        self.output = nn.Conv2d(
            in_channels,
            num_anchors * (5 + num_classes),
            kernel_size=1
        )

    def forward(self, x):
        x = self.msca(x)
        x = self.detect_conv(x)
        x = self.output(x)
        return x


if __name__ == "__main__":
    # Test MSCA
    print("Testing MSCA...")

    x = torch.randn(2, 256, 40, 40)

    # Test full MSCA
    msca = MSCA(256)
    out = msca(x)
    print(f"MSCA: {x.shape} -> {out.shape}")

    # Test light version
    msca_v2 = MSCA_v2(256)
    out = msca_v2(x)
    print(f"MSCA_v2: {x.shape} -> {out.shape}")

    # Test MSCA Head
    head = MSCAHead(256, num_classes=5)
    out = head(x)
    print(f"MSCAHead (5 classes): {x.shape} -> {out.shape}")

    # Count parameters
    print(f"\nParameters:")
    print(f"  MSCA: {sum(p.numel() for p in msca.parameters()):,}")
    print(f"  MSCA_v2: {sum(p.numel() for p in msca_v2.parameters()):,}")
    print(f"  MSCAHead: {sum(p.numel() for p in head.parameters()):,}")

    # Test different scales
    print("\nTesting different scales:")
    for channels, size in [(64, 80), (128, 40), (256, 20)]:
        x = torch.randn(1, channels, size, size)
        msca = MSCA_v2(channels)
        out = msca(x)
        print(f"  Input: {x.shape} -> Output: {out.shape}")
