"""
AKConv - Arbitrary Kernel Convolution
From paper: SSViT-YOLOv11: fusing lightweight YOLO & ViT for coffee fruit maturity detection

AKConv dynamically adjusts kernel sampling positions, reducing parameters
while improving feature alignment with target shapes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AKConv(nn.Module):
    """
    Arbitrary Kernel Convolution (AKConv)

    Unlike standard convolutions with fixed square kernels, AKConv uses
    adaptive sampling positions that can better align with irregular object shapes.

    Key features:
    - Arbitrary kernel size (not limited to square)
    - Learnable sampling offsets
    - Better adaptation to coffee bean shapes
    """

    def __init__(self, in_channels, out_channels, num_points=9, stride=1, dilation=1):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            num_points: Number of sampling points (default 9, like 3x3 kernel)
            stride: Convolution stride
            dilation: Dilation rate
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_points = num_points
        self.stride = stride
        self.dilation = dilation

        # Generate initial sampling positions (similar to standard conv)
        self.initial_offsets = self._get_initial_offsets(num_points)

        # Learnable offset predictor - predicts offsets for each spatial location
        self.offset_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, 2 * num_points, kernel_size=1)  # 2 for (x, y) offsets
        )

        # Main convolution weights
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, num_points))
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = in_channels * num_points
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

        # Initialize offset conv to output near-zero offsets initially
        nn.init.zeros_(self.offset_conv[-1].weight)
        nn.init.zeros_(self.offset_conv[-1].bias)

    def _get_initial_offsets(self, num_points):
        """Generate initial sampling offsets in a grid pattern"""
        # Approximate square grid
        grid_size = int(math.ceil(math.sqrt(num_points)))
        center = (grid_size - 1) / 2

        offsets = []
        for i in range(grid_size):
            for j in range(grid_size):
                if len(offsets) < num_points:
                    offsets.append([i - center, j - center])

        return torch.tensor(offsets, dtype=torch.float32)

    def forward(self, x):
        """
        Forward pass with adaptive sampling

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Output tensor [B, out_channels, H', W']
        """
        B, C, H, W = x.shape

        # Predict offsets for each spatial location
        offsets = self.offset_conv(x)  # [B, 2*num_points, H, W]

        # Reshape offsets
        offsets = offsets.view(B, self.num_points, 2, H, W)
        offsets = offsets.permute(0, 3, 4, 1, 2)  # [B, H, W, num_points, 2]

        # Get initial offsets on the same device
        initial_offsets = self.initial_offsets.to(x.device)

        # Add initial offsets to learned offsets
        sampling_offsets = offsets + initial_offsets.view(1, 1, 1, self.num_points, 2)

        # Apply dilation
        sampling_offsets = sampling_offsets * self.dilation

        # Create sampling grid
        # Normalize to [-1, 1] for grid_sample
        grid_h = torch.arange(H, device=x.device, dtype=x.dtype)
        grid_w = torch.arange(W, device=x.device, dtype=x.dtype)
        grid_y, grid_x = torch.meshgrid(grid_h, grid_w, indexing='ij')

        # Expand grid for batch and num_points
        grid_y = grid_y.view(1, H, W, 1).expand(B, -1, -1, self.num_points)
        grid_x = grid_x.view(1, H, W, 1).expand(B, -1, -1, self.num_points)

        # Add offsets to grid
        sample_y = grid_y + sampling_offsets[..., 0]
        sample_x = grid_x + sampling_offsets[..., 1]

        # Normalize to [-1, 1]
        sample_y = 2.0 * sample_y / (H - 1) - 1.0
        sample_x = 2.0 * sample_x / (W - 1) - 1.0

        # Stack to form grid [B, H, W, num_points, 2]
        sample_grid = torch.stack([sample_x, sample_y], dim=-1)

        # Reshape for grid_sample: [B, H*W*num_points, 1, 2]
        sample_grid = sample_grid.view(B, H * W * self.num_points, 1, 2)

        # Sample features
        # Reshape input for sampling
        sampled = F.grid_sample(
            x, sample_grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )  # [B, C, H*W*num_points, 1]

        # Reshape sampled features
        sampled = sampled.view(B, C, H, W, self.num_points)
        sampled = sampled.permute(0, 2, 3, 1, 4)  # [B, H, W, C, num_points]

        # Apply convolution weights
        # weight: [out_channels, in_channels, num_points]
        output = torch.einsum('bhwcp,ocp->bhwo', sampled, self.weight)

        # Add bias
        output = output + self.bias.view(1, 1, 1, -1)

        # Permute to [B, out_channels, H, W]
        output = output.permute(0, 3, 1, 2)

        # Apply stride if needed
        if self.stride > 1:
            output = output[:, :, ::self.stride, ::self.stride]

        return output


class AKConvBlock(nn.Module):
    """
    AKConv block with BatchNorm and activation
    """

    def __init__(self, in_channels, out_channels, num_points=9, stride=1):
        super().__init__()

        self.conv = AKConv(in_channels, out_channels, num_points, stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# Simplified AKConv for efficiency (used in paper)
class AKConvLight(nn.Module):
    """
    Lightweight AKConv using depthwise separable convolution with adaptive kernels
    More efficient version for mobile/embedded deployment
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()

        self.stride = stride
        padding = kernel_size // 2

        # Depthwise conv with slightly larger kernel to capture offsets implicitly
        self.dw_conv = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size + 2,  # Larger kernel for adaptive receptive field
            stride=stride,
            padding=padding + 1,
            groups=in_channels,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)

        # Pointwise conv
        self.pw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.act = nn.SiLU()

    def forward(self, x):
        x = self.act(self.bn1(self.dw_conv(x)))
        x = self.act(self.bn2(self.pw_conv(x)))
        return x


if __name__ == "__main__":
    # Test AKConv
    print("Testing AKConv...")

    x = torch.randn(2, 64, 32, 32)

    # Test full AKConv
    akconv = AKConv(64, 128, num_points=9)
    out = akconv(x)
    print(f"AKConv: {x.shape} -> {out.shape}")

    # Test AKConvBlock
    akblock = AKConvBlock(64, 128)
    out = akblock(x)
    print(f"AKConvBlock: {x.shape} -> {out.shape}")

    # Test lightweight version
    aklight = AKConvLight(64, 128)
    out = aklight(x)
    print(f"AKConvLight: {x.shape} -> {out.shape}")

    # Count parameters
    print(f"\nParameters:")
    print(f"  AKConv: {sum(p.numel() for p in akconv.parameters()):,}")
    print(f"  AKConvBlock: {sum(p.numel() for p in akblock.parameters()):,}")
    print(f"  AKConvLight: {sum(p.numel() for p in aklight.parameters()):,}")
