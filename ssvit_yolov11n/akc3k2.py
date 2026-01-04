"""
AKC3K2 - Arbitrary Kernel Convolution in C3K2 Module
From paper: SSViT-YOLOv11: fusing lightweight YOLO & ViT for coffee fruit maturity detection

AKC3K2 replaces standard convolutions in C3K2 (Cross Stage Partial with 2 kernels)
with AKConv for better feature alignment with irregular object shapes.

C3K2 is a key building block in YOLOv11 backbone and neck.
"""

import torch
import torch.nn as nn
from akconv import AKConv, AKConvLight


def autopad(k, p=None, d=1):
    """Calculate padding for 'same' output size"""
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p


class Conv(nn.Module):
    """Standard convolution with BatchNorm and SiLU activation"""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class AKConvModule(nn.Module):
    """AKConv with BatchNorm and activation - drop-in replacement for Conv"""

    def __init__(self, c1, c2, k=3, s=1, num_points=None, act=True):
        super().__init__()

        # Default num_points based on kernel size
        if num_points is None:
            num_points = k * k

        self.akconv = AKConvLight(c1, c2, kernel_size=k, stride=s)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.akconv(x))


class Bottleneck(nn.Module):
    """Standard bottleneck block"""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class AKBottleneck(nn.Module):
    """Bottleneck with AKConv - replaces standard conv with adaptive kernel"""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)

        # First conv: standard 1x1 for channel reduction
        self.cv1 = Conv(c1, c_, k[0], 1)

        # Second conv: AKConv for adaptive feature extraction
        self.cv2 = AKConvModule(c_, c2, k=k[1], s=1)

        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3K2(nn.Module):
    """
    C3K2 - Cross Stage Partial with 2 convolution kernels
    Standard YOLOv11 building block
    """

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        # c3k controls whether to use 3x3 or smaller kernels
        if c3k:
            self.m = nn.ModuleList(
                Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0)
                for _ in range(n)
            )
        else:
            self.m = nn.ModuleList(
                Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=0.5)
                for _ in range(n)
            )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class AKC3K2(nn.Module):
    """
    AKC3K2 - C3K2 with AKConv (Arbitrary Kernel Convolution)

    Replaces standard convolutions in bottlenecks with AKConv for better
    feature alignment with irregular object shapes like coffee beans.

    From paper: "On the basis of YOLOv11n, compared by integrating
    AKConv/AKC3K2T/AKC3K2F modules increases mAP by 1.71%/2.27%/2.69%"
    """

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        # Use AKBottleneck instead of standard Bottleneck
        self.m = nn.ModuleList(
            AKBottleneck(self.c, self.c, shortcut, g, k=(1, 3), e=1.0 if c3k else 0.5)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class AKC3K2_True(nn.Module):
    """
    AKC3K2T - AKC3K2 with c3k=True
    Uses larger effective receptive field, better for larger objects
    """

    def __init__(self, c1, c2, n=1, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        self.m = nn.ModuleList(
            AKBottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class AKC3K2_False(nn.Module):
    """
    AKC3K2F - AKC3K2 with c3k=False
    Uses smaller effective receptive field, more parameter efficient
    Best performance in paper: mAP +2.69%
    """

    def __init__(self, c1, c2, n=1, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        self.m = nn.ModuleList(
            AKBottleneck(self.c, self.c, shortcut, g, k=(1, 3), e=0.5)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


if __name__ == "__main__":
    # Test AKC3K2
    print("Testing AKC3K2 modules...")

    x = torch.randn(2, 64, 40, 40)

    # Test standard C3K2
    c3k2 = C3K2(64, 128, n=2)
    out = c3k2(x)
    print(f"C3K2: {x.shape} -> {out.shape}")

    # Test AKC3K2
    akc3k2 = AKC3K2(64, 128, n=2)
    out = akc3k2(x)
    print(f"AKC3K2: {x.shape} -> {out.shape}")

    # Test AKC3K2_True
    akc3k2_t = AKC3K2_True(64, 128, n=2)
    out = akc3k2_t(x)
    print(f"AKC3K2_True: {x.shape} -> {out.shape}")

    # Test AKC3K2_False (best in paper)
    akc3k2_f = AKC3K2_False(64, 128, n=2)
    out = akc3k2_f(x)
    print(f"AKC3K2_False: {x.shape} -> {out.shape}")

    # Count parameters
    print(f"\nParameters:")
    print(f"  C3K2: {sum(p.numel() for p in c3k2.parameters()):,}")
    print(f"  AKC3K2: {sum(p.numel() for p in akc3k2.parameters()):,}")
    print(f"  AKC3K2_True: {sum(p.numel() for p in akc3k2_t.parameters()):,}")
    print(f"  AKC3K2_False: {sum(p.numel() for p in akc3k2_f.parameters()):,}")

    # Paper reports:
    # - AKConv: mAP +1.71%, params -17.2%
    # - AKC3K2T: mAP +2.27%, params -25.6%
    # - AKC3K2F: mAP +2.69%, params -19.1%
