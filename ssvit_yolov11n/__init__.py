"""
SSViT-YOLOv11n: Single Scale Vision Transformer + YOLOv11n
For Coffee Bean Ripeness Detection

Based on paper: "SSViT-YOLOv11: fusing lightweight YOLO & ViT for coffee fruit maturity detection"

Modules:
- AKConv: Arbitrary Kernel Convolution
- SSViT: Single Scale Vision Transformer
- MSCA: Multi-Scale Convolutional Attention
- AKC3K2: AKConv integrated with C3K2 block
"""

from .akconv import AKConv, AKConvLight, AKConvBlock
from .ssvit import SSViT, SSViTLight
from .msca import MSCA, MSCA_v2, MSCAHead
from .akc3k2 import AKC3K2, AKC3K2_True, AKC3K2_False, C3K2
from .ssvit_yolov11 import SSViTYOLOv11n, create_ssvit_yolov11n

__version__ = "1.0.0"
__author__ = "Based on SSViT-YOLOv11 paper"

__all__ = [
    # AKConv
    'AKConv',
    'AKConvLight',
    'AKConvBlock',

    # SSViT
    'SSViT',
    'SSViTLight',

    # MSCA
    'MSCA',
    'MSCA_v2',
    'MSCAHead',

    # AKC3K2
    'AKC3K2',
    'AKC3K2_True',
    'AKC3K2_False',
    'C3K2',

    # Full model
    'SSViTYOLOv11n',
    'create_ssvit_yolov11n',
]
