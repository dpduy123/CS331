"""
SSViT - Single Scale Vision Transformer
From paper: SSViT-YOLOv11: fusing lightweight YOLO & ViT for coffee fruit maturity detection

SSViT is a lightweight Vision Transformer that only uses C5 (the deepest) feature map,
making it computationally efficient while still capturing global context.

Key features:
- Single scale processing (C5 only, not C3/C4/C5)
- Patch embedding with small patches
- Lightweight transformer encoder
- 94% fewer parameters than standard ViT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PatchEmbedding(nn.Module):
    """
    Convert feature map to patch embeddings

    Takes C5 feature map and splits it into patches, then projects to embedding dimension
    """

    def __init__(self, in_channels, embed_dim, patch_size=2):
        """
        Args:
            in_channels: Number of input channels (C5 channels)
            embed_dim: Embedding dimension for transformer
            patch_size: Size of each patch (default 2x2)
        """
        super().__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Convolution to project patches to embedding dimension
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args:
            x: Input feature map [B, C, H, W]

        Returns:
            Patch embeddings [B, num_patches, embed_dim]
        """
        B, C, H, W = x.shape

        # Project patches
        x = self.proj(x)  # [B, embed_dim, H/patch_size, W/patch_size]

        # Flatten spatial dimensions
        x = x.flatten(2)  # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, embed_dim]

        # Normalize
        x = self.norm(x)

        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention for SSViT
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # QKV projection
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)

        # Output projection
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Input tensor [B, N, embed_dim]

        Returns:
            Output tensor [B, N, embed_dim]
        """
        B, N, C = x.shape

        # Compute QKV
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # Output projection
        x = self.proj(x)
        x = self.dropout(x)

        return x


class MLP(nn.Module):
    """
    MLP (Feed-Forward Network) for Transformer
    """

    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.0):
        super().__init__()

        hidden_dim = int(embed_dim * mlp_ratio)

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single Transformer Encoder Block
    """

    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)

    def forward(self, x):
        # Self-attention with residual
        x = x + self.attn(self.norm1(x))

        # MLP with residual
        x = x + self.mlp(self.norm2(x))

        return x


class SSViT(nn.Module):
    """
    Single Scale Vision Transformer (SSViT)

    Lightweight ViT that processes only C5 feature maps for global context.

    Architecture:
    1. Patch Embedding: Convert C5 features to patch tokens
    2. Positional Encoding: Add learnable position embeddings
    3. Transformer Encoder: Stack of self-attention + MLP blocks
    4. Reshape: Convert back to spatial feature map

    Benefits:
    - 94% fewer parameters than full ViT
    - Captures global context for small objects
    - Complements CNN's local features
    """

    def __init__(
        self,
        in_channels=512,      # C5 channels (YOLOv11n: 512)
        embed_dim=256,        # Transformer embedding dimension
        num_heads=4,          # Number of attention heads
        num_layers=2,         # Number of transformer blocks
        mlp_ratio=2.0,        # MLP hidden dim ratio
        patch_size=2,         # Patch size for embedding
        dropout=0.0
    ):
        super().__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        # Patch embedding
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)

        # Learnable positional encoding (will be interpolated if size changes)
        # Initialize for typical C5 size (20x20 -> 10x10 patches)
        self.pos_embed = nn.Parameter(torch.zeros(1, 100, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer encoder
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Project back to original channels
        self.proj_out = nn.Conv2d(embed_dim, in_channels, kernel_size=1)

    def interpolate_pos_embed(self, x, h, w):
        """Interpolate positional embeddings if input size differs from training"""
        num_patches = h * w
        N = self.pos_embed.shape[1]

        if num_patches == N:
            return self.pos_embed

        # Interpolate
        pos_embed = self.pos_embed.reshape(1, int(N**0.5), int(N**0.5), -1)
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # [1, embed_dim, sqrt(N), sqrt(N)]
        pos_embed = F.interpolate(pos_embed, size=(h, w), mode='bilinear', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, h * w, -1)

        return pos_embed

    def forward(self, x):
        """
        Args:
            x: C5 feature map [B, C, H, W]

        Returns:
            Enhanced C5 feature map [B, C, H, W]
        """
        B, C, H, W = x.shape
        residual = x

        # Patch embedding
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]

        # Get patch grid size
        h = H // self.patch_size
        w = W // self.patch_size

        # Add positional encoding
        pos_embed = self.interpolate_pos_embed(x, h, w)
        x = x + pos_embed

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Reshape back to spatial
        x = x.transpose(1, 2).reshape(B, self.embed_dim, h, w)

        # Upsample to original size if needed
        if h != H or w != W:
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)

        # Project to original channels
        x = self.proj_out(x)

        # Residual connection
        x = x + residual

        return x


class SSViTLight(nn.Module):
    """
    Even lighter version of SSViT with fewer parameters
    Suitable for very small datasets
    """

    def __init__(
        self,
        in_channels=512,
        embed_dim=128,
        num_heads=2,
        num_layers=1,
        patch_size=2
    ):
        super().__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        # Simple projection
        self.proj_in = nn.Conv2d(in_channels, embed_dim, kernel_size=1)

        # Single attention layer
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio=2.0)

        # Project back
        self.proj_out = nn.Conv2d(embed_dim, in_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x

        # Project to lower dimension
        x = self.proj_in(x)  # [B, embed_dim, H, W]

        # Flatten for attention
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, embed_dim]

        # Attention
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        # Reshape back
        x = x.transpose(1, 2).reshape(B, self.embed_dim, H, W)

        # Project to original channels
        x = self.proj_out(x)

        # Residual
        x = x + residual

        return x


if __name__ == "__main__":
    # Test SSViT
    print("Testing SSViT...")

    # Simulate C5 feature map from YOLOv11n
    # For 640x640 input, C5 is typically 20x20
    x = torch.randn(2, 512, 20, 20)

    # Test full SSViT
    ssvit = SSViT(in_channels=512, embed_dim=256, num_heads=4, num_layers=2)
    out = ssvit(x)
    print(f"SSViT: {x.shape} -> {out.shape}")

    # Test light version
    ssvit_light = SSViTLight(in_channels=512, embed_dim=128, num_heads=2)
    out = ssvit_light(x)
    print(f"SSViTLight: {x.shape} -> {out.shape}")

    # Count parameters
    print(f"\nParameters:")
    print(f"  SSViT: {sum(p.numel() for p in ssvit.parameters()):,}")
    print(f"  SSViTLight: {sum(p.numel() for p in ssvit_light.parameters()):,}")

    # Test with different input sizes
    print("\nTesting different input sizes:")
    for size in [10, 20, 40]:
        x = torch.randn(1, 512, size, size)
        out = ssvit(x)
        print(f"  Input: {x.shape} -> Output: {out.shape}")
