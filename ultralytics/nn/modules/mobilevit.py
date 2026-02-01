# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
MobileViT modules adapted for YOLO.

This module provides MobileViT and MobileViTv2 blocks that combine convolutional layers 
with transformer-based global representations for efficient mobile-friendly architectures.

Original implementations from:
- MobileViT: https://arxiv.org/abs/2110.02178
- MobileViTv2: https://arxiv.org/abs/2206.02680
"""

import math
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ("MobileViTBlock", "MobileViTBlockv2")


# ============================================================================
# Base Helper Classes
# ============================================================================


class LayerNorm2d(nn.Module):
    """2D Layer Normalization for NCHW format."""

    def __init__(self, num_features, eps=1e-5):
        """Initialize LayerNorm2d with number of features and epsilon."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, x):
        """Apply layer normalization on NCHW tensor."""
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class LinearSelfAttention(nn.Module):
    """Linear Self-Attention mechanism for MobileViTv2."""

    def __init__(self, embed_dim, attn_dropout=0.0, bias=True):
        """Initialize Linear Self-Attention."""
        super().__init__()
        self.qkv_proj = nn.Conv2d(embed_dim, 1 + (2 * embed_dim), 1, bias=bias)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_proj = nn.Conv2d(embed_dim, embed_dim, 1, bias=bias)
        self.embed_dim = embed_dim

    def forward(self, x, x_prev=None):
        """Apply linear self-attention mechanism."""
        # x: [B, C, P, N]
        qkv = self.qkv_proj(x)

        # Project x into query, key and value
        # Query: [B, 1, P, N]
        # Key, Value: [B, C, P, N]
        query, key, value = torch.split(qkv, [1, self.embed_dim, self.embed_dim], dim=1)

        # Apply softmax along N dimension
        context_scores = F.softmax(query, dim=-1)
        context_scores = self.attn_dropout(context_scores)

        # Compute context vector: [B, C, P, N] x [B, 1, P, N] -> [B, C, P, N]
        context_vector = key * context_scores
        # [B, C, P, N] --> [B, C, P, 1]
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)

        # Combine context vector with values: [B, C, P, N] * [B, C, P, 1] -> [B, C, P, N]
        out = F.relu(value) * context_vector.expand_as(value)
        out = self.out_proj(out)
        return out


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention mechanism."""

    def __init__(self, embed_dim, num_heads, attn_dropout=0.0, bias=True):
        """Initialize Multi-Head Attention."""
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, x_q, x_kv=None, key_padding_mask=None, attn_mask=None):
        """Apply multi-head attention."""
        if x_kv is None:
            x_kv = x_q

        b_sz, n_patches, in_channels = x_q.shape

        # Compute QKV: [B, N, C] -> [B, N, 3*C]
        qkv = self.qkv_proj(x_q)
        # [B, N, 3*C] -> [B, N, 3, num_heads, head_dim]
        qkv = qkv.reshape(b_sz, n_patches, 3, self.num_heads, self.head_dim)

        # [B, N, 3, h, d] --> [3, B, h, N, d]
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        query = query * self.scaling

        # [B, h, N, d] x [B, h, d, N] --> [B, h, N, N]
        attn = torch.matmul(query, key.transpose(-1, -2))

        if attn_mask is not None:
            attn = attn + attn_mask

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # [B, h, N, N] x [B, h, N, d] --> [B, h, N, d]
        out = torch.matmul(attn, value)

        # [B, h, N, d] --> [B, N, h, d] --> [B, N, h*d]
        out = out.transpose(1, 2).reshape(b_sz, n_patches, self.embed_dim)
        out = self.out_proj(out)

        return out


# ============================================================================
# Transformer Components
# ============================================================================


class TransformerEncoder(nn.Module):
    """Transformer Encoder block with pre-norm architecture."""

    def __init__(
        self,
        embed_dim,
        ffn_latent_dim,
        num_heads=8,
        attn_dropout=0.0,
        dropout=0.0,
        ffn_dropout=0.0,
        norm_layer="layer_norm",
    ):
        """Initialize Transformer Encoder."""
        super().__init__()

        self.pre_norm_mha = nn.Sequential(
            nn.LayerNorm(embed_dim),
            MultiHeadAttention(embed_dim, num_heads, attn_dropout=attn_dropout, bias=True),
            nn.Dropout(dropout),
        )

        self.pre_norm_ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, ffn_latent_dim, bias=True),
            nn.SiLU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(ffn_latent_dim, embed_dim, bias=True),
            nn.Dropout(dropout),
        )

        self.embed_dim = embed_dim
        self.ffn_dim = ffn_latent_dim

    def forward(self, x, x_prev=None, key_padding_mask=None, attn_mask=None):
        """Apply transformer encoder block."""
        # Multi-head attention
        res = x
        x = self.pre_norm_mha[0](x)  # norm
        x = self.pre_norm_mha[1](x, x_prev, key_padding_mask, attn_mask)  # mha
        x = self.pre_norm_mha[2](x)  # dropout
        x = x + res

        # Feed forward network
        x = x + self.pre_norm_ffn(x)
        return x


class LinearAttnFFN(nn.Module):
    """Linear Attention with FFN for MobileViTv2."""

    def __init__(
        self,
        embed_dim,
        ffn_latent_dim,
        attn_dropout=0.0,
        dropout=0.1,
        ffn_dropout=0.0,
        norm_layer="layer_norm_2d",
    ):
        """Initialize Linear Attention FFN block."""
        super().__init__()

        if norm_layer == "layer_norm_2d":
            norm_class = LayerNorm2d
        else:
            norm_class = nn.BatchNorm2d

        self.pre_norm_attn = nn.Sequential(
            norm_class(embed_dim),
            LinearSelfAttention(embed_dim=embed_dim, attn_dropout=attn_dropout, bias=True),
            nn.Dropout(dropout),
        )

        self.pre_norm_ffn = nn.Sequential(
            norm_class(embed_dim),
            nn.Conv2d(embed_dim, ffn_latent_dim, 1, 1, bias=True),
            nn.SiLU(),
            nn.Dropout(ffn_dropout),
            nn.Conv2d(ffn_latent_dim, embed_dim, 1, 1, bias=True),
            nn.Dropout(dropout),
        )

        self.embed_dim = embed_dim
        self.ffn_dim = ffn_latent_dim

    def forward(self, x, x_prev=None):
        """Apply linear attention and FFN."""
        if x_prev is None:
            # self-attention
            x = x + self.pre_norm_attn(x)
        else:
            # cross-attention
            res = x
            x = self.pre_norm_attn[0](x)  # norm
            x = self.pre_norm_attn[1](x, x_prev)  # attn
            x = self.pre_norm_attn[2](x)  # drop
            x = x + res

        # Feed forward network
        x = x + self.pre_norm_ffn(x)
        return x


# ============================================================================
# Convolution Layers
# ============================================================================


def autopad(k, p=None, d=1):
    """Auto-calculate padding for 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class ConvLayer(nn.Module):
    """Convolutional layer with normalization and activation."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=False,
        use_norm=True,
        use_act=True,
    ):
        """Initialize convolutional layer."""
        super().__init__()
        padding = autopad(kernel_size, None, dilation)

        layers = []
        layers.append(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        )

        if use_norm:
            layers.append(nn.BatchNorm2d(out_channels))

        if use_act:
            layers.append(nn.SiLU())

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        """Apply convolution block."""
        return self.block(x)


# ============================================================================
# MobileViT Blocks
# ============================================================================


class MobileViTBlock(nn.Module):
    """
    MobileViT block combining local and global representations.

    This class implements the MobileViT block from https://arxiv.org/abs/2110.02178.
    It combines convolutional layers for local representations and transformers for
    global representations.

    Args:
        in_channels (int): Number of input channels
        transformer_dim (int): Dimension of transformer
        ffn_dim (int): Dimension of FFN in transformer
        n_transformer_blocks (int): Number of transformer blocks. Default: 2
        head_dim (int): Head dimension in multi-head attention. Default: 32
        attn_dropout (float): Dropout in attention. Default: 0.0
        dropout (float): Dropout rate. Default: 0.0
        ffn_dropout (float): Dropout in FFN. Default: 0.0
        patch_h (int): Patch height. Default: 8
        patch_w (int): Patch width. Default: 8
        conv_ksize (int): Kernel size for local convolution. Default: 3
        dilation (int): Dilation rate. Default: 1
        no_fusion (bool): Whether to skip fusion layer. Default: False
    """

    def __init__(
        self,
        in_channels,
        transformer_dim,
        ffn_dim,
        n_transformer_blocks=2,
        head_dim=32,
        attn_dropout=0.0,
        dropout=0.0,
        ffn_dropout=0.0,
        patch_h=8,
        patch_w=8,
        conv_ksize=3,
        dilation=1,
        no_fusion=False,
    ):
        """Initialize MobileViT block."""
        super().__init__()

        # Local representation
        conv_3x3_in = ConvLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=conv_ksize,
            stride=1,
            dilation=dilation,
            use_norm=True,
            use_act=True,
        )
        conv_1x1_in = ConvLayer(
            in_channels=in_channels,
            out_channels=transformer_dim,
            kernel_size=1,
            stride=1,
            use_norm=False,
            use_act=False,
        )

        self.local_rep = nn.Sequential(conv_3x3_in, conv_1x1_in)

        # Global representation
        assert transformer_dim % head_dim == 0
        num_heads = transformer_dim // head_dim

        global_rep = [
            TransformerEncoder(
                embed_dim=transformer_dim,
                ffn_latent_dim=ffn_dim,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                dropout=dropout,
                ffn_dropout=ffn_dropout,
            )
            for _ in range(n_transformer_blocks)
        ]
        global_rep.append(nn.LayerNorm(transformer_dim))
        self.global_rep = nn.Sequential(*global_rep)

        # Projection
        self.conv_proj = ConvLayer(
            in_channels=transformer_dim,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            use_norm=True,
            use_act=True,
        )

        # Fusion
        self.fusion = None
        if not no_fusion:
            self.fusion = ConvLayer(
                in_channels=2 * in_channels,
                out_channels=in_channels,
                kernel_size=conv_ksize,
                stride=1,
                use_norm=True,
                use_act=True,
            )

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h

    def unfolding(self, feature_map):
        """Convert feature map to patches."""
        patch_w, patch_h = self.patch_w, self.patch_h
        patch_area = int(patch_w * patch_h)
        batch_size, in_channels, orig_h, orig_w = feature_map.shape

        new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
        new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)

        interpolate = False
        if new_w != orig_w or new_h != orig_h:
            # Resize to match patch dimensions
            feature_map = F.interpolate(feature_map, size=(new_h, new_w), mode="bilinear", align_corners=False)
            interpolate = True

        # Number of patches along width and height
        num_patch_w = new_w // patch_w
        num_patch_h = new_h // patch_h
        num_patches = num_patch_h * num_patch_w

        # [B, C, H, W] --> [B * C * n_h, p_h, n_w, p_w]
        reshaped_fm = feature_map.reshape(batch_size * in_channels * num_patch_h, patch_h, num_patch_w, patch_w)
        # [B * C * n_h, p_h, n_w, p_w] --> [B * C * n_h, n_w, p_h, p_w]
        transposed_fm = reshaped_fm.transpose(1, 2)
        # [B * C * n_h, n_w, p_h, p_w] --> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
        reshaped_fm = transposed_fm.reshape(batch_size, in_channels, num_patches, patch_area)
        # [B, C, N, P] --> [B, P, N, C]
        transposed_fm = reshaped_fm.transpose(1, 3)
        # [B, P, N, C] --> [BP, N, C]
        patches = transposed_fm.reshape(batch_size * patch_area, num_patches, -1)

        info_dict = {
            "orig_size": (orig_h, orig_w),
            "batch_size": batch_size,
            "interpolate": interpolate,
            "total_patches": num_patches,
            "num_patches_w": num_patch_w,
            "num_patches_h": num_patch_h,
        }

        return patches, info_dict

    def folding(self, patches, info_dict):
        """Convert patches back to feature map."""
        # [BP, N, C] --> [B, P, N, C]
        patches = patches.contiguous().view(
            info_dict["batch_size"], self.patch_area, info_dict["total_patches"], -1
        )

        batch_size, pixels, num_patches, channels = patches.size()
        num_patch_h = info_dict["num_patches_h"]
        num_patch_w = info_dict["num_patches_w"]

        # [B, P, N, C] --> [B, C, N, P]
        patches = patches.transpose(1, 3)

        # [B, C, N, P] --> [B*C*n_h, n_w, p_h, p_w]
        feature_map = patches.reshape(
            batch_size * channels * num_patch_h, num_patch_w, self.patch_h, self.patch_w
        )
        # [B*C*n_h, n_w, p_h, p_w] --> [B*C*n_h, p_h, n_w, p_w]
        feature_map = feature_map.transpose(1, 2)
        # [B*C*n_h, p_h, n_w, p_w] --> [B, C, H, W]
        feature_map = feature_map.reshape(
            batch_size, channels, num_patch_h * self.patch_h, num_patch_w * self.patch_w
        )
        if info_dict["interpolate"]:
            feature_map = F.interpolate(
                feature_map, size=info_dict["orig_size"], mode="bilinear", align_corners=False
            )
        return feature_map

    def forward(self, x):
        """Apply MobileViT block."""
        res = x

        # Local representation
        fm = self.local_rep(x)

        # Convert feature map to patches
        patches, info_dict = self.unfolding(fm)

        # Learn global representations
        for transformer_layer in self.global_rep:
            patches = transformer_layer(patches)

        # Convert patches back to feature map
        fm = self.folding(patches=patches, info_dict=info_dict)

        # Projection
        fm = self.conv_proj(fm)

        # Fusion
        if self.fusion is not None:
            fm = self.fusion(torch.cat((res, fm), dim=1))

        return fm


class MobileViTBlockv2(nn.Module):
    """
    MobileViTv2 block with linear attention.

    This class implements the MobileViTv2 block from https://arxiv.org/abs/2206.02680.
    It uses linear attention instead of standard multi-head attention for efficiency.

    Args:
        in_channels (int): Number of input channels
        attn_unit_dim (int): Dimension of attention unit
        ffn_multiplier (float): FFN expansion ratio. Default: 2.0
        n_attn_blocks (int): Number of attention blocks. Default: 2
        attn_dropout (float): Dropout in attention. Default: 0.0
        dropout (float): Dropout rate. Default: 0.0
        ffn_dropout (float): Dropout in FFN. Default: 0.0
        patch_h (int): Patch height. Default: 8
        patch_w (int): Patch width. Default: 8
        conv_ksize (int): Kernel size for local convolution. Default: 3
        dilation (int): Dilation rate. Default: 1
    """

    def __init__(
        self,
        in_channels,
        attn_unit_dim,
        ffn_multiplier=2.0,
        n_attn_blocks=2,
        attn_dropout=0.0,
        dropout=0.0,
        ffn_dropout=0.0,
        patch_h=8,
        patch_w=8,
        conv_ksize=3,
        dilation=1,
    ):
        """Initialize MobileViTv2 block."""
        super().__init__()

        cnn_out_dim = attn_unit_dim

        # Local representation with depthwise separable convolution
        conv_3x3_in = ConvLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=conv_ksize,
            stride=1,
            dilation=dilation,
            groups=in_channels,  # Depthwise
            use_norm=True,
            use_act=True,
        )
        conv_1x1_in = ConvLayer(
            in_channels=in_channels,
            out_channels=cnn_out_dim,
            kernel_size=1,
            stride=1,
            use_norm=False,
            use_act=False,
        )

        self.local_rep = nn.Sequential(conv_3x3_in, conv_1x1_in)

        # Global representation with linear attention
        self.global_rep, attn_unit_dim = self._build_attn_layer(
            d_model=attn_unit_dim,
            ffn_mult=ffn_multiplier,
            n_layers=n_attn_blocks,
            attn_dropout=attn_dropout,
            dropout=dropout,
            ffn_dropout=ffn_dropout,
        )

        # Projection
        self.conv_proj = ConvLayer(
            in_channels=cnn_out_dim,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            use_norm=True,
            use_act=False,
        )

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h
        self.cnn_out_dim = cnn_out_dim

    def _build_attn_layer(self, d_model, ffn_mult, n_layers, attn_dropout, dropout, ffn_dropout):
        """Build attention layers."""
        if isinstance(ffn_mult, Sequence) and len(ffn_mult) == 2:
            ffn_dims = np.linspace(ffn_mult[0], ffn_mult[1], n_layers, dtype=float) * d_model
        elif isinstance(ffn_mult, Sequence) and len(ffn_mult) == 1:
            ffn_dims = [ffn_mult[0] * d_model] * n_layers
        elif isinstance(ffn_mult, (int, float)):
            ffn_dims = [ffn_mult * d_model] * n_layers
        else:
            raise NotImplementedError

        # Ensure dims are multiple of 16
        ffn_dims = [int((d // 16) * 16) for d in ffn_dims]

        global_rep = [
            LinearAttnFFN(
                embed_dim=d_model,
                ffn_latent_dim=ffn_dims[block_idx],
                attn_dropout=attn_dropout,
                dropout=dropout,
                ffn_dropout=ffn_dropout,
                norm_layer="layer_norm_2d",
            )
            for block_idx in range(n_layers)
        ]
        global_rep.append(LayerNorm2d(d_model))

        return nn.Sequential(*global_rep), d_model

    def unfolding(self, feature_map):
        """Convert feature map to patches using unfold operation."""
        batch_size, in_channels, img_h, img_w = feature_map.shape

        # [B, C, H, W] --> [B, C, P, N]
        patches = F.unfold(
            feature_map,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )
        patches = patches.reshape(batch_size, in_channels, self.patch_h * self.patch_w, -1)

        return patches, (img_h, img_w)

    def folding(self, patches, output_size):
        """Convert patches back to feature map using fold operation."""
        batch_size, in_dim, patch_size, n_patches = patches.shape

        # [B, C, P, N]
        patches = patches.reshape(batch_size, in_dim * patch_size, n_patches)

        feature_map = F.fold(
            patches,
            output_size=output_size,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )

        return feature_map

    def resize_input_if_needed(self, x):
        """Resize input if dimensions don't match patch size."""
        batch_size, in_channels, orig_h, orig_w = x.shape
        if orig_h % self.patch_h != 0 or orig_w % self.patch_w != 0:
            new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
            new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        return x

    def forward(self, x):
        """Apply MobileViTv2 block."""
        # Resize if needed
        x = self.resize_input_if_needed(x)

        # Local representation
        fm = self.local_rep(x)

        # Convert feature map to patches
        patches, output_size = self.unfolding(fm)

        # Learn global representations on all patches
        patches = self.global_rep(patches)

        # Convert patches back to feature map
        fm = self.folding(patches=patches, output_size=output_size)

        # Projection
        fm = self.conv_proj(fm)

        return fm
