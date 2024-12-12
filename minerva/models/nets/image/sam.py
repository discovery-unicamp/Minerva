# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Modified at 2024 by Filipe A. Sampaio
# Changes: Grouping all original SAM files into a single file
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# minerva/THIRD_PARTY_LICENSES/LICENSE_SAM_apache2.txt

import numpy as np
import math
from typing import Any, Dict, List, Tuple, Optional, Type
from functools import partial
import lightning as L
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchmetrics import Metric
from minerva.models.finetune_adapters import LoRA
from minerva.models.nets.mlp import MLP

# based on: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/common.py
class MLPBlock(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) block with two linear layers and an activation function.

    This block applies a linear transformation, followed by an activation function,
    and then another linear transformation. It is typically used in transformer-based
    architectures and MLP-based models for feature transformation.

    Parameters
    ----------
    embedding_dim : int
        The size of the input and output embedding dimension.
    mlp_dim : int
        The size of the hidden layer in the MLP.
    act : Type[nn.Module], optional
        The activation function to use between the two linear layers. 
        By default, it uses GELU activation (`nn.GELU`).

    Attributes
    ----------
    lin1 : nn.Linear
        The first linear layer, which projects from `embedding_dim` to `mlp_dim`.
    lin2 : nn.Linear
        The second linear layer, which projects from `mlp_dim` back to `embedding_dim`.
    act : nn.Module
        The activation function applied between the two linear layers.

    Methods
    -------
    forward(x)
        Applies the MLP block transformation to the input tensor.

    Examples
    --------
    >>> import torch
    >>> from torch import nn
    >>> block = MLPBlock(embedding_dim=128, mlp_dim=256)
    >>> x = torch.randn(32, 128)  # Batch of 32 samples, each with 128 features
    >>> output = block(x)
    >>> output.shape
    torch.Size([32, 128])
    """

    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the MLP block transformation to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor of shape (batch_size, embedding_dim).

        Returns
        -------
        torch.Tensor
            The output tensor of shape (batch_size, embedding_dim), transformed by the MLP block.
        """
        return self.lin2(self.act(self.lin1(x)))

# based on: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/common.py
class LayerNorm2d(nn.Module):
    """
    A 2D Layer Normalization module.

    This module normalizes each channel independently across the spatial dimensions
    (height and width) for a 4D input tensor. It is commonly used in vision-based models 
    to stabilize training and improve convergence.

    Parameters
    ----------
    num_channels : int
        The number of channels in the input tensor.
    eps : float, optional
        A small value added to the denominator for numerical stability. Default is 1e-6.

    Attributes
    ----------
    weight : nn.Parameter
        A learnable scaling factor for each channel, initialized to ones.
    bias : nn.Parameter
        A learnable bias term for each channel, initialized to zeros.
    eps : float
        The epsilon value for numerical stability.

    Methods
    -------
    forward(x)
        Applies 2D layer normalization to the input tensor.

    Examples
    --------
    >>> import torch
    >>> from torch import nn
    >>> layer_norm = LayerNorm2d(num_channels=3)
    >>> x = torch.randn(2, 3, 4, 4)  # Batch of 2 images, 3 channels, 4x4 spatial dimensions
    >>> output = layer_norm(x)
    >>> output.shape
    torch.Size([2, 3, 4, 4])
    """

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies 2D layer normalization to the input tensor.

        The normalization is applied across the spatial dimensions (height and width) 
        for each channel independently.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor of shape (batch_size, num_channels, height, width).

        Returns
        -------
        torch.Tensor
            The normalized tensor of the same shape as the input.
        """
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

"""
*****
ImageEncoderViT
*****
"""

# based on: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py
class ImageEncoderViT(nn.Module):
    """
    Vision Transformer (ViT)-based image encoder for feature extraction.

    This class implements an image encoder based on the Vision Transformer (ViT) architecture. 
    It divides an image into patches, embeds these patches into a higher-dimensional space, 
    processes them through a series of transformer blocks, and outputs a feature map 
    suitable for downstream tasks.

    Parameters
    ----------
    img_size : int, optional
        The size of the input image (assumed to be square). Default is 1024.
    patch_size : int, optional
        The size of each image patch (assumed to be square). Default is 16.
    in_chans : int, optional
        Number of channels in the input image. Default is 3.
    embed_dim : int, optional
        Dimensionality of the patch embeddings. Default is 768.
    depth : int, optional
        Number of transformer blocks. Default is 12.
    num_heads : int, optional
        Number of attention heads in each transformer block. Default is 12.
    mlp_ratio : float, optional
        The ratio of the hidden layer size in the MLP to the embedding dimension. Default is 4.0.
    out_chans : int, optional
        Number of output channels in the final feature map. Default is 256.
    qkv_bias : bool, optional
        If True, add a learnable bias to the query, key, and value projections. Default is True.
    norm_layer : Type[nn.Module], optional
        The normalization layer to use. Default is nn.LayerNorm.
    act_layer : Type[nn.Module], optional
        The activation layer to use. Default is nn.GELU.
    use_abs_pos : bool, optional
        If True, use absolute positional embeddings. Default is True.
    use_rel_pos : bool, optional
        If True, add relative positional embeddings to the attention map. Default is False.
    rel_pos_zero_init : bool, optional
        If True, initialize relative positional parameters to zero. Default is True.
    window_size : int, optional
        The size of the window for windowed self-attention blocks. Default is 0 (global attention).
    global_attn_indexes : Tuple[int, ...], optional
        Indices of transformer blocks that use global attention instead of windowed attention. Default is ().

    Attributes
    ----------
    patch_embed : PatchEmbed
        Module to divide the input image into patches and project them into the embedding space.
    pos_embed : nn.Parameter or None
        Absolute positional embedding tensor, initialized to zeros if `use_abs_pos` is True.
    blocks : nn.ModuleList
        A list of transformer blocks for processing the patch embeddings.
    neck : nn.Sequential
        A series of convolutional and normalization layers applied to the final output.

    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor
        Forward pass of the image encoder.

    Examples
    --------
    >>> import torch
    >>> model = ImageEncoderViT(img_size=1024, patch_size=16, embed_dim=768, out_chans=256)
    >>> x = torch.randn(1, 3, 1024, 1024)  # Batch of 1 image with 3 channels
    >>> output = model(x)
    >>> output.shape
    torch.Size([1, 256, 64, 64])
    """

    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        self.img_size = img_size

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Vision Transformer encoder.

        Divides the input image into patches, applies positional embeddings (if enabled),
        processes the patches through a series of transformer blocks, and applies a 
        convolutional neck to produce the final feature map.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor of shape (batch_size, in_chans, img_size, img_size).

        Returns
        -------
        torch.Tensor
            The output feature map of shape (batch_size, out_chans, H, W), where
            H and W are the spatial dimensions of the output.
        """
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.neck(x.permute(0, 3, 1, 2))

        return x

# based on: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py
class Block(nn.Module):
    """
    Transformer block with support for window attention and residual propagation blocks.

    Parameters
    ----------
    dim : int
        Number of input channels.
    num_heads : int
        Number of attention heads in each ViT block.
    mlp_ratio : float, optional, default=4.0
        Ratio of the MLP hidden dimension to the embedding dimension.
    qkv_bias : bool, optional, default=True
        If True, adds a learnable bias to query, key, and value.
    norm_layer : nn.Module, optional, default=nn.LayerNorm
        Normalization layer to use.
    act_layer : nn.Module, optional, default=nn.GELU
        Activation layer to use.
    use_rel_pos : bool, optional, default=False
        If True, adds relative positional embeddings to the attention map.
    rel_pos_zero_init : bool, optional, default=True
        If True, zero-initializes relative positional parameters.
    window_size : int, optional, default=0
        Window size for window attention blocks. If it equals 0, global attention is used.
    input_size : tuple of int, optional, default=None
        Input resolution for calculating the relative positional parameter size.
        Required when `window_size` > 0.

    Attributes
    ----------
    norm1 : nn.Module
        Normalization layer applied to input before attention.
    attn : Attention
        Attention block used in the transformer.
    norm2 : nn.Module
        Normalization layer applied before MLP block.
    mlp : MLPBlock
        MLP block used for feed-forward processing.
    window_size : int
        The window size for window attention, or 0 if global attention is used.

    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor
        Performs the forward pass of the transformer block.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the transformer block.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor with shape `(batch_size, height, width, channels)`.

        Returns
        -------
        torch.Tensor
            The output tensor after the attention and MLP blocks with residual connections.
        """
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = self.window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = self.window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x
    
    # based on: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py
    def window_partition(self, x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Partition into non-overlapping windows with padding if needed.
        Args:
            x (tensor): input tokens with [B, H, W, C].
            window_size (int): window size.

        Returns:
            windows: windows after partition with [B * num_windows, window_size, window_size, C].
            (Hp, Wp): padded height and width before partition
        """
        B, H, W, C = x.shape

        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w

        x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows, (Hp, Wp)
    
    # based on: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py
    def window_unpartition(
        self, windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Window unpartition into original sequences and removing padding.
        Args:
            windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
            window_size (int): window size.
            pad_hw (Tuple): padded height and width (Hp, Wp).
            hw (Tuple): original height and width (H, W) before padding.

        Returns:
            x: unpartitioned sequences with [B, H, W, C].
        """
        Hp, Wp = pad_hw
        H, W = hw
        B = windows.shape[0] // (Hp * Wp // window_size // window_size)
        x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

        if Hp > H or Wp > W:
            x = x[:, :H, :W, :].contiguous()
        return x

# based on: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py
class Attention(nn.Module):
    """
    Multi-head Attention block with relative position embeddings.

    This class implements a multi-head attention mechanism with optional support
    for relative positional embeddings. It applies scaled dot-product attention
    on the input tensor, with the option to include relative position embeddings
    to enhance the model's ability to capture positional information in the input.

    Parameters
    ----------
    dim : int
        Number of input channels (features).
    num_heads : int, optional
        Number of attention heads (default is 8).
    qkv_bias : bool, optional
        If True, adds a learnable bias to the query, key, and value projections
        (default is True).
    use_rel_pos : bool, optional
        If True, adds relative positional embeddings to the attention map
        (default is False).
    rel_pos_zero_init : bool, optional
        If True, initializes the relative positional parameters to zero
        (default is True).
    input_size : tuple of int, optional
        A tuple (height, width) representing the input resolution, required if
        `use_rel_pos` is True, to calculate the size of the relative positional
        embeddings.

    Attributes
    ----------
    num_heads : int
        The number of attention heads.
    scale : float
        The scaling factor applied to the query in attention computation.
    qkv : nn.Linear
        Linear layer to project the input to queries, keys, and values.
    proj : nn.Linear
        Linear layer to project the output back to the original input dimension.
    use_rel_pos : bool
        Indicates whether relative positional embeddings are used.
    rel_pos_h : nn.Parameter, optional
        Relative positional embeddings for the height dimension, initialized to zeros
        if `use_rel_pos` is True.
    rel_pos_w : nn.Parameter, optional
        Relative positional embeddings for the width dimension, initialized to zeros
        if `use_rel_pos` is True.

    Methods
    -------
    forward(x)
        Performs the forward pass of the attention block, computing the attention
        scores and applying the learned projections.
    
    Raises
    ------
    AssertionError
        If `use_rel_pos` is True and `input_size` is None.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass through the attention block.

        The input tensor `x` is passed through the query, key, and value projections
        and the attention map is computed. If relative positional embeddings are used,
        they are incorporated into the attention scores. Finally, the output is projected
        back to the input dimension.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor with shape (batch_size, height, width, channels).

        Returns
        -------
        torch.Tensor
            The output tensor after applying multi-head attention with the same
            shape as the input (batch_size, height, width, channels).
        """
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = self.add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x
    
    # based on: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py
    def get_rel_pos(self, q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
        """
        Get relative positional embeddings according to the relative positions of
            query and key sizes.
        Args:
            q_size (int): size of query q.
            k_size (int): size of key k.
            rel_pos (Tensor): relative position embeddings (L, C).

        Returns:
            Extracted positional embeddings according to relative positions.
        """
        max_rel_dist = int(2 * max(q_size, k_size) - 1)
        # Interpolate rel pos if needed.
        if rel_pos.shape[0] != max_rel_dist:
            # Interpolate rel pos.
            rel_pos_resized = F.interpolate(
                rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
                size=max_rel_dist,
                mode="linear",
            )
            rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
        else:
            rel_pos_resized = rel_pos

        # Scale the coords with short length if shapes for q and k are different.
        q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
        k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
        relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

        return rel_pos_resized[relative_coords.long()]

    # based on: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py
    def add_decomposed_rel_pos(
            self,
            attn: torch.Tensor,
            q: torch.Tensor,
            rel_pos_h: torch.Tensor,
            rel_pos_w: torch.Tensor,
            q_size: Tuple[int, int],
            k_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
        https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
        Args:
            attn (Tensor): attention map.
            q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
            rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
            rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
            q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
            k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

        Returns:
            attn (Tensor): attention map with added relative positional embeddings.
        """
        q_h, q_w = q_size
        k_h, k_w = k_size
        Rh = self.get_rel_pos(q_h, k_h, rel_pos_h)
        Rw = self.get_rel_pos(q_w, k_w, rel_pos_w)

        B, _, dim = q.shape
        r_q = q.reshape(B, q_h, q_w, dim)
        rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
        rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

        attn = (
            attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
        ).view(B, q_h * q_w, k_h * k_w)

        return attn

# based on: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py
class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.

    This class performs the conversion of input images into a sequence of patch embeddings. 
    It uses a convolutional layer to project the image patches into the desired embedding dimension.

    Parameters
    ----------
    kernel_size : tuple of int, optional
        The size of the convolutional kernel, which determines the size of each patch. 
        Default is (16, 16).
    stride : tuple of int, optional
        The stride of the convolutional layer, which controls the step size of the kernel when sliding over the image. 
        Default is (16, 16).
    padding : tuple of int, optional
        The padding size applied to the input image before performing the convolution. 
        Default is (0, 0).
    in_chans : int, optional
        The number of input channels in the image. Typically, 3 for RGB images.
        Default is 3.
    embed_dim : int, optional
        The number of output channels (embedding dimension) for each patch. 
        This determines the size of the resulting patch embeddings.
        Default is 768.

    Attributes
    ----------
    proj : nn.Conv2d
        A 2D convolutional layer that projects the input image patches into the desired embedding dimension.

    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor
        Applies the patch embedding operation to the input tensor `x` and returns the patch embeddings.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the PatchEmbed layer.

        Args
        ----
        x : torch.Tensor
            The input tensor of shape (B, C, H, W), where B is the batch size, 
            C is the number of channels, and H and W are the height and width of the input image.

        Returns
        -------
        torch.Tensor
            The output tensor of shape (B, H', W', C'), where H' and W' are the spatial 
            dimensions of the patch embeddings, and C' is the embedding dimension.
        """
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x

"""
*****
PromptEncoder
*****
"""

# based on: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/prompt_encoder.py
class PromptEncoder(nn.Module):
    """
    A neural network module that encodes prompts (points, boxes, and masks) for input to the SAM's mask decoder.

    This module supports encoding point prompts (with associated labels), box prompts, and mask prompts
    into embeddings that are used in the mask decoding process. It utilizes a random positional embedding
    strategy and various neural network layers to transform and scale the input data.

    Parameters:
    -----------
        embed_dim : int
            The prompts' embedding dimension.
        image_embedding_size : tuple(int, int)
            The spatial size of the image embedding, as (H, W).
        input_image_size : tuple(int, int)
            The padded size of the image as input to the image encoder, as (H, W).
        mask_in_chans : int
            The number of hidden channels used for encoding input masks.
        activation : Type[nn.Module], optional:
            The activation function to use when encoding input masks. Default is nn.GELU.

    Attributes:
    -----------
        embed_dim : int
            The dimension of the embeddings produced for the prompts.
        input_image_size : tuple(int, int)
            The size of the image input, typically the padded size used by the encoder.
        image_embedding_size : tuple(int, int)
            The spatial dimensions (H, W) of the image embedding.
        pe_layer : PositionEmbeddingRandom
            Positional embedding layer used for encoding point prompts.
        num_point_embeddings : int
            The number of different point embeddings, including for positive/negative points and box corners.
        point_embeddings : nn.ModuleList
            A list of point embeddings for different point categories.
        not_a_point_embed : nn.Embedding
            Embedding for points that are not valid or marked as such.
        mask_input_size : tuple(int, int)
            The size of the input masks after downscaling.
        mask_downscaling : nn.Sequential
            A series of convolutional layers for downscaling and processing mask inputs.
        no_mask_embed : nn.Embedding
            Embedding for cases where no mask is provided.
    
    Methods:
    --------
        get_dense_pe():
            Returns the positional encoding applied to a dense set of points matching the shape of the image encoding.

        _embed_points(points, labels, pad):
            Embeds the point prompts with their respective labels.
        
        _embed_boxes(boxes):
            Embeds the box prompts.
        
        _embed_masks(masks):
            Embeds the mask prompts.
        
        _get_batch_size(points, boxes, masks):
            Returns the batch size based on the input prompts (points, boxes, or masks).
        
        _get_device():
            Returns the device of the point embeddings (used for placement of tensors).
        
        forward(points, boxes, masks):
            Encodes the provided prompts into both sparse and dense embeddings.
    """

    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners
        point_embeddings = [nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        pad: bool,
    ) -> torch.Tensor:
        """ Embeds point prompts. """
        points = points + 0.5  # Shift to center of pixel
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """ Embeds box prompts. """
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """ Embeds mask inputs. """
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def _get_batch_size(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Parameters
        -------
          points : tuple(torch.Tensor, torch.Tensor) or none
            point coordinates and labels to embed.
          boxes : torch.Tensor or none
            boxes to embed
          masks : torch.Tensor or none
            masks to embed

        Returns
        --------
          Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - Sparse embeddings (BxNx(embed_dim)) for points and boxes.
                - Dense embeddings (Bx(embed_dim)x(embed_H)x(embed_W)) for masks.
        """
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self._get_device())
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings

# based on: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/prompt_encoder.py
class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies for embedding coordinates.

    This module generates positional encodings using random spatial frequencies, which are 
    learned through a Gaussian distribution. The resulting encoding represents spatial 
    positions in a coordinate system, and can be used for tasks such as image processing 
    or object detection where positional information is required.

    Parameters:
    -----------
    num_pos_feats : int, optional, default: 64
        The number of positional features to generate for each coordinate. Higher values
        provide more fine-grained spatial encoding.
    
    scale : float, optional, default: None
        A scaling factor for the random positional encoding matrix. If `None` or non-positive,
        the scale defaults to 1.0. This can be used to control the magnitude of the encoding.
    
    Attributes:
    -----------
    positional_encoding_gaussian_matrix : torch.Tensor
        A buffer tensor containing the Gaussian distribution used to generate the positional 
        encodings with spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Positionally encode points that are normalized to [0, 1].

        This method applies a random spatial frequency encoding to normalized coordinates in 
        the range [0, 1] to produce sinusoidal encodings. The encoding involves mapping the 
        coordinates to a new space using a learned random matrix, followed by applying sine and 
        cosine transformations.

        Parameters:
        -----------
        coords : torch.Tensor
            A tensor of coordinates with shape (..., 2), where each coordinate is a point
            in a 2D space. The values should be normalized to the range [0, 1].

        Returns:
        --------
        torch.Tensor
            A tensor containing the positional encoding with the shape (..., 2 * num_pos_feats), 
            where `num_pos_feats` is the number of positional features per coordinate.
        """
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """
        Generate positional encoding for a grid of the specified size.

        This method generates a positional encoding for a 2D grid of the given height and width. 
        The coordinates of the grid are normalized to the range [0, 1], and the corresponding 
        positional encoding is computed using the learned random spatial frequencies.

        Parameters:
        -----------
        size : Tuple[int, int]
            A tuple representing the height (h) and width (w) of the grid for which the 
            positional encoding should be generated.

        Returns:
        --------
        torch.Tensor
            A tensor containing the positional encoding for the grid, with shape (C, H, W),
            where C is the number of positional features, and H and W are the height and 
            width of the grid, respectively.
        """
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Positionally encode points that are not normalized to [0, 1].

        This method generates positional encodings for coordinates that are not normalized 
        to the range [0, 1]. The coordinates are first rescaled to the [0, 1] range based 
        on the provided image size, and then the positional encoding is computed.

        Parameters:
        -----------
        coords_input : torch.Tensor
            A tensor of coordinates with shape (B, N, 2), where B is the batch size and N is 
            the number of points. Each coordinate is a 2D point that is not normalized.

        image_size : Tuple[int, int]
            A tuple representing the height and width of the image, used to normalize the coordinates 
            to the [0, 1] range.

        Returns:
        --------
        torch.Tensor
            A tensor containing the positional encoding for the input coordinates, with 
            shape (B, N, C), where B is the batch size, N is the number of points, and C 
            is the number of positional features.
        """
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C

"""
*****
MaskDecoder
*****
"""

# based on: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/mask_decoder.py
class MaskDecoder(nn.Module):
    """
    A Mask Decoder module that predicts segmentation masks from image and prompt embeddings
    using a transformer architecture.

    This model takes an image embedding, a set of prompt embeddings, and generates segmentation 
    masks, optionally predicting multiple masks. It also predicts the quality of the masks using
    an IOU (Intersection over Union) prediction head.

    Parameters:
    ----------
    transformer_dim : int
        The channel dimension of the transformer.
    transformer : nn.Module
        The transformer model used to predict masks.
    num_multimask_outputs : int, optional
        The number of masks to predict when disambiguating masks (default is 3).
    activation : Type[nn.Module], optional
        The activation function used when upscaling the masks (default is nn.GELU).
    iou_head_depth : int, optional
        The depth of the MLP used to predict mask quality (default is 3).
    iou_head_hidden_dim : int, optional
        The hidden dimension of the MLP used to predict mask quality (default is 256).

    Attributes:
    ----------
    transformer_dim : int
        The channel dimension of the transformer.
    transformer : nn.Module
        The transformer module used to predict masks.
    num_multimask_outputs : int, default=3
        The number of masks to predict when disambiguating masks.
    iou_token : nn.Embedding
        The embedding for the IOU token.
    num_mask_tokens : int
        The total number of mask tokens, including the IOU token.
    mask_tokens : nn.Embedding
        The embeddings for the mask tokens.
    output_upscaling : nn.Sequential
        A sequence of layers used to upscale the output mask embeddings.
    output_hypernetworks_mlps : nn.ModuleList
        A list of MLPs for each mask token used to generate the final mask outputs.
    iou_prediction_head : MLP
        An MLP used to predict the mask quality (IOU) for each mask token.

    Methods
    -------
    forward(image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, multimask_output)
        Predicts masks and mask quality given the image and prompt embeddings.

    predict_masks(image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings)
        Generates the segmentation masks and IOU predictions from the image and prompt embeddings.
    """

    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        # self.output_hypernetworks_mlps = nn.ModuleList(
        #     [
        #         MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        #         for i in range(self.num_mask_tokens)
        #     ]
        # )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(
                    layer_sizes=[
                        transformer_dim,  # Input layer
                        transformer_dim,  # Hidden layer 1
                        transformer_dim,  # Hidden layer 2
                        transformer_dim // 8  # Output layer
                    ],
                    activation_cls=nn.ReLU  # Define a ativação como ReLU
                )
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            layer_sizes=[
                transformer_dim,
            ]* (iou_head_depth - 1) + [iou_head_hidden_dim] + [self.num_mask_tokens], # Hidden layers e output layer
            activation_cls=nn.ReLU
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts masks and IOU scores given image and prompt embeddings.

        Parameters
        ----------
        image_embeddings : torch.Tensor
            The embeddings generated by the image encoder.
        image_pe : torch.Tensor
            Positional encodings with the same shape as image_embeddings.
        sparse_prompt_embeddings : torch.Tensor
            The embeddings corresponding to sparse prompts (e.g., points, boxes).
        dense_prompt_embeddings : torch.Tensor
            The embeddings corresponding to dense prompts (e.g., mask inputs).
        multimask_output : bool
            Whether to return multiple masks or a single mask.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            - The predicted masks of shape (batch_size, num_masks, height, width).
            - The predicted IOU scores for each mask of shape (batch_size, num_masks).
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates segmentation masks and IOU predictions given image and prompt embeddings.

        Parameters
        ----------
        image_embeddings : torch.Tensor
            The embeddings generated by the image encoder.
        image_pe : torch.Tensor
            Positional encodings with the same shape as image_embeddings.
        sparse_prompt_embeddings : torch.Tensor
            The embeddings for sparse prompts.
        dense_prompt_embeddings : torch.Tensor
            The embeddings for dense prompts.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            - The predicted masks of shape (batch_size, num_masks, height, width).
            - The predicted IOU scores of shape (batch_size, num_masks).
        """

        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred

# based on: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/mask_decoder.py
class TwoWayTransformer(nn.Module):
    """
    A Transformer decoder that attends to an input image using queries whose positional
    encoding is supplied. The model processes input embeddings through a series of transformer
    layers and a final attention mechanism that integrates point and image embeddings.

    Parameters:
    -----
        depth : int
            Number of layers in the transformer.
        embedding_dim : int
            The channel dimension for the input embeddings.
        num_heads : int
            The number of heads for multihead attention. Must divide embedding_dim.
        mlp_dim : int
            The channel dimension internal to the MLP block.
        activation : nn.Module, optional
            The activation function to use in the MLP block. Default is nn.ReLU.
        attention_downsample_rate : int, optional
            Downsampling rate for attention mechanisms. Default is 2.
    
    Attributes:
    -----------
        depth : int
            Number of transformer layers.
        embedding_dim : int
            The dimensionality of the input embeddings.
        num_heads : int
            Number of attention heads in the multihead attention mechanism. Must divide the embedding_dim evenly.
        mlp_dim : int
            The dimensionality of the MLP block inside each transformer layer.
        layers : nn.ModuleList
            A list of transformer blocks.
        final_attn_token_to_image : AttentionMaskDecoder
            Final attention mechanism to attend from points to image.
        norm_final_attn : nn.LayerNorm
            Final layer normalization applied to the query embeddings.
    """

    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = AttentionMaskDecoder(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through the transformer layers, processing image and point embeddings.

        Args:
        -----
            image_embedding : torch.Tensor
                The image embedding to attend to. Shape should be (B, embedding_dim, H, W), 
                where B is the batch size, and H, W are the spatial dimensions of the image.
            image_pe : torch.Tensor
                The positional encoding for the image, with the same shape as image_embedding.
            point_embedding : torch.Tensor
                The embedding for the query points. Shape should be (B, N_points, embedding_dim), 
                where N_points is the number of points.

        Returns:
        --------
            Tuple[torch.Tensor, torch.Tensor]:
                - The processed point embeddings after the transformer layers.
                - The processed image embeddings after the transformer layers.
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys

# based on: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/mask_decoder.py
class TwoWayAttentionBlock(nn.Module):
    """
    Two-Way Attention Transformer Block.

    This class implements a transformer block consisting of four layers:

    1. **Self-Attention**: Applied to the sparse inputs (queries).
    2. **Cross-Attention** (Tokens to Image): Sparse inputs attend to dense inputs.
    3. **MLP Block**: Applies a Multi-Layer Perceptron to the sparse inputs.
    4. **Cross-Attention** (Image to Tokens): Dense inputs attend to sparse inputs.

    The block supports positional embeddings and layer normalization after each operation.

    Parameters
    ----------
    embedding_dim : int
        The channel dimension of the embeddings.
    num_heads : int
        The number of heads in the attention layers.
    mlp_dim : int, optional, default=2048
        The hidden dimension of the MLP block.
    activation : Type[nn.Module], optional, default=nn.ReLU
        The activation function used in the MLP block.
    attention_downsample_rate : int, optional, default=2
        The downsampling rate for the cross-attention layers.
    skip_first_layer_pe : bool, optional, default=False
        If True, skips the positional embeddings in the first layer.

    Attributes
    ----------
    self_attn : AttentionMaskDecoder
        Self-attention mechanism for sparse inputs (queries).
    norm1 : nn.LayerNorm
        Layer normalization applied after self-attention.
    cross_attn_token_to_image : AttentionMaskDecoder
        Cross-attention mechanism where sparse inputs attend to dense inputs.
    norm2 : nn.LayerNorm
        Layer normalization applied after the first cross-attention.
    mlp : MLPBlock
        Multi-Layer Perceptron applied to the sparse inputs.
    norm3 : nn.LayerNorm
        Layer normalization applied after the MLP block.
    cross_attn_image_to_token : AttentionMaskDecoder
        Cross-attention mechanism where dense inputs attend to sparse inputs.
    norm4 : nn.LayerNorm
        Layer normalization applied after the second cross-attention.
    skip_first_layer_pe : bool
        Flag indicating whether to skip positional embeddings in the first layer.

    Methods
    -------
    forward(queries, keys, query_pe, key_pe)
        Computes the output of the Two-Way Attention Block.

    Examples
    --------
    >>> import torch
    >>> from torch import nn
    >>> embedding_dim = 512
    >>> num_heads = 8
    >>> block = TwoWayAttentionBlock(embedding_dim=embedding_dim, num_heads=num_heads)
    >>> queries = torch.randn(10, 16, embedding_dim)  # Batch size: 10, Tokens: 16
    >>> keys = torch.randn(10, 64, embedding_dim)    # Batch size: 10, Image tokens: 64
    >>> query_pe = torch.randn(10, 16, embedding_dim)
    >>> key_pe = torch.randn(10, 64, embedding_dim)
    >>> output_queries, output_keys = block(queries, keys, query_pe, key_pe)
    >>> output_queries.shape, output_keys.shape
    (torch.Size([10, 16, 512]), torch.Size([10, 64, 512]))
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = AttentionMaskDecoder(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = AttentionMaskDecoder(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = AttentionMaskDecoder(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for the Two-Way Attention Block.

        Parameters
        ----------
        queries : Tensor
            Sparse input tensor of shape (batch_size, num_queries, embedding_dim).
        keys : Tensor
            Dense input tensor of shape (batch_size, num_keys, embedding_dim).
        query_pe : Tensor
            Positional embeddings for the sparse inputs, same shape as `queries`.
        key_pe : Tensor
            Positional embeddings for the dense inputs, same shape as `keys`.

        Returns
        -------
        Tuple[Tensor, Tensor]
            Updated queries and keys tensors.

            - queries: Tensor of shape (batch_size, num_queries, embedding_dim)
            - keys: Tensor of shape (batch_size, num_keys, embedding_dim)
        """
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys

# based on: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/mask_decoder.py
class AttentionMaskDecoder(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.

    This class implements a multi-head self-attention mechanism with optional 
    downsampling of the embedding dimension. It projects the input tensors 
    (queries, keys, and values) into a lower-dimensional space, performs scaled 
    dot-product attention, and then projects the result back to the original 
    embedding dimension.

    Parameters
    ----------
    embedding_dim : int
        The dimension of the input embeddings.
    num_heads : int
        The number of attention heads. Must evenly divide the `internal_dim`.
    downsample_rate : int, optional
        The rate at which the embedding dimension is downscaled internally. 
        Default is 1 (no downscaling).

    Attributes
    ----------
    embedding_dim : int
        The original embedding dimension of the input.
    internal_dim : int
        The downsampled embedding dimension, calculated as 
        `embedding_dim // downsample_rate`.
    num_heads : int
        The number of attention heads.
    q_proj : nn.Linear
        Linear layer for projecting the input query tensor.
    k_proj : nn.Linear
        Linear layer for projecting the input key tensor.
    v_proj : nn.Linear
        Linear layer for projecting the input value tensor.
    out_proj : nn.Linear
        Linear layer for projecting the output tensor back to the original embedding space.

    Methods
    -------
    forward(q, k, v)
        Computes the output of the attention mechanism.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        """
        Splits the embedding into multiple attention heads.

        Parameters
        ----------
        x : Tensor
            The input tensor of shape (batch_size, num_tokens, embedding_dim).
        num_heads : int
            The number of attention heads.

        Returns
        -------
        Tensor
            Tensor with shape (batch_size, num_heads, num_tokens, head_dim), 
            where `head_dim = embedding_dim // num_heads`.
        """
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # [B, N_heads, N_tokens, C_per_head]

    def _recombine_heads(self, x: Tensor) -> Tensor:
        """
        Recombines the attention heads into a single embedding.

        Parameters
        ----------
        x : Tensor
            Tensor with shape (batch_size, num_heads, num_tokens, head_dim).

        Returns
        -------
        Tensor
            Tensor with shape (batch_size, num_tokens, embedding_dim).
        """
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # [B, N_tokens, C]

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """
        Computes the output of the attention mechanism.

        Parameters
        ----------
        q : Tensor
            Query tensor of shape (batch_size, num_tokens, embedding_dim).
        k : Tensor
            Key tensor of shape (batch_size, num_tokens, embedding_dim).
        v : Tensor
            Value tensor of shape (batch_size, num_tokens, embedding_dim).

        Returns
        -------
        Tensor
            Output tensor of shape (batch_size, num_tokens, embedding_dim).
        """
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out

"""
*****
SAM class
*****
"""

# based on: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/sam.py
class _SAM(nn.Module):
    """
    SAM predicts object masks from an image and input prompts.

    This class integrates an image encoder, prompt encoder, and mask decoder
    to process images and associated prompts, and output object masks.

    Parameters
    ----------
    image_encoder : ImageEncoderViT
        The backbone used to encode the image into embeddings for mask prediction.
    prompt_encoder : PromptEncoder
        Encodes various types of input prompts such as points, boxes, and masks.
    mask_decoder : MaskDecoder
        Predicts masks from the image embeddings and encoded prompts.
    pixel_mean : list of float, optional
        Mean values for normalizing the pixels in the input image, by default [123.675, 116.28, 103.53].
    pixel_std : list of float, optional
        Standard deviation values for normalizing the pixels in the input image, by default [58.395, 57.12, 57.375].
    mask_threshold : float, optional
        Threshold value for binarizing the masks, by default 0.0.

    Examples
    --------
    >>> model = _SAM(image_encoder, prompt_encoder, mask_decoder)
    >>> input_data = [{"image": image_tensor, "original_size": (500, 500)}]
    >>> outputs = model(input_data, multimask_output=True)
    >>> masks = outputs[0]["masks"]
    """

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
        mask_threshold: float = 0.0
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.mask_threshold = mask_threshold

    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predict masks end-to-end from images and prompts.

        Parameters
        ----------
        batched_input : list of dict
            A list of dictionaries, each containing:
            - 'image' : torch.Tensor
                The image as a 3xHxW tensor, already preprocessed.
            - 'original_size' : tuple of int
                Original size (height, width) of the image.
            - 'point_coords' : torch.Tensor, optional
                Point coordinates for prompts, shape (B, N, 2).
            - 'point_labels' : torch.Tensor, optional
                Labels for point prompts, shape (B, N).
            - 'boxes' : torch.Tensor, optional
                Box prompts, shape (B, 4).
            - 'mask_inputs' : torch.Tensor, optional
                Input masks, shape (B, 1, H, W).
        multimask_output : bool
            If True, predicts multiple disambiguating masks; otherwise,
            returns a single mask.

        Returns
        -------
        list of dict
            A list of dictionaries with the following keys:
            - 'masks' : torch.Tensor
                Binary mask predictions, shape (B, C, H, W).
            - 'iou_predictions' : torch.Tensor
                Predicted IoU scores for masks, shape (B, C).
            - 'low_res_logits' : torch.Tensor
                Low-resolution mask logits, shape (B, C, H, W).
            - 'masks_logits' : torch.Tensor
                Non-binarized mask predictions.

        Notes
        -----
        This method processes each input image and associated prompts through
        the encoders and decoder to generate masks.
        """
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            # check if point coordinates was sent
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            # apply prompt encoder
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            """
            apply mask decoder using:
                current image embedding,
                positional encoding used in prompt,
                sparse embeddings (prompt encoder),
                prompt embeddings,
                flag multimask_output (for 1 or n masks)
            """
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output
            )
            # apply postprosses in mask for get original size
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            # binaring masks with threshold
            masks_logits = masks
            masks = masks > self.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                    "masks_logits": masks_logits,
                }
            )
        return outputs
    
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize pixel values and pad input images to a square size.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor with shape (3, H, W).

        Returns
        -------
        torch.Tensor
            Preprocessed image tensor with normalized pixels and padding
            applied to match the encoder input size.
        """
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
    
    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Upscale masks to the original image size and remove padding.

        Parameters
        ----------
        masks : torch.Tensor
            Batched masks output from the decoder, shape (B, C, H, W).
        input_size : tuple of int
            Input size of the image after preprocessing, (height, width).
        original_size : tuple of int
            Original size of the image, (height, width).

        Returns
        -------
        torch.Tensor
            Resized masks, shape (B, C, H, W), where H and W correspond
            to the original image size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

"""
*****
SAM torch lightning
*****
"""

class Sam(L.LightningModule):
    """
    SAM (Segment Anything Model) predicts object masks from an image and input prompts.

    Parameters
    ----------
    image_encoder : ImageEncoderViT, optional
        The backbone used to encode the image into image embeddings that allow for efficient 
        mask prediction. Defaults to None.
    prompt_encoder : PromptEncoder, optional
        Encodes various types of input prompts (point/box). Defaults to None.
    mask_decoder : MaskDecoder, optional
        Predicts masks from the image embeddings and encoded prompts. Defaults to None.
    pixel_mean : list of float, optional
        Mean values for normalizing pixels in the input image. Defaults to [123.675, 116.28, 103.53].
    pixel_std : list of float, optional
        Standard deviation values for normalizing pixels in the input image. Defaults to [58.395, 57.12, 57.375].
    mask_threshold : float, optional
        Threshold to apply to masks for converting predictions to binary. Defaults to 0.0.
    learning_rate : float, optional
        Learning rate for the training process. Defaults to 1e-5.
    vit_type : str, optional
        Type of Vision Transformer model. Options: 'vit-b', 'vit-h', 'vit-l'. Defaults to 'vit-b'.
    checkpoint : str, optional
        Path to a model checkpoint to load pre-trained weights. Defaults to None.
    num_multimask_outputs : int, optional
        Number of multi-mask outputs to predict. Defaults to 3.
    iou_head_depth : int, optional
        Depth of the Intersection over Union (IoU) head. Defaults to 3.
    loss_fn : nn.Module, optional
        Loss function for the model. Defaults to CrossEntropyLoss if not provided.
    train_metrics : dict of Metric, optional
        Dictionary of metrics to compute during training. Defaults to None.
    val_metrics : dict of Metric, optional
        Dictionary of metrics to compute during validation. Defaults to None.
    test_metrics : dict of Metric, optional
        Dictionary of metrics to compute during testing. Defaults to None.
    apply_freeze : dict of bool, optional
        Dictionary specifying whether to freeze individual model components. Keys include 
        'image_encoder', 'prompt_encoder', and 'mask_decoder'. Defaults to freezing 'prompt_encoder'.
    apply_adapter : dict of LoRA, optional
        Dictionary containing LoRA adapters to apply to model components. Defaults to an empty dictionary.
    lora_rank : int, optional
        Rank parameter for LoRA layers. Defaults to 4.
    lora_alpha : int, optional
        Scaling factor for LoRA layers. Defaults to 1.

    Raises
    ------
    ValueError
        If `vit_type` is not one of the valid options: 'vit-b', 'vit-h', 'vit-l'.

    Examples
    --------
    >>> sam_model = Sam()
    >>> sam_model
    Sam(...)
    """

    def __init__(
            self,
            image_encoder:ImageEncoderViT=None,
            prompt_encoder:PromptEncoder=None,
            mask_decoder:MaskDecoder=None,
            pixel_mean=[123.675, 116.28, 103.53],
            pixel_std=[58.395, 57.12, 57.375],
            mask_threshold=0.0,
            learning_rate=1e-5,
            vit_type:str='vit-b',
            checkpoint:str=None,
            num_multimask_outputs:int=3,
            iou_head_depth:int=3,
            loss_fn: Optional[nn.Module] = None,
            train_metrics: Optional[Dict[str, Metric]] = None,
            val_metrics: Optional[Dict[str, Metric]] = None,
            test_metrics: Optional[Dict[str, Metric]] = None,
            apply_freeze:Optional[Dict[str, bool]] = {"image_encoder": False, "prompt_encoder": True, "mask_decoder": False},
            apply_adapter:Optional[Dict[str, LoRA]] = {},
            lora_rank:int=4,
            lora_alpha:int=1
    ):
        super().__init__()
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.mask_threshold = mask_threshold
        self.learning_rate = learning_rate
        self.num_multimask_outputs = num_multimask_outputs
        self.iou_head_depth = iou_head_depth

        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()

        self.metrics = {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        }

        valid_vit_types = {'vit-b', 'vit-h', 'vit-l'}
        if vit_type not in valid_vit_types:
            raise ValueError(f"Invalid vit_type '{vit_type}'. Choose one of {valid_vit_types}.")
        
        self.vit_type = vit_type

        self.vit_params = {
            'vit-b': {
                'vit_type': 'vit-b',
                'image_encoder': image_encoder,
                'prompt_encoder': prompt_encoder,
                'mask_decoder': mask_decoder,
                'encoder_embed_dim': 768,
                'encoder_depth': 12,
                'encoder_num_heads': 12,
                'mask_threshold': self.mask_threshold,
                'encoder_global_attn_indexes': [2, 5, 8, 11],
                'checkpoint': checkpoint,
                'num_multimask_outputs': self.num_multimask_outputs,
                'iou_head_depth': self.iou_head_depth
            },
            'vit-l': {
                'vit_type': 'vit-l',
                'image_encoder': image_encoder,
                'prompt_encoder': prompt_encoder,
                'mask_decoder': mask_decoder,
                'encoder_embed_dim': 1024,
                'encoder_depth': 24,
                'encoder_num_heads': 16,
                'mask_threshold': self.mask_threshold,
                'encoder_global_attn_indexes': [5, 11, 17, 23],
                'checkpoint': checkpoint,
                'num_multimask_outputs': self.num_multimask_outputs,
                'iou_head_depth': self.iou_head_depth
            },
            'vit-h': {
                'vit_type': 'vit-h',
                'image_encoder': image_encoder,
                'prompt_encoder': prompt_encoder,
                'mask_decoder': mask_decoder,
                'encoder_embed_dim': 1280,
                'encoder_depth': 32,
                'encoder_num_heads': 16,
                'mask_threshold': self.mask_threshold,
                'encoder_global_attn_indexes': [7, 15, 23, 31],
                'checkpoint': checkpoint,
                'num_multimask_outputs': self.num_multimask_outputs,
                'iou_head_depth': self.iou_head_depth
            },
        }

        self.model = self._build_sam(**self.vit_params[self.vit_type])

        self._apply_freeze(apply_freeze)
        self._apply_adapter(apply_adapter, alpha=lora_alpha, rank=lora_rank)
    
    def _apply_freeze(self, apply_freeze):
        """
        Freezes specified components of the model.

        Parameters
        ----------
        apply_freeze : dict of bool
            Dictionary specifying components to freeze. Keys include 'image_encoder', 
            'prompt_encoder', and 'mask_decoder'.
        """
        if 'image_encoder' in apply_freeze and apply_freeze['image_encoder'] == True:
            print("Image Encoder freeze!")
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False
        if 'prompt_encoder' in apply_freeze and apply_freeze['prompt_encoder'] == True:
            print("Prompt Encoder freeze!")
            for param in self.model.prompt_encoder.parameters():
                param.requires_grad = False
        if 'mask_decoder' in apply_freeze and apply_freeze['mask_decoder'] == True:
            print("Mask Decoder freeze!")
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = False
        
    def _apply_adapter(self, apply_adapter, alpha=1, rank=4):
        """
        Applies LoRA adapters to specified model components.

        Parameters
        ----------
        apply_adapter : dict of LoRA
            Dictionary containing LoRA adapters to apply.
        alpha : int, optional
            Scaling factor for LoRA layers. Defaults to 1.
        rank : int, optional
            Rank parameter for LoRA layers. Defaults to 4.
        """
        if 'image_encoder' in apply_adapter:
            print("LoRA applied in Image Encoder!")
            for layer in self.model.image_encoder.blocks:
                layer.attn.qkv = apply_adapter['image_encoder'](original_module=layer.attn.qkv, bias=True, alpha=alpha, r=rank)
        if 'mask_decoder' in apply_adapter:
            print("LoRA applied in Mask Decoder!")
            for layer in self.model.mask_decoder.transformer.layers:
                layer.self_attn.q_proj = apply_adapter['mask_decoder'](original_module=layer.self_attn.q_proj, bias=True, alpha=alpha, r=rank)
                layer.self_attn.v_proj = apply_adapter['mask_decoder'](original_module=layer.self_attn.v_proj, bias=True, alpha=alpha, r=rank)
                layer.cross_attn_token_to_image.q_proj = apply_adapter['mask_decoder'](original_module=layer.cross_attn_token_to_image.q_proj, bias=True, alpha=alpha, r=rank)
                layer.cross_attn_token_to_image.v_proj = apply_adapter['mask_decoder'](original_module=layer.cross_attn_token_to_image.v_proj, bias=True, alpha=alpha, r=rank)
    
    def _build_sam(
        self,
        vit_type,
        image_encoder,
        prompt_encoder,
        mask_decoder,
        encoder_embed_dim,
        encoder_depth,
        encoder_num_heads,
        mask_threshold,
        encoder_global_attn_indexes,
        num_multimask_outputs=3, # classes
        iou_head_depth=3, # classes
        checkpoint=None,
    ):
        """
        Builds the SAM model with the specified components and configurations.

        Parameters
        ----------
        vit_type : str
            Type of Vision Transformer model (e.g., 'vit-b', 'vit-h', 'vit-l').
        image_encoder : ImageEncoderViT
            Image encoder backbone.
        prompt_encoder : PromptEncoder
            Encoder for input prompts.
        mask_decoder : MaskDecoder
            Decoder for mask prediction.
        encoder_embed_dim : int
            Embedding dimension of the image encoder.
        encoder_depth : int
            Depth of the Vision Transformer encoder.
        encoder_num_heads : int
            Number of attention heads in the Vision Transformer encoder.
        mask_threshold : float
            Threshold to apply to predicted masks.
        encoder_global_attn_indexes : list of int
            Indexes specifying where global attention is applied in the encoder.
        num_multimask_outputs : int, optional
            Number of multi-mask outputs to predict. Defaults to 3.
        iou_head_depth : int, optional
            Depth of the IoU head. Defaults to 3.
        checkpoint : str, optional
            Path to a checkpoint for loading pre-trained weights.

        Returns
        -------
        _SAM
            An initialized SAM model.
        """
        prompt_embed_dim = 256
        image_size = 1024
        vit_patch_size = 16
        image_embedding_size = image_size // vit_patch_size
        sam = _SAM(
            mask_threshold=mask_threshold,
            image_encoder=image_encoder or ImageEncoderViT(
                depth=encoder_depth,
                embed_dim=encoder_embed_dim,
                img_size=image_size,
                mlp_ratio=4,
                norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                num_heads=encoder_num_heads,
                patch_size=vit_patch_size,
                qkv_bias=True,
                use_rel_pos=True,
                global_attn_indexes=encoder_global_attn_indexes,
                window_size=14,
                out_chans=prompt_embed_dim,
            ),
            prompt_encoder=prompt_encoder or PromptEncoder(
                embed_dim=prompt_embed_dim,
                image_embedding_size=(image_embedding_size, image_embedding_size),
                input_image_size=(image_size, image_size),
                mask_in_chans=16,
            ),
            mask_decoder=mask_decoder or MaskDecoder(
                num_multimask_outputs=num_multimask_outputs, # default = 3 classes
                iou_head_depth=iou_head_depth, # default = 3 classes
                transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_hidden_dim=256,
            ),
            pixel_mean=[123.675, 116.28, 103.53],
            pixel_std=[58.395, 57.12, 57.375],
        )
        if checkpoint is not None:
            with open(checkpoint, "rb") as f:
                state_dict = torch.load(f)
            try:
                sam.load_state_dict(state_dict)
            except:
                print("Error when load original weights. Applying now remaping.")
                if vit_type == 'vit-b':
                    new_state_dict = self.load_from_b(sam, state_dict, image_size, vit_patch_size) # using remaping for vit_b
                elif vit_type == 'vit-h':
                    new_state_dict = self.load_from_h(sam, state_dict, image_size, vit_patch_size, encoder_global_attn_indexes) # using remaping for vit_h
                sam.load_state_dict(new_state_dict)
        return sam

    # mapping weights: developed by https://github.com/hitachinsk/SAMed/blob/main/segment_anything/build_sam.py
    def load_from_b(self, sam, state_dict, image_size, vit_patch_size):
        """
        Loads and maps weights for the 'vit-b' model variant.

        Parameters
        ----------
        sam : _SAM
            The SAM model instance.
        state_dict : dict
            State dictionary containing pre-trained weights.
        image_size : int
            Size of the input image.
        vit_patch_size : int
            Patch size for Vision Transformer.

        Returns
        -------
        dict
            Remapped state dictionary compatible with the SAM model.
        """
        sam_dict = sam.state_dict()
        except_keys = ['mask_tokens', 'output_hypernetworks_mlps', 'iou_prediction_head']
        new_state_dict = {k: v for k, v in state_dict.items() if
                        k in sam_dict.keys() and except_keys[0] not in k and except_keys[1] not in k and except_keys[2] not in k}
        pos_embed = new_state_dict['image_encoder.pos_embed']
        token_size = int(image_size // vit_patch_size)
        if pos_embed.shape[1] != token_size:
            # resize pos embedding, which may sacrifice the performance, but I have no better idea
            pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
            pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
            pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
            new_state_dict['image_encoder.pos_embed'] = pos_embed
            rel_pos_keys = [k for k in sam_dict.keys() if 'rel_pos' in k]
            global_rel_pos_keys = [k for k in rel_pos_keys if '2' in k or '5' in  k or '8' in k or '11' in k]
            for k in global_rel_pos_keys:
                rel_pos_params = new_state_dict[k]
                h, w = rel_pos_params.shape
                rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
                rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
                new_state_dict[k] = rel_pos_params[0, 0, ...]
        sam_dict.update(new_state_dict)
        return sam_dict
    
    # mapping weights: developed by https://github.com/hitachinsk/SAMed/blob/main/SAMed_h/segment_anything/build_sam.py
    def load_from_h(self, sam, state_dict, image_size, vit_patch_size, encoder_global_attn_indexes):
        """
        Loads and remaps the pretrained weights for 'vit-h' model.

        Parameters
        ----------
        sam : torch.nn.Module
            The SAM model instance to load the weights into.
        state_dict : dict
            The state dictionary containing the pretrained weights.
        image_size : int
            The size of the input image.
        vit_patch_size : int
            The size of the patches used by the Vision Transformer.
        encoder_global_attn_indexes : list of int
            The indexes indicating which layers use global attention in the encoder.

        Returns
        -------
        dict
            The remapped state dictionary ready to be loaded into the SAM model.
        """
        ega = encoder_global_attn_indexes
        sam_dict = sam.state_dict()
        except_keys = ['mask_tokens', 'output_hypernetworks_mlps', 'iou_prediction_head']
        new_state_dict = {k: v for k, v in state_dict.items() if
                        k in sam_dict.keys() and except_keys[0] not in k and except_keys[1] not in k and except_keys[2] not in k}
        pos_embed = new_state_dict['image_encoder.pos_embed']
        token_size = int(image_size // vit_patch_size)
        if pos_embed.shape[1] != token_size:
            # resize pos embedding, which may sacrifice the performance, but I have no better idea
            pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
            pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
            pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
            new_state_dict['image_encoder.pos_embed'] = pos_embed
            rel_pos_keys = [k for k in sam_dict.keys() if 'rel_pos' in k]
            global_rel_pos_keys = []
            for rel_pos_key in rel_pos_keys:
                num = int(rel_pos_key.split('.')[2])
                if num in encoder_global_attn_indexes:
                    global_rel_pos_keys.append(rel_pos_key)
            for k in global_rel_pos_keys:
                rel_pos_params = new_state_dict[k]
                h, w = rel_pos_params.shape
                rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
                rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
                new_state_dict[k] = rel_pos_params[0, 0, ...]
        sam_dict.update(new_state_dict)
        return sam_dict

    def forward(self, batched_input:List[Dict[str, Any]], multimask_output:bool) -> torch.Tensor:
        return self.model(batched_input, multimask_output)
    
    def _compute_metrics(self, y_hat: torch.Tensor, y: torch.Tensor, step_name: str):
        if self.metrics[step_name] is None:
            return {}

        return {
            f"{step_name}_{metric_name}": metric.to(self.device)(
                torch.argmax(y_hat, dim=1, keepdim=True).squeeze(1), y
            )
            for metric_name, metric in self.metrics[step_name].items()
        }
    
    def _single_step(self, batch, batch_idx:int, step_name:str):
        batched_input = batch
        outputs = self(batched_input, multimask_output=batched_input[0]['multimask_output'])

        # stack logits 'masks_logits' and 'labels' for loss and metrics function
        masks_logits = torch.stack([output['masks_logits'].squeeze(0) for output in outputs])  # [batch_size, num_classes, H, W]
        labels = torch.stack([input['label'].squeeze(0) for input in batched_input])  # [batch_size, H, W]

        loss = self._loss(masks_logits, labels)
        metrics = self._compute_metrics(masks_logits, labels, step_name)
        
        self.log(f"{step_name}_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        for metric_name, metric_value in metrics.items():
            self.log(metric_name, metric_value.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        return self._single_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._single_step(batch, batch_idx, "val")
    
    def test_step(self, batch: torch.Tensor, batch_idx: int):
        return self._single_step(batch, batch_idx, "test")

    def predict_step(self, batch: torch.Tensor, batch_idx: int, dataloader_idx: Optional[int] = None):
        batched_input = batch
        
        outputs = self(batched_input, multimask_output=batched_input[0]['multimask_output'])
        return outputs
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def _loss(self, masks, label):
        loss_ce = self.loss_fn(masks, label[:].long())
        return loss_ce