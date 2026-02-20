import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final

from timm.layers import (
    Mlp,
    DropPath,
    AttentionPoolLatent,
    PatchDropout,
    trunc_normal_,
    lecun_normal_,
    resample_patch_embed,
    resample_abs_pos_embed,
    use_fused_attn,
    get_act_layer,
    get_norm_layer,
    LayerType,
)

from timm.models._manipulate import named_apply, checkpoint_seq, adapt_input_conv
from .patch_embed import PatchEmbed


class Attention(nn.Module):
    """
    Multi-head self-attention module.

    This class implements the standard multi-head attention mechanism used in
    Transformer architectures. It supports both standard and fused attention
    implementations for improved performance when available.
    """

    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
    ):
        """
        Initialize the Attention module.

        Parameters
        ----------
        dim : int
            Total dimension of the input and output features.
        num_heads : int, default=8
            Number of attention heads.
        qkv_bias : bool, default=False
            If True, add a bias term to the query, key, and value projections.
        qk_norm : bool, default=False
            If True, apply normalization to query and key tensors.
        proj_bias : bool, default=True
            If True, include bias in the output projection layer.
        attn_drop : float, default=0.0
            Dropout rate applied to the attention weights.
        proj_drop : float, default=0.0
            Dropout rate applied after the output projection.
        norm_layer : Type[nn.Module], default=nn.LayerNorm
            Normalization layer type applied to query and key vectors when `qk_norm=True`.
        """
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the multi-head attention mechanism.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, N, C), where
            B is the batch size, N is the sequence length, and C is the feature dimension.

        Returns
        -------
        torch.Tensor
            Output tensor of the same shape as input (B, N, C),
            containing the attended feature representations.
        """
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    """LayerScale module."""

    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5,
        inplace: bool = False,
    ):
        """
        Initialize the LayerScale module.

        Parameters
        ----------
        dim : int
            Number of feature dimensions (channels) to scale.
        init_values : float, default=1e-5
            Initial value for the learnable scaling parameter.
        inplace : bool, default=False
            If True, performs the scaling operation in-place to save memory.
        """
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applying per-channel scaling to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, N, C) or (B, C, H, W), depending on context.

        Returns
        -------
        torch.Tensor
            Scaled tensor of the same shape as the input.
        """
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):
    """Transformer block module."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_bias: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        mlp_layer: Type[nn.Module] = Mlp,
    ):
        """
        Initialize the Transformer block.

        Parameters
        ----------
        dim : int
            Embedding dimension of the input and output features.
        num_heads : int
            Number of attention heads in the self-attention layer.
        mlp_ratio : float, default=4.0
            Expansion ratio for the hidden dimension in the MLP layer.
        qkv_bias : bool, default=False
            If True, add bias to the query, key, and value projections.
        qk_norm : bool, default=False
            If True, apply normalization to query and key tensors.
        proj_bias : bool, default=True
            If True, include bias in the projection layers.
        proj_drop : float, default=0.0
            Dropout rate applied to the output of the attention and MLP layers.
        attn_drop : float, default=0.0
            Dropout rate applied to the attention weights.
        init_values : float, optional
            If specified, enables LayerScale with this initial scaling value.
        drop_path : float, default=0.0
            Stochastic depth rate; set > 0 to apply DropPath regularization.
        act_layer : Type[nn.Module], default=nn.GELU
            Activation function used in the MLP layer.
        norm_layer : Type[nn.Module], default=nn.LayerNorm
            Normalization layer type applied before attention and MLP.
        mlp_layer : Type[nn.Module], default=Mlp
            Module type used for the feed-forward network.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            bias=proj_bias,
            drop=proj_drop,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class ResPostBlock(nn.Module):
    """Residual Post-Norm Transformer block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_bias: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        mlp_layer: Type[nn.Module] = Mlp,
    ):
        super().__init__()
        """
        Initialize the Residual Post-Norm Transformer block.

        Parameters
        ----------
        dim : int
            Embedding dimension of the input and output features.
        num_heads : int
            Number of attention heads in the self-attention layer.
        mlp_ratio : float, default=4.0
            Expansion ratio for the hidden dimension in the MLP layer.
        qkv_bias : bool, default=False
            If True, add bias to the query, key, and value projections.
        qk_norm : bool, default=False
            If True, apply normalization to query and key tensors.
        proj_bias : bool, default=True
            If True, include bias in the projection layers.
        proj_drop : float, default=0.0
            Dropout rate applied to the output of the attention and MLP layers.
        attn_drop : float, default=0.0
            Dropout rate applied to the attention weights.
        init_values : float, optional
            If specified, initializes normalization layer weights with this constant.
        drop_path : float, default=0.0
            Stochastic depth rate; set > 0 to apply DropPath regularization.
        act_layer : Type[nn.Module], default=nn.GELU
            Activation function used in the MLP layer.
        norm_layer : Type[nn.Module], default=nn.LayerNorm
            Normalization layer type applied after attention and MLP.
        mlp_layer : Type[nn.Module], default=Mlp
            Module type used for the feed-forward network.
        """
        self.init_values = init_values

        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.norm1 = norm_layer(dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            bias=proj_bias,
            drop=proj_drop,
        )
        self.norm2 = norm_layer(dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.init_weights()

    def init_weights(self) -> None:
        # NOTE this init overrides that base model init with specific changes for the block type
        if self.init_values is not None:
            nn.init.constant_(self.norm1.weight, self.init_values)
            nn.init.constant_(self.norm2.weight, self.init_values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Residual Post-Norm Transformer block.

        The input tensor passes through attention and MLP sublayers, each followed
        by normalization and residual connections. DropPath is optionally applied
        for regularization.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, N, C), where
            B is batch size, N is sequence length, and C is embedding dimension.

        Returns
        -------
        torch.Tensor
            Output tensor of the same shape (B, N, C), representing the transformed features.
        """
        x = x + self.drop_path1(self.norm1(self.attn(x)))
        x = x + self.drop_path2(self.norm2(self.mlp(x)))
        return x


class ParallelScalingBlock(nn.Module):
    """
    Parallel Scaling Vision Transformer block.

    This module implements a parallel Transformer block that computes the
    multi-head self-attention and MLP branches concurrently and then combines
    their outputs. The design follows the architecture from
    "Scaling Vision Transformers to 22 Billion Parameters"
    (https://arxiv.org/abs/2302.05442).

    The block includes LayerScale for stable deep scaling, optional DropPath for
    stochastic depth regularization, and supports fused attention when available
    for performance efficiency.
    """

    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_bias: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        mlp_layer: Optional[Type[nn.Module]] = None,
    ):
        """
        Initialize the ParallelScalingBlock.

        Parameters
        ----------
        dim : int
            Embedding dimension of the input and output features.
        num_heads : int
            Number of attention heads in the multi-head self-attention layer.
        mlp_ratio : float, default=4.0
            Expansion ratio for the hidden dimension in the MLP branch.
        qkv_bias : bool, default=False
            If True, add bias to the query, key, and value projections.
        qk_norm : bool, default=False
            If True, apply normalization to the query and key tensors.
        proj_bias : bool, default=True
            If True, include bias in the output projection layers.
        proj_drop : float, default=0.0
            Dropout rate applied after the projection layers.
        attn_drop : float, default=0.0
            Dropout rate applied to the attention weights.
        init_values : float, optional
            If specified, enables LayerScale with this initialization value.
        drop_path : float, default=0.0
            Stochastic depth rate; set > 0 to apply DropPath regularization.
        act_layer : Type[nn.Module], default=nn.GELU
            Activation function used in the MLP branch.
        norm_layer : Type[nn.Module], default=nn.LayerNorm
            Normalization layer applied before the parallel branches.
        mlp_layer : Type[nn.Module], optional
            Optional custom MLP implementation; defaults to a standard linear MLP.
        """
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = use_fused_attn()
        mlp_hidden_dim = int(mlp_ratio * dim)
        in_proj_out_dim = mlp_hidden_dim + 3 * dim

        self.in_norm = norm_layer(dim)
        self.in_proj = nn.Linear(dim, in_proj_out_dim, bias=qkv_bias)
        self.in_split = [mlp_hidden_dim] + [dim] * 3
        if qkv_bias:
            self.register_buffer("qkv_bias", None)
            self.register_parameter("mlp_bias", None)
        else:
            self.register_buffer("qkv_bias", torch.zeros(3 * dim), persistent=False)
            self.mlp_bias = nn.Parameter(torch.zeros(mlp_hidden_dim))

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_out_proj = nn.Linear(dim, dim, bias=proj_bias)

        self.mlp_drop = nn.Dropout(proj_drop)
        self.mlp_act = act_layer()
        self.mlp_out_proj = nn.Linear(mlp_hidden_dim, dim, bias=proj_bias)

        self.ls = (
            LayerScale(dim, init_values=init_values)
            if init_values is not None
            else nn.Identity()
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Parallel Scaling Transformer block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, N, C), where
            B is batch size, N is sequence length, and C is embedding dimension.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, N, C), containing the updated feature representations.
        """
        B, N, C = x.shape

        # Combined MLP fc1 & qkv projections
        y = self.in_norm(x)
        if self.mlp_bias is not None:
            # Concat constant zero-bias for qkv w/ trainable mlp_bias.
            # Appears faster than adding to x_mlp separately
            y = F.linear(
                y, self.in_proj.weight, torch.cat((self.qkv_bias, self.mlp_bias))
            )
        else:
            y = self.in_proj(y)
        x_mlp, q, k, v = torch.split(y, self.in_split, dim=-1)

        # Dot product attention w/ qk norm
        q = self.q_norm(q.view(B, N, self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(k.view(B, N, self.num_heads, self.head_dim)).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        if self.fused_attn:
            x_attn = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x_attn = attn @ v
        x_attn = x_attn.transpose(1, 2).reshape(B, N, C)
        x_attn = self.attn_out_proj(x_attn)

        # MLP activation, dropout, fc2
        x_mlp = self.mlp_act(x_mlp)
        x_mlp = self.mlp_drop(x_mlp)
        x_mlp = self.mlp_out_proj(x_mlp)

        # Add residual w/ drop path & layer scale applied
        y = self.drop_path(self.ls(x_attn + x_mlp))
        x = x + y
        return x


class ParallelThingsBlock(nn.Module):
    """
    Parallel Things Vision Transformer block.

    This module implements a Transformer block that processes the input through
    multiple parallel attention layers followed by multiple parallel MLP layers.
    The outputs of each parallel branch are summed together, enabling a richer
    representation and improved learning capacity.

    The design follows the architecture from
    "Three Things Everyone Should Know About Vision Transformers"
    (https://arxiv.org/abs/2203.09795).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_parallel: int = 2,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_bias: bool = True,
        init_values: Optional[float] = None,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        mlp_layer: Type[nn.Module] = Mlp,
    ) -> None:
        """
        Initialize the ParallelThingsBlock.

        Parameters
        ----------
        dim : int
            Embedding dimension of the input and output features.
        num_heads : int
            Number of attention heads in each attention branch.
        num_parallel : int, default=2
            Number of parallel attention and MLP branches.
        mlp_ratio : float, default=4.0
            Expansion ratio for the hidden dimension in the MLP layers.
        qkv_bias : bool, default=False
            If True, add bias to the query, key, and value projections.
        qk_norm : bool, default=False
            If True, apply normalization to query and key tensors.
        proj_bias : bool, default=True
            If True, include bias in the projection layers.
        init_values : float, optional
            If specified, enables LayerScale with this initialization value.
        proj_drop : float, default=0.0
            Dropout rate applied to the output of the projection layers.
        attn_drop : float, default=0.0
            Dropout rate applied to the attention weights.
        drop_path : float, default=0.0
            Stochastic depth rate; set > 0 to apply DropPath regularization.
        act_layer : Type[nn.Module], default=nn.GELU
            Activation function used in the MLP layers.
        norm_layer : Type[nn.Module], default=nn.LayerNorm
            Normalization layer type applied in each sub-block.
        mlp_layer : Type[nn.Module], default=Mlp
            Module type used for the feed-forward MLP networks.
        """
        super().__init__()
        self.num_parallel = num_parallel
        self.attns = nn.ModuleList()
        self.ffns = nn.ModuleList()
        for _ in range(num_parallel):
            self.attns.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            ("norm", norm_layer(dim)),
                            (
                                "attn",
                                Attention(
                                    dim,
                                    num_heads=num_heads,
                                    qkv_bias=qkv_bias,
                                    qk_norm=qk_norm,
                                    proj_bias=proj_bias,
                                    attn_drop=attn_drop,
                                    proj_drop=proj_drop,
                                    norm_layer=norm_layer,
                                ),
                            ),
                            (
                                "ls",
                                (
                                    LayerScale(dim, init_values=init_values)
                                    if init_values
                                    else nn.Identity()
                                ),
                            ),
                            (
                                "drop_path",
                                (
                                    DropPath(drop_path)
                                    if drop_path > 0.0
                                    else nn.Identity()
                                ),
                            ),
                        ]
                    )
                )
            )
            self.ffns.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            ("norm", norm_layer(dim)),
                            (
                                "mlp",
                                mlp_layer(
                                    dim,
                                    hidden_features=int(dim * mlp_ratio),
                                    act_layer=act_layer,
                                    bias=proj_bias,
                                    drop=proj_drop,
                                ),
                            ),
                            (
                                "ls",
                                (
                                    LayerScale(dim, init_values=init_values)
                                    if init_values
                                    else nn.Identity()
                                ),
                            ),
                            (
                                "drop_path",
                                (
                                    DropPath(drop_path)
                                    if drop_path > 0.0
                                    else nn.Identity()
                                ),
                            ),
                        ]
                    )
                )
            )

    def _forward_jit(self, x: torch.Tensor) -> torch.Tensor:
        x = x + torch.stack([attn(x) for attn in self.attns]).sum(dim=0)
        x = x + torch.stack([ffn(x) for ffn in self.ffns]).sum(dim=0)
        return x

    @torch.jit.ignore
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + sum(attn(x) for attn in self.attns)
        x = x + sum(ffn(x) for ffn in self.ffns)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ParallelThingsBlock.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, N, C).

        Returns
        -------
        torch.Tensor
            Output tensor of the same shape (B, N, C), representing
            the combined outputs from the parallel attention and MLP branches.
        """
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            return self._forward_jit(x)
        else:
            return self._forward(x)


def global_pool_nlc(
    x: torch.Tensor,
    pool_type: str = "token",
    num_prefix_tokens: int = 1,
    reduce_include_prefix: bool = False,
):
    if not pool_type:
        return x

    if pool_type == "token":
        x = x[:, 0]  # class token
    else:
        x = x if reduce_include_prefix else x[:, num_prefix_tokens:]
        if pool_type == "avg":
            x = x.mean(dim=1)
        elif pool_type == "avgmax":
            x = 0.5 * (x.amax(dim=1) + x.mean(dim=1))
        elif pool_type == "max":
            x = x.amax(dim=1)
        else:
            assert not pool_type, f"Unknown pool type {pool_type}"

    return x


class VisionTransformer(nn.Module):
    """Vision Transformer (ViT)

    A PyTorch implementation of the Vision Transformer architecture from
    *"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"*
    (https://arxiv.org/abs/2010.11929).

    This model divides an input image into fixed-size patches, embeds them,
    adds positional information, and processes them through a sequence of
    Transformer encoder blocks to learn global image representations.
    """

    dynamic_img_size: Final[bool]

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        global_pool: Literal["", "avg", "avgmax", "max", "token", "map"] = "token",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        proj_bias: bool = True,
        init_values: Optional[float] = None,
        class_token: bool = True,
        pos_embed: str = "learn",
        no_embed_class: bool = False,
        reg_tokens: int = 0,
        pre_norm: bool = False,
        final_norm: bool = True,
        fc_norm: Optional[bool] = None,
        dynamic_img_size: bool = False,
        dynamic_img_pad: bool = False,
        pos_drop_rate: float = 0.0,
        patch_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        weight_init: Literal["skip", "jax", "jax_nlhb", "moco", ""] = "",
        fix_init: bool = False,
        embed_norm_layer: Optional[LayerType] = None,
        norm_layer: Optional[LayerType] = None,
        act_layer: Optional[LayerType] = None,
        block_fn: Type[nn.Module] = Block,
        mlp_layer: Type[nn.Module] = Mlp,
    ):
        """
        Initialize the Vision Transformer model.

        Parameters
        ----------
        img_size : int or tuple of int, default=224
            Input image size (height, width).
        patch_size : int or tuple of int, default=16
            Size of each image patch.
        in_chans : int, default=3
            Number of input channels (e.g., 3 for RGB images).
        num_classes : int, default=1000
            Number of output classes for classification.
        global_pool : {'', 'avg', 'avgmax', 'max', 'token', 'map'}, default='token'
            Type of global pooling applied to obtain the final representation.
        embed_dim : int, default=768
            Dimension of the patch embeddings.
        depth : int, default=12
            Number of Transformer encoder blocks.
        num_heads : int, default=12
            Number of attention heads per block.
        mlp_ratio : float, default=4.0
            Expansion ratio for the MLP hidden dimension.
        qkv_bias : bool, default=True
            If True, include bias in the query, key, and value projections.
        qk_norm : bool, default=False
            If True, apply normalization to query and key vectors.
        proj_bias : bool, default=True
            If True, include bias in projection layers.
        init_values : float, optional
            Initial value for LayerScale; if None, LayerScale is disabled.
        class_token : bool, default=True
            If True, use a learnable class token.
        pos_embed : {'', 'none', 'learn'}, default='learn'
            Type of positional embedding; 'learn' enables learnable embeddings.
        no_embed_class : bool, default=False
            If True, exclude class and reg tokens from position embedding.
        reg_tokens : int, default=0
            Number of auxiliary regression tokens.
        pre_norm : bool, default=False
            If True, apply normalization before Transformer blocks.
        final_norm : bool, default=True
            If True, apply final layer normalization after all blocks.
        fc_norm : bool, optional
            Whether to normalize before the classifier head.
        dynamic_img_size : bool, default=False
            If True, enables dynamic image resizing during inference.
        dynamic_img_pad : bool, default=False
            If True, apply padding to dynamically sized images.
        drop_rate : float, default=0.0
            Dropout rate applied globally.
        pos_drop_rate : float, default=0.0
            Dropout rate applied to positional embeddings.
        patch_drop_rate : float, default=0.0
            Probability of randomly dropping patch tokens during training.
        proj_drop_rate : float, default=0.0
            Dropout rate applied to projection layers.
        attn_drop_rate : float, default=0.0
            Dropout rate applied to attention weights.
        drop_path_rate : float, default=0.0
            Stochastic depth drop rate across layers.
        weight_init : {'skip', 'jax', 'jax_nlhb', 'moco', ''}, default=''
            Weight initialization strategy.
        fix_init : bool, default=False
            If True, rescales initialization following original ViT heuristics.
        embed_norm_layer : nn.Module, optional
            Normalization layer applied to embeddings.
        norm_layer : nn.Module, optional
            Normalization layer applied to Transformer blocks.
        act_layer : nn.Module, optional
            Activation function used in MLP layers.
        block_fn : nn.Module, default=Block
            Type of Transformer block used.
        mlp_layer : nn.Module, default=Mlp
            Type of MLP module used in each block.
        """
        super().__init__()
        assert global_pool in ("", "avg", "avgmax", "max", "token", "map")
        assert class_token or global_pool != "token"
        assert pos_embed in ("", "none", "learn")
        use_fc_norm = (
            global_pool in ("avg", "avgmax", "max") if fc_norm is None else fc_norm
        )
        norm_layer = get_norm_layer(norm_layer) or partial(nn.LayerNorm, eps=1e-6)
        embed_norm_layer = get_norm_layer(embed_norm_layer)
        act_layer = get_act_layer(act_layer) or nn.GELU

        self.global_pool = global_pool
        self.num_features = self.head_hidden_size = self.embed_dim = (
            embed_dim  # for consistency with other models
        )
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.num_reg_tokens = reg_tokens
        self.has_class_token = class_token
        self.no_embed_class = (
            no_embed_class  # don't embed prefix positions (includes reg)
        )
        self.dynamic_img_size = dynamic_img_size
        self.grad_checkpointing = False

        embed_args = {}
        if dynamic_img_size:
            # flatten deferred until after pos embed
            embed_args.update(dict(strict_img_size=False, output_fmt="NHWC"))
        if embed_norm_layer is not None:
            embed_args["norm_layer"] = embed_norm_layer
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            dynamic_img_pad=dynamic_img_pad,
            **embed_args,
        )
        self.patch_size = self.patch_embed.patch_size
        self.in_channels = in_chans
        num_patches = self.patch_embed.num_patches
        reduction = (
            self.patch_embed.feat_ratio()
            if hasattr(self.patch_embed, "feat_ratio")
            else patch_size
        )

        self.cls_token = (
            nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        )
        self.reg_token = (
            nn.Parameter(torch.zeros(1, reg_tokens, embed_dim)) if reg_tokens else None
        )
        embed_len = (
            num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        )
        if not pos_embed or pos_embed == "none":
            self.pos_embed = None
        else:
            self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.Sequential(
            *[
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    proj_bias=proj_bias,
                    init_values=init_values,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    mlp_layer=mlp_layer,
                )
                for i in range(depth)
            ]
        )
        self.feature_info = [
            dict(module=f"blocks.{i}", num_chs=embed_dim, reduction=reduction)
            for i in range(depth)
        ]
        self.norm = (
            norm_layer(embed_dim) if final_norm and not use_fc_norm else nn.Identity()
        )

        # Classifier Head
        if global_pool == "map":
            self.attn_pool = AttentionPoolLatent(
                self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
                act_layer=act_layer,
            )
        else:
            self.attn_pool = None

        if weight_init != "skip":
            self.init_weights(weight_init)
        if fix_init:
            self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, _layer_id):
            param.div_(math.sqrt(2.0 * _layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def init_weights(self, mode: str = "") -> None:
        assert mode in ("jax", "jax_nlhb", "moco", "")
        head_bias = -math.log(self.num_classes) if "nlhb" in mode else 0.0
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        if self.reg_token is not None:
            nn.init.normal_(self.reg_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)

    def _init_weights(self, m: nn.Module) -> None:
        # this fn left here for compat with downstream users
        init_weights_vit_timm(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path: str, prefix: str = "") -> None:
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        return {"pos_embed", "cls_token", "dist_token"}

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict:
        return dict(
            stem=r"^cls_token|pos_embed|patch_embed",  # stem and embed
            blocks=[(r"^blocks\.(\d+)", None), (r"^norm", (99999,))],
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing = enable
        if hasattr(self.patch_embed, "set_grad_checkpointing"):
            self.patch_embed.set_grad_checkpointing(enable)

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ("", "avg", "avgmax", "max", "token", "map")
            if global_pool == "map" and self.attn_pool is None:
                assert (
                    False
                ), "Cannot currently add attention pooling in reset_classifier()."
            elif global_pool != "map" and self.attn_pool is not None:
                self.attn_pool = None  # remove attention pooling
            self.global_pool = global_pool
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def set_input_size(
        self,
        img_size: Optional[Tuple[int, int]] = None,
        patch_size: Optional[Tuple[int, int]] = None,
    ):
        """Method updates the input image resolution, patch size

        Args:
            img_size: New input resolution, if None current resolution is used
            patch_size: New patch size, if None existing patch size is used
        """
        prev_grid_size = self.patch_embed.grid_size
        self.patch_embed.set_input_size(img_size=img_size, patch_size=patch_size)
        if self.pos_embed is not None:
            num_prefix_tokens = 0 if self.no_embed_class else self.num_prefix_tokens
            num_new_tokens = self.patch_embed.num_patches + num_prefix_tokens
            if num_new_tokens != self.pos_embed.shape[1]:
                self.pos_embed = nn.Parameter(
                    resample_abs_pos_embed(
                        self.pos_embed,
                        new_size=self.patch_embed.grid_size,
                        old_size=prev_grid_size,
                        num_prefix_tokens=num_prefix_tokens,
                        verbose=True,
                    )
                )

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        if self.pos_embed is None:
            return x.view(x.shape[0], -1, x.shape[-1])

        if self.dynamic_img_size:
            B, H, W, C = x.shape
            prev_grid_size = self.patch_embed.grid_size
            pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                new_size=(H, W),
                old_size=prev_grid_size,
                num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.pos_embed

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            x = x + pos_embed

        return self.pos_drop(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Vision Transformer.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (B, C, H, W), where
            B is batch size, C is number of channels, and H, W are image dimensions.

        Returns
        -------
        torch.Tensor
            Encoded features of shape (B, N, D), where
            N is the number of patches (plus any prefix tokens) and
            D is the embedding dimension.
        """
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x


def init_weights_vit_timm(module: nn.Module, name: str = "") -> None:
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, "init_weights"):
        module.init_weights()


def init_weights_vit_jax(
    module: nn.Module, name: str = "", head_bias: float = 0.0
) -> None:
    """ViT weight initialization, matching JAX (Flax) impl"""
    if isinstance(module, nn.Linear):
        if name.startswith("head"):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                (
                    nn.init.normal_(module.bias, std=1e-6)
                    if "mlp" in name
                    else nn.init.zeros_(module.bias)
                )
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, "init_weights"):
        module.init_weights()


def init_weights_vit_moco(module: nn.Module, name: str = "") -> None:
    """ViT weight initialization, matching moco-v3 impl minus fixed PatchEmbed"""
    if isinstance(module, nn.Linear):
        if "qkv" in name:
            # treat the weights of Q, K, V separately
            val = math.sqrt(
                6.0 / float(module.weight.shape[0] // 3 + module.weight.shape[1])
            )
            nn.init.uniform_(module.weight, -val, val)
        else:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, "init_weights"):
        module.init_weights()


def get_init_weights_vit(mode: str = "jax", head_bias: float = 0.0) -> Callable:
    if "jax" in mode:
        return partial(init_weights_vit_jax, head_bias=head_bias)
    elif "moco" in mode:
        return init_weights_vit_moco
    else:
        return init_weights_vit_timm


def resize_pos_embed(
    posemb: torch.Tensor,
    posemb_new: torch.Tensor,
    num_prefix_tokens: int = 1,
    gs_new: Tuple[int, int] = (),
    interpolation: str = "bicubic",
    antialias: bool = False,
) -> torch.Tensor:
    """Rescale the grid of position embeddings when loading from state_dict.
    *DEPRECATED* This function is being deprecated in favour of using resample_abs_pos_embed
    """
    ntok_new = posemb_new.shape[1] - num_prefix_tokens
    ntok_old = posemb.shape[1] - num_prefix_tokens
    gs_old = [int(math.sqrt(ntok_old))] * 2
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    return resample_abs_pos_embed(
        posemb,
        gs_new,
        gs_old,
        num_prefix_tokens=num_prefix_tokens,
        interpolation=interpolation,
        antialias=antialias,
        verbose=True,
    )


@torch.no_grad()
def _load_weights(
    model: VisionTransformer,
    checkpoint_path: str,
    prefix: str = "",
    load_bfloat16: bool = False,
) -> None:
    """Load weights from .npz checkpoints for official Google Brain Flax implementation"""
    import numpy as np

    assert not load_bfloat16
    # if load_bfloat16:
    #     import jax.numpy as jnp
    #     import ml_dtypes

    def _n2p(_w, t=True, idx=None):
        if idx is not None:
            _w = _w[idx]

        # if load_bfloat16:
        #     _w = _w.view(ml_dtypes.bfloat16).astype(jnp.float32)
        #     _w = np.array(_w)

        if _w.ndim == 4 and _w.shape[0] == _w.shape[1] == _w.shape[2] == 1:
            _w = _w.flatten()
        if t:
            if _w.ndim == 4:
                _w = _w.transpose([3, 2, 0, 1])
            elif _w.ndim == 3:
                _w = _w.transpose([2, 0, 1])
            elif _w.ndim == 2:
                _w = _w.transpose([1, 0])

        _w = torch.from_numpy(_w)
        return _w

    # if load_bfloat16:
    #     w = jnp.load(checkpoint_path)
    # else:
    #     w = np.load(checkpoint_path)
    w = np.load(checkpoint_path)

    interpolation = "bilinear"
    antialias = False
    big_vision = False
    if not prefix:
        if "opt/target/embedding/kernel" in w:
            prefix = "opt/target/"
        elif "params/embedding/kernel" in w:
            prefix = "params/"
            big_vision = True
        elif "params/img/embedding/kernel" in w:
            prefix = "params/img/"
            big_vision = True

    if hasattr(model.patch_embed, "backbone"):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, "stem")
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(
            adapt_input_conv(
                stem.conv.weight.shape[1], _n2p(w[f"{prefix}conv_root/kernel"])
            )
        )
        stem.norm.weight.copy_(_n2p(w[f"{prefix}gn_root/scale"]))
        stem.norm.bias.copy_(_n2p(w[f"{prefix}gn_root/bias"]))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f"{prefix}block{i + 1}/unit{j + 1}/"
                    for r in range(3):
                        getattr(block, f"conv{r + 1}").weight.copy_(
                            _n2p(w[f"{bp}conv{r + 1}/kernel"])
                        )
                        getattr(block, f"norm{r + 1}").weight.copy_(
                            _n2p(w[f"{bp}gn{r + 1}/scale"])
                        )
                        getattr(block, f"norm{r + 1}").bias.copy_(
                            _n2p(w[f"{bp}gn{r + 1}/bias"])
                        )
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(
                            _n2p(w[f"{bp}conv_proj/kernel"])
                        )
                        block.downsample.norm.weight.copy_(
                            _n2p(w[f"{bp}gn_proj/scale"])
                        )
                        block.downsample.norm.bias.copy_(_n2p(w[f"{bp}gn_proj/bias"]))
        embed_conv_w = _n2p(w[f"{prefix}embedding/kernel"])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f"{prefix}embedding/kernel"])
        )
    if embed_conv_w.shape[-2:] != model.patch_embed.proj.weight.shape[-2:]:
        embed_conv_w = resample_patch_embed(
            embed_conv_w,
            model.patch_embed.proj.weight.shape[-2:],
            interpolation=interpolation,
            antialias=antialias,
            verbose=True,
        )

    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f"{prefix}embedding/bias"]))
    if model.cls_token is not None:
        model.cls_token.copy_(_n2p(w[f"{prefix}cls"], t=False))
    if big_vision:
        pos_embed_w = _n2p(w[f"{prefix}pos_embedding"], t=False)
    else:
        pos_embed_w = _n2p(
            w[f"{prefix}Transformer/posembed_input/pos_embedding"], t=False
        )
    if pos_embed_w.shape != model.pos_embed.shape:
        num_prefix_tokens = (
            0
            if getattr(model, "no_embed_class", False)
            else getattr(model, "num_prefix_tokens", 1)
        )
        pos_embed_w = resample_abs_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w,
            new_size=model.patch_embed.grid_size,
            num_prefix_tokens=num_prefix_tokens,
            interpolation=interpolation,
            antialias=antialias,
            verbose=True,
        )
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f"{prefix}Transformer/encoder_norm/scale"]))
    model.norm.bias.copy_(_n2p(w[f"{prefix}Transformer/encoder_norm/bias"]))
    if (
        isinstance(model.head, nn.Linear)
        and f"{prefix}head/bias" in w
        and model.head.bias.shape[0] == w[f"{prefix}head/bias"].shape[-1]
    ):
        model.head.weight.copy_(_n2p(w[f"{prefix}head/kernel"]))
        model.head.bias.copy_(_n2p(w[f"{prefix}head/bias"]))
    # NOTE representation layer has been removed, not used in latest 21k/1k pretrained weights
    # if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
    #     model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
    #     model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
    if model.attn_pool is not None:
        block_prefix = f"{prefix}MAPHead_0/"
        mha_prefix = block_prefix + f"MultiHeadDotProductAttention_0/"
        model.attn_pool.latent.copy_(_n2p(w[f"{block_prefix}probe"], t=False))
        model.attn_pool.kv.weight.copy_(
            torch.cat(
                [
                    _n2p(w[f"{mha_prefix}{n}/kernel"], t=False).flatten(1).T
                    for n in ("key", "value")
                ]
            )
        )
        model.attn_pool.kv.bias.copy_(
            torch.cat(
                [
                    _n2p(w[f"{mha_prefix}{n}/bias"], t=False).reshape(-1)
                    for n in ("key", "value")
                ]
            )
        )
        model.attn_pool.q.weight.copy_(
            _n2p(w[f"{mha_prefix}query/kernel"], t=False).flatten(1).T
        )
        model.attn_pool.q.bias.copy_(
            _n2p(w[f"{mha_prefix}query/bias"], t=False).reshape(-1)
        )
        model.attn_pool.proj.weight.copy_(_n2p(w[f"{mha_prefix}out/kernel"]).flatten(1))
        model.attn_pool.proj.bias.copy_(_n2p(w[f"{mha_prefix}out/bias"]))
        model.attn_pool.norm.weight.copy_(_n2p(w[f"{block_prefix}LayerNorm_0/scale"]))
        model.attn_pool.norm.bias.copy_(_n2p(w[f"{block_prefix}LayerNorm_0/bias"]))
        for r in range(2):
            getattr(model.attn_pool.mlp, f"fc{r + 1}").weight.copy_(
                _n2p(w[f"{block_prefix}MlpBlock_0/Dense_{r}/kernel"])
            )
            getattr(model.attn_pool.mlp, f"fc{r + 1}").bias.copy_(
                _n2p(w[f"{block_prefix}MlpBlock_0/Dense_{r}/bias"])
            )

    mha_sub, b_sub, ln1_sub = (0, 0, 1) if big_vision else (1, 3, 2)
    for i, block in enumerate(model.blocks.children()):
        if f"{prefix}Transformer/encoderblock/LayerNorm_0/scale" in w:
            block_prefix = f"{prefix}Transformer/encoderblock/"
            idx = i
        else:
            block_prefix = f"{prefix}Transformer/encoderblock_{i}/"
            idx = None
        mha_prefix = block_prefix + f"MultiHeadDotProductAttention_{mha_sub}/"
        block.norm1.weight.copy_(_n2p(w[f"{block_prefix}LayerNorm_0/scale"], idx=idx))
        block.norm1.bias.copy_(_n2p(w[f"{block_prefix}LayerNorm_0/bias"], idx=idx))
        block.attn.qkv.weight.copy_(
            torch.cat(
                [
                    _n2p(w[f"{mha_prefix}{n}/kernel"], t=False, idx=idx).flatten(1).T
                    for n in ("query", "key", "value")
                ]
            )
        )
        block.attn.qkv.bias.copy_(
            torch.cat(
                [
                    _n2p(w[f"{mha_prefix}{n}/bias"], t=False, idx=idx).reshape(-1)
                    for n in ("query", "key", "value")
                ]
            )
        )
        block.attn.proj.weight.copy_(
            _n2p(w[f"{mha_prefix}out/kernel"], idx=idx).flatten(1)
        )
        block.attn.proj.bias.copy_(_n2p(w[f"{mha_prefix}out/bias"], idx=idx))
        block.norm2.weight.copy_(
            _n2p(w[f"{block_prefix}LayerNorm_{ln1_sub}/scale"], idx=idx)
        )
        block.norm2.bias.copy_(
            _n2p(w[f"{block_prefix}LayerNorm_{ln1_sub}/bias"], idx=idx)
        )
        for r in range(2):
            getattr(block.mlp, f"fc{r + 1}").weight.copy_(
                _n2p(w[f"{block_prefix}MlpBlock_{b_sub}/Dense_{r}/kernel"], idx=idx)
            )
            getattr(block.mlp, f"fc{r + 1}").bias.copy_(
                _n2p(w[f"{block_prefix}MlpBlock_{b_sub}/Dense_{r}/bias"], idx=idx)
            )


def _convert_openai_clip(
    state_dict: Dict[str, torch.Tensor],
    model: VisionTransformer,
    prefix: str = "visual.",
) -> Dict[str, torch.Tensor]:
    out_dict = {}
    swaps = [
        ("conv1", "patch_embed.proj"),
        ("positional_embedding", "pos_embed"),
        ("transformer.resblocks.", "blocks."),
        ("ln_pre", "norm_pre"),
        ("ln_post", "norm"),
        ("ln_", "norm"),
        ("in_proj_", "qkv."),
        ("out_proj", "proj"),
        ("mlp.c_fc", "mlp.fc1"),
        ("mlp.c_proj", "mlp.fc2"),
    ]
    for k, v in state_dict.items():
        if not k.startswith(prefix):
            continue
        k = k.replace(prefix, "")
        for sp in swaps:
            k = k.replace(sp[0], sp[1])

        if k == "proj":
            k = "head.weight"
            v = v.transpose(0, 1)
            out_dict["head.bias"] = torch.zeros(v.shape[0])
        elif k == "class_embedding":
            k = "cls_token"
            v = v.unsqueeze(0).unsqueeze(1)
        elif k == "pos_embed":
            v = v.unsqueeze(0)
        out_dict[k] = v
    return out_dict


def _convert_dinov2(
    state_dict: Dict[str, torch.Tensor],
    model: VisionTransformer,
) -> Dict[str, torch.Tensor]:
    import re

    out_dict = {}
    state_dict.pop("mask_token", None)
    if "register_tokens" in state_dict:
        # convert dinov2 w/ registers to no_embed_class timm model (neither cls or reg tokens overlap pos embed)
        out_dict["reg_token"] = state_dict.pop("register_tokens")
        out_dict["cls_token"] = (
            state_dict.pop("cls_token") + state_dict["pos_embed"][:, 0]
        )
        out_dict["pos_embed"] = state_dict.pop("pos_embed")[:, 1:]
    for k, v in state_dict.items():
        if re.match(r"blocks\.(\d+)\.mlp\.w12\.(?:weight|bias)", k):
            out_dict[k.replace("w12", "fc1")] = v
            continue
        elif re.match(r"blocks\.(\d+)\.mlp\.w3\.(?:weight|bias)", k):
            out_dict[k.replace("w3", "fc2")] = v
            continue
        out_dict[k] = v
    return out_dict


def _convert_aimv2(
    state_dict: Dict[str, torch.Tensor],
    model: VisionTransformer,
) -> Dict[str, torch.Tensor]:
    out_dict = {}
    for k, v in state_dict.items():
        k = k.replace("norm_1", "norm1")
        k = k.replace("norm_2", "norm2")
        k = k.replace("preprocessor.patchifier.", "patch_embed.")
        k = k.replace("preprocessor.pos_embed", "pos_embed")
        k = k.replace("trunk.", "")
        k = k.replace("post_trunk_norm.", "norm.")
        k = k.replace("mlp.fc1", "mlp.fc1_g")
        k = k.replace("mlp.fc3", "mlp.fc1_x")
        out_dict[k] = v
    return out_dict


def checkpoint_filter_fn(
    state_dict: Dict[str, torch.Tensor],
    model: VisionTransformer,
    adapt_layer_scale: bool = False,
    interpolation: str = "bicubic",
    antialias: bool = True,
) -> Dict[str, torch.Tensor]:
    """convert patch embedding weight from manual patchify + linear proj to conv"""
    import re

    out_dict = {}
    state_dict = state_dict.get("model", state_dict)
    state_dict = state_dict.get("state_dict", state_dict)
    prefix = ""

    if "visual.class_embedding" in state_dict:
        state_dict = _convert_openai_clip(state_dict, model)
    elif "module.visual.class_embedding" in state_dict:
        state_dict = _convert_openai_clip(state_dict, model, prefix="module.visual.")
    elif "mask_token" in state_dict:
        state_dict = _convert_dinov2(state_dict, model)
    elif "encoder" in state_dict:
        # IJEPA, vit in an 'encoder' submodule
        state_dict = state_dict["encoder"]
        prefix = "module."
    elif (
        "visual.trunk.pos_embed" in state_dict
        or "visual.trunk.blocks.0.norm1.weight" in state_dict
    ):
        # OpenCLIP model with timm vision encoder
        prefix = "visual.trunk."
        if "visual.head.proj.weight" in state_dict and isinstance(
            model.head, nn.Linear
        ):
            # remap final nn.Linear if it exists outside of the timm .trunk (ie in visual.head.proj)
            out_dict["head.weight"] = state_dict["visual.head.proj.weight"]
            out_dict["head.bias"] = torch.zeros(
                state_dict["visual.head.proj.weight"].shape[0]
            )
    elif "preprocessor.patchifier.proj.weight" in state_dict:
        state_dict = _convert_aimv2(state_dict, model)

    if prefix:
        # filter on & remove prefix string from keys
        state_dict = {
            k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)
        }

    for k, v in state_dict.items():
        if "patch_embed.proj.weight" in k:
            O, I, H, W = model.patch_embed.proj.weight.shape
            if len(v.shape) < 4:
                # For old models that I trained prior to conv based patchification
                O, I, H, W = model.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            if v.shape[-1] != W or v.shape[-2] != H:
                v = resample_patch_embed(
                    v,
                    (H, W),
                    interpolation=interpolation,
                    antialias=antialias,
                    verbose=True,
                )
        elif k == "pos_embed" and v.shape[1] != model.pos_embed.shape[1]:
            # To resize pos embedding when using model at different size from pretrained weights
            num_prefix_tokens = (
                0
                if getattr(model, "no_embed_class", False)
                else getattr(model, "num_prefix_tokens", 1)
            )
            v = resample_abs_pos_embed(
                v,
                new_size=model.patch_embed.grid_size,
                num_prefix_tokens=num_prefix_tokens,
                interpolation=interpolation,
                antialias=antialias,
                verbose=True,
            )
        elif adapt_layer_scale and "gamma_" in k:
            # remap layer-scale gamma into sub-module (deit3 models)
            k = re.sub(r"gamma_([0-9])", r"ls\1.gamma", k)
        elif "pre_logits" in k:
            # NOTE representation layer removed as not used in latest 21k/1k pretrained weights
            continue
        out_dict[k] = v
    return out_dict
