# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py


from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_
from typing import Callable, Optional, Tuple, Union

from torch import Tensor
import torch.nn as nn
from functools import partial
import logging

from torch import Tensor
from torch import nn
# import loralib as lora
from torchmetrics import JaccardIndex

logger = logging.getLogger("dinov2")

import lightning as L

logger = logging.getLogger("dinov2")

import logging
from typing import Callable, List, Any, Tuple, Dict

import torch
from torch import nn, Tensor


logger = logging.getLogger("dinov2")


try:
    from xformers.ops import SwiGLU

    XFORMERS_AVAILABLE = True
except ImportError:
    SwiGLU = SwiGLUFFN
    XFORMERS_AVAILABLE = False


try:
    from xformers.ops import fmha
    from xformers.ops import scaled_index_add, index_select_cat

    XFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("xFormers not available")
    XFORMERS_AVAILABLE = False



from typing import Union

import torch
from torch import Tensor
from torch import nn


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


class PatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_size

        assert (
            H % patch_H == 0
        ), f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert (
            W % patch_W == 0
        ), f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x = self.proj(x)  # B C H W
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x

    def flops(self) -> float:
        Ho, Wo = self.patches_resolution
        flops = (
            Ho
            * Wo
            * self.embed_dim
            * self.in_chans
            * (self.patch_size[0] * self.patch_size[1])
        )
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)





class SwiGLUFFNFused(SwiGLU):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            bias=bias,
        )


try:
    from xformers.ops import memory_efficient_attention, unbind, fmha

    XFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("xFormers not available")
    XFORMERS_AVAILABLE = False


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# class Attention_lora(nn.Module):
#     def __init__(
#         self,
#         dim: int,
#         num_heads: int = 8,
#         qkv_bias: bool = False,
#         proj_bias: bool = True,
#         attn_drop: float = 0.0,
#         proj_drop: float = 0.0,
#     ) -> None:
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = head_dim**-0.5

#         self.qkv = lora.Linear(dim, dim * 3, bias=qkv_bias, r=8)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = lora.Linear(dim, dim, bias=proj_bias, r=8)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x: Tensor) -> Tensor:
#         B, N, C = x.shape
#         qkv = (
#             self.qkv(x)
#             .reshape(B, N, 3, self.num_heads, C // self.num_heads)
#             .permute(2, 0, 3, 1, 4)
#         )

#         q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
#         attn = q @ k.transpose(-2, -1)

#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)

#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            assert (
                attn_bias is None
            ), "xFormers is required for nested tensors usage"
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# class MemEffAttention_lora(Attention_lora):
#     def forward(self, x: Tensor, attn_bias=None) -> Tensor:
#         if not XFORMERS_AVAILABLE:
#             assert (
#                 attn_bias is None
#             ), "xFormers is required for nested tensors usage"
#             return super().forward(x)

#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

#         q, k, v = unbind(qkv, 2)

#         x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
#         x = x.reshape([B, N, C])

#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x




class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
    ) -> None:
        super().__init__()
        # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values)
            if init_values
            else nn.Identity()
        )
        self.drop_path1 = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values)
            if init_values
            else nn.Identity()
        )
        self.drop_path2 = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

        self.sample_drop_ratio = drop_path

    def forward(self, x: Tensor) -> Tensor:
        def attn_residual_func(x: Tensor) -> Tensor:
            return self.ls1(self.attn(self.norm1(x)))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x))
            x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        else:
            x = x + attn_residual_func(x)
            x = x + ffn_residual_func(x)
        return x


def drop_add_residual_stochastic_depth(
    x: Tensor,
    residual_func: Callable[[Tensor], Tensor],
    sample_drop_ratio: float = 0.0,
) -> Tensor:
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    # 2) apply residual_func to get residual
    residual = residual_func(x_subset)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = torch.index_add(
        x_flat,
        0,
        brange,
        residual.to(dtype=x.dtype),
        alpha=residual_scale_factor,
    )
    return x_plus_residual.view_as(x)


def get_branges_scales(x, sample_drop_ratio=0.0):
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    residual_scale_factor = b / sample_subset_size
    return brange, residual_scale_factor


def add_residual(
    x, brange, residual, residual_scale_factor, scaling_vector=None
):
    if scaling_vector is None:
        x_flat = x.flatten(1)
        residual = residual.flatten(1)
        x_plus_residual = torch.index_add(
            x_flat,
            0,
            brange,
            residual.to(dtype=x.dtype),
            alpha=residual_scale_factor,
        )
    else:
        x_plus_residual = scaled_index_add(
            x,
            brange,
            residual.to(dtype=x.dtype),
            scaling=scaling_vector,
            alpha=residual_scale_factor,
        )
    return x_plus_residual


attn_bias_cache: Dict[Tuple, Any] = {}


def get_attn_bias_and_cat(x_list, branges=None):
    """
    this will perform the index select, cat the tensors, and provide the attn_bias from cache
    """
    batch_sizes = (
        [b.shape[0] for b in branges]
        if branges is not None
        else [x.shape[0] for x in x_list]
    )
    all_shapes = tuple((b, x.shape[1]) for b, x in zip(batch_sizes, x_list))
    if all_shapes not in attn_bias_cache.keys():
        seqlens = []
        for b, x in zip(batch_sizes, x_list):
            for _ in range(b):
                seqlens.append(x.shape[1])
        attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens)
        attn_bias._batch_sizes = batch_sizes
        attn_bias_cache[all_shapes] = attn_bias

    if branges is not None:
        cat_tensors = index_select_cat(
            [x.flatten(1) for x in x_list], branges
        ).view(1, -1, x_list[0].shape[-1])
    else:
        tensors_bs1 = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in x_list)
        cat_tensors = torch.cat(tensors_bs1, dim=1)

    return attn_bias_cache[all_shapes], cat_tensors


def drop_add_residual_stochastic_depth_list(
    x_list: List[Tensor],
    residual_func: Callable[[Tensor, Any], Tensor],
    sample_drop_ratio: float = 0.0,
    scaling_vector=None,
) -> Tensor:
    # 1) generate random set of indices for dropping samples in the batch
    branges_scales = [
        get_branges_scales(x, sample_drop_ratio=sample_drop_ratio)
        for x in x_list
    ]
    branges = [s[0] for s in branges_scales]
    residual_scale_factors = [s[1] for s in branges_scales]

    # 2) get attention bias and index+concat the tensors
    attn_bias, x_cat = get_attn_bias_and_cat(x_list, branges)

    # 3) apply residual_func to get residual, and split the result
    residual_list = attn_bias.split(residual_func(x_cat, attn_bias=attn_bias))  # type: ignore

    outputs = []
    for x, brange, residual, residual_scale_factor in zip(
        x_list, branges, residual_list, residual_scale_factors
    ):
        outputs.append(
            add_residual(
                x, brange, residual, residual_scale_factor, scaling_vector
            ).view_as(x)
        )
    return outputs


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class NestedTensorBlock(Block):
    def forward_nested(self, x_list: List[Tensor]) -> List[Tensor]:
        """
        x_list contains a list of tensors to nest together and run
        """
        assert isinstance(self.attn, MemEffAttention)

        if self.training and self.sample_drop_ratio > 0.0:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.attn(self.norm1(x), attn_bias=attn_bias)

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.mlp(self.norm2(x))

            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=(
                    self.ls1.gamma if isinstance(self.ls1, LayerScale) else None
                ),
            )
            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=(
                    self.ls2.gamma if isinstance(self.ls1, LayerScale) else None
                ),
            )
            return x_list
        else:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls1(self.attn(self.norm1(x), attn_bias=attn_bias))

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls2(self.mlp(self.norm2(x)))

            attn_bias, x = get_attn_bias_and_cat(x_list)
            x = x + attn_residual_func(x, attn_bias=attn_bias)
            x = x + ffn_residual_func(x)
            return attn_bias.split(x)

    def forward(self, x_or_x_list):
        if isinstance(x_or_x_list, Tensor):
            return super().forward(x_or_x_list)
        elif isinstance(x_or_x_list, list):
            assert (
                XFORMERS_AVAILABLE
            ), "Please install xFormers for nested tensors usage"
            return self.forward_nested(x_or_x_list)
        else:
            raise AssertionError


def named_apply(
    fn: Callable,
    module: nn.Module,
    name="",
    depth_first=True,
    include_root=False,
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x


class IntermediateSequential(nn.Sequential):
    def __init__(self, *args, return_intermediate=True):
        super().__init__(*args)
        self.return_intermediate = return_intermediate

    def forward(self, input):
        if not self.return_intermediate:
            return super().forward(input)

        intermediate_outputs = {}
        output = input
        for name, module in self.named_children():
            output = intermediate_outputs[name] = module(output)

        return output, intermediate_outputs


class SETR_PUP(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(SETR_PUP, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        extra_in_channels = int(self.embedding_dim / 4)
        in_channels = [
            self.embedding_dim,
            extra_in_channels,
            extra_in_channels,
            extra_in_channels,
        ]
        out_channels = [
            extra_in_channels,
            extra_in_channels,
            extra_in_channels,
            extra_in_channels,
        ]

        modules = []
        for i, (in_channel, out_channel) in enumerate(
            zip(in_channels, out_channels)
        ):
            modules.append(self.conv_block(in_channel, out_channel))
            modules.append(
                nn.Upsample(
                    size=(1 // (2 ** (3 - i)), 1 // (2 ** (3 - i))),
                    mode="bilinear",
                )
            )

        modules.append(
            nn.Conv2d(
                in_channels=out_channels[-1],
                out_channels=self.num_classes,
                kernel_size=1,
                stride=1,
                padding=self._get_padding(
                    "VALID",
                    (1, 1),
                ),
            )
        )
        self.decode_net = IntermediateSequential(
            *modules, return_intermediate=False
        )

    def forward(self, x, size):
        n1, n2 = size
        self.decode_net[1] = nn.Upsample(
            size=(n1 // (2 ** (3)), n2 // (2 ** (3))), mode="bilinear"
        )
        self.decode_net[3] = nn.Upsample(
            size=(n1 // (2 ** (2)), n2 // (2 ** (2))), mode="bilinear"
        )
        self.decode_net[5] = nn.Upsample(
            size=(n1 // (2 ** (1)), n2 // (2 ** (1))), mode="bilinear"
        )
        self.decode_net[7] = nn.Upsample(size=(n1, n2), mode="bilinear")
        return self.decode_net(x)

    def conv_block(self, in_channels, out_channels):
        conv = nn.Sequential(
            nn.Conv2d(
                int(in_channels),
                int(out_channels),
                3,
                1,
                padding=self._get_padding(
                    "SAME",
                    (3, 3),
                ),
            ),
            nn.BatchNorm2d(int(out_channels)),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                int(out_channels),
                int(out_channels),
                3,
                1,
                padding=self._get_padding(
                    "SAME",
                    (3, 3),
                ),
            ),
            nn.BatchNorm2d(int(out_channels)),
            nn.ReLU(inplace=True),
        )
        return conv

    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ["SAME", "VALID"]
        if padding_type == "SAME":
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)


class SETR_MLA(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(SETR_MLA, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        self.net1_in, self.net1_intmd, self.net1_out = self._define_agg_net()
        self.net2_in, self.net2_intmd, self.net2_out = self._define_agg_net()
        self.net3_in, self.net3_intmd, self.net3_out = self._define_agg_net()
        self.net4_in, self.net4_intmd, self.net4_out = self._define_agg_net()

        self.output_net = IntermediateSequential(return_intermediate=False)
        self.output_net.add_module(
            "conv_1",
            nn.Conv2d(
                in_channels=self.embedding_dim,
                out_channels=self.num_classes,
                kernel_size=1,
                stride=1,
                padding=self._get_padding(
                    "VALID",
                    (1, 1),
                ),
            ),
        )
        self.output_net.add_module(
            "upsample_1", nn.Upsample(size=(1, 1), mode="bilinear")
        )

    def forward(self, x, size):
        n1, n2 = size
        self.output_net[-1] = nn.Upsample(size=(n1, n2), mode="bilinear")
        x3, x6, x9, x12 = x

        x12_intmd_in = self.net1_in(x12)
        x12_out = self.net1_out(x12_intmd_in)

        x9_in = self.net2_in(x9)
        x9_intmd_in = x9_in + x12_intmd_in
        x9_intmd_out = self.net2_intmd(x9_intmd_in)
        x9_out = self.net2_out(x9_intmd_out)

        x6_in = self.net3_in(x6)
        x6_intmd_in = x6_in + x9_intmd_in
        x6_intmd_out = self.net3_intmd(x6_intmd_in)
        x6_out = self.net3_out(x6_intmd_out)

        x3_in = self.net4_in(x3)
        x3_intmd_in = x3_in + x6_intmd_in
        x3_intmd_out = self.net4_intmd(x3_intmd_in)
        x3_out = self.net4_out(x3_intmd_out)

        out = torch.cat((x12_out, x9_out, x6_out, x3_out), dim=1)
        out = self.output_net(out)

        return out

    def conv_block(self, in_channels, out_channels):
        conv = nn.Sequential(
            nn.Conv2d(
                int(in_channels),
                int(out_channels),
                3,
                1,
                padding=self._get_padding(
                    "SAME",
                    (3, 3),
                ),
            ),
            nn.BatchNorm2d(int(out_channels)),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                int(out_channels),
                int(out_channels),
                3,
                1,
                padding=self._get_padding(
                    "SAME",
                    (3, 3),
                ),
            ),
            nn.BatchNorm2d(int(out_channels)),
            nn.ReLU(inplace=True),
        )
        return conv

    def _define_agg_net(self):
        model_in = IntermediateSequential(return_intermediate=False)
        model_in.add_module(
            "layer_1",
            self.conv_block(self.embedding_dim, int(self.embedding_dim / 2)),
        )

        model_intmd = IntermediateSequential(return_intermediate=False)
        model_intmd.add_module(
            "layer_intmd",
            self.conv_block(
                int(self.embedding_dim / 2), int(self.embedding_dim / 2)
            ),
        )

        model_out = IntermediateSequential(return_intermediate=False)
        model_out.add_module(
            "layer_2",
            self.conv_block(
                int(self.embedding_dim / 2), int(self.embedding_dim / 2)
            ),
        )
        model_out.add_module(
            "layer_3",
            self.conv_block(
                int(self.embedding_dim / 2), int(self.embedding_dim / 4)
            ),
        )
        model_out.add_module(
            "upsample", nn.Upsample(scale_factor=4, mode="bilinear")
        )
        model_out.add_module(
            "layer_4",
            self.conv_block(
                int(self.embedding_dim / 4), int(self.embedding_dim / 4)
            ),
        )
        return model_in, model_intmd, model_out

    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ["SAME", "VALID"]
        if padding_type == "SAME":
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)


class ResidualConvUnit_custom(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups = 1

        self.conv1 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        self.conv2 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock_custom(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
    ):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock_custom, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features,
            out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            groups=1,
        )

        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, size, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            res = nn.functional.interpolate(
                res,
                size=(size[0] // 2, size[1] // 2),
                mode="bilinear",
                align_corners=self.align_corners,
            )
            output = self.skip_add.add(output, res)
            # output += res

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(
            output, size=size, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output


class DPT(nn.Module):
    def __init__(self, embedding_dim, num_classes, features: int = 256):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.features = features
        self.num_classes = num_classes

        self.scratch = self._make_scratch(
            in_shape=(
                self.embedding_dim,
                self.embedding_dim,
                self.embedding_dim,
                self.embedding_dim,
            ),
            out_shape=self.features,
            groups=1,
            expand=False,
        )

        self.scratch.refinenet1 = self._make_fusion_block(features, use_bn=True)
        self.scratch.refinenet2 = self._make_fusion_block(features, use_bn=True)
        self.scratch.refinenet3 = self._make_fusion_block(features, use_bn=True)
        self.scratch.refinenet4 = self._make_fusion_block(features, use_bn=True)
        self.scratch.single_conv = nn.Conv2d(
            features, features // 2, kernel_size=3, stride=1, padding=1
        )
        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, num_classes, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Identity(),
        )

    @staticmethod
    def _make_fusion_block(features: int, use_bn: bool):
        return FeatureFusionBlock_custom(
            features,
            nn.ReLU(False),
            deconv=False,
            bn=use_bn,
            expand=False,
            align_corners=True,
        )

    @staticmethod
    def _make_scratch(in_shape, out_shape, groups=1, expand=False):
        scratch = nn.Module()

        out_shape1 = out_shape
        out_shape2 = out_shape
        out_shape3 = out_shape
        out_shape4 = out_shape
        if expand == True:
            out_shape1 = out_shape
            out_shape2 = out_shape * 2
            out_shape3 = out_shape * 4
            out_shape4 = out_shape * 8

        scratch.layer1_rn = nn.Conv2d(
            in_shape[0],
            out_shape1,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=groups,
        )
        scratch.layer2_rn = nn.Conv2d(
            in_shape[1],
            out_shape2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=groups,
        )
        scratch.layer3_rn = nn.Conv2d(
            in_shape[2],
            out_shape3,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=groups,
        )
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3],
            out_shape4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=groups,
        )

        return scratch

    def forward(self, x, size):
        layer_1, layer_2, layer_3, layer_4 = x
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(
            (size[0] // 16, size[1] // 16), layer_4_rn
        )
        path_3 = self.scratch.refinenet3(
            (size[0] // 8, size[1] // 8), path_4, layer_3_rn
        )
        path_2 = self.scratch.refinenet2(
            (size[0] // 4, size[1] // 4), path_3, layer_2_rn
        )
        path_1 = self.scratch.refinenet1(
            (size[0] // 2, size[1] // 2), path_2, layer_1_rn
        )

        out = self.scratch.single_conv(path_1)
        out = F.interpolate(out, size=size)
        out = self.scratch.output_conv(out)
        return out


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
    ):
        """
        Args:size
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1114, embed_dim)
        )

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [
                x.item() for x in torch.linspace(0, drop_path_rate, depth)
            ]  # stochastic depth decay rule

        if ffn_layer == "mlp":
            logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append(
                    [nn.Identity()] * i + blocks_list[i : i + chunksize]
                )
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.init_weights()

        print(f"Initialized DinoVisionTransformer with img_size={img_size},")
        print(f"patch_size={patch_size},")
        print(f"in_chans={in_chans},")
        print(f"embed_dim={embed_dim},")
        print(f"depth={depth},")
        print(f"num_heads={num_heads},")
        print(f"mlp_ratio={mlp_ratio},")
        print(f"qkv_bias={qkv_bias},")
        print(f"ffn_bias={ffn_bias},")
        print(f"proj_bias={proj_bias},")
        print(f"drop_path_rate={drop_path_rate},")
        print(f"drop_path_uniform={drop_path_uniform},")
        print(f"init_values={init_values},")
        print(f"embed_layer={embed_layer},")
        print(f"act_layer={act_layer},")
        print(f"block_fn={block_fn},")
        print(f"ffn_layer={ffn_layer},")
        print(f"block_chunks={block_chunks}")

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        # named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(
                1, int(math.sqrt(N)), int(math.sqrt(N)), dim
            ).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )

        assert (
            int(w0) == patch_pos_embed.shape[-2]
            and int(h0) == patch_pos_embed.shape[-1]
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat(
            (class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1
        ).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        # print(f"[PREPARE TOKENS] X: {x.shape}, MASKS: {masks is not None}")
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        # print(f"[PREPARE TOKENS] PATCH EMBED: {x.shape}")
        if masks is not None:
            x = torch.where(
                masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x
            )

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)

        return x

    def forward_features_list(self, x_list, masks_list):
        x = [
            self.prepare_tokens_with_masks(x, masks)
            for x, masks in zip(x_list, masks_list)
        ]

        for blk in self.blocks:
            x = blk(x)

        all_x = x
        output = []
        for x, masks in zip(all_x, masks_list):
            x_norm = self.norm(x)
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_patchtokens": x_norm[:, 1:],
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x, masks=None):
        if isinstance(x, list):
            # print(f"FORWARD FEATURES: As list")
            return self.forward_features_list(x, masks)

        # print(f"FORWARD FEATURES: Normal")
        x = self.prepare_tokens_with_masks(x, masks)

        count = 1
        x_middle = {}
        for blk in self.blocks:
            x = blk(x)
            if count == 3 or count == 6 or count == 9 or count == 12:
                x_middle[str(count)] = self.norm(x)[:, 1:]
            count = count + 1

        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_patchtokens": x_norm[:, 1:],
            "x_prenorm": x,
            "masks": masks,
        }, x_middle

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = (
            range(total_block_len - n, total_block_len)
            if isinstance(n, int)
            else n
        )
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(
            blocks_to_take
        ), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        # If n is an int, take the n last blocks. If it's a list, take them
        blocks_to_take = (
            range(total_block_len - n, total_block_len)
            if isinstance(n, int)
            else n
        )
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:  # Passing the nn.Identity()
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(
            blocks_to_take
        ), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1:] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])


class DinoV2(L.LightningModule):
    def __init__(
        self,
        backbone: torch.nn.Module,
        head: torch.nn.Module,
        loss_fn: torch.nn.Module,
        n1: int = 1006,
        n2: int = 782,
        emb_dim: int = 384,
    ):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.loss_fn = loss_fn
        self.n1 = n1
        self.n2 = n2
        self.emb_dim = emb_dim

    def forward(self, x):
        size = (self.n1, self.n2)
        B, _, H, W = x.shape
        features, _ = self.backbone.forward_features(x)
        fea_img = features["x_norm_patchtokens"]
        fea_img = fea_img.view(
            fea_img.size(0), int(H / 14), int(W / 14), self.emb_dim
        )
        fea_img = fea_img.permute(0, 3, 1, 2).contiguous()
        out = self.head(fea_img, size)
        return out

    def training_step(self, batch, batch_idx):
        data, label = batch
        b1, b2, c, h, w = data.shape
        data = data.reshape(b1 * b2, c, h, w)
        b1, b2, c, h, w = label.shape
        label = label.reshape(b1 * b2, h, w)
        outputs = self.forward(data)
        loss = self.loss_fn(outputs, label.long())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        data, label = batch
        b1, b2, c, h, w = data.shape
        data = data.reshape(b1 * b2, c, h, w)
        b1, b2, c, h, w = label.shape
        label = label.reshape(b1 * b2, h, w)
        outputs = self.forward(data)
        loss = self.loss_fn(outputs, label.long())
        self.log("val_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        data, label = batch
        b1, b2, c, h, w = data.shape
        data = data.reshape(b1 * b2, c, h, w)
        b1, b2, c, h, w = label.shape
        label = label.reshape(b1 * b2, h, w)
        outputs = self.forward(data)
        loss = self.loss_fn(outputs, label.long())

        self.metric = self.metric.to(self.device)
        return {"miou": self.metric(outputs, label.long())}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer


# import lightning as L


# backbone = DinoVisionTransformer(
#     patch_size=16,
#     embed_dim=384,
#     depth=12,
#     num_heads=6,
#     mlp_ratio=4,
#     block_fn=partial(Block, attn_class=MemEffAttention), # type: ignore
# )

# head = SETR_PUP(embedding_dim=384, num_classes=6)


# def init_weights_vit_timm(module: nn.Module, name: str = ""):
#     """ViT weight initialization, original timm impl (for reproducibility)"""
#     if isinstance(module, nn.Linear):
#         trunc_normal_(module.weight, std=0.02)
#         if module.bias is not None:
#             nn.init.zeros_(module.bias)


# def vit_small(patch_size=16, **kwargs):
#     model = DinoVisionTransformer(
#         patch_size=patch_size,
#         embed_dim=384,
#         depth=12,
#         num_heads=6,
#         mlp_ratio=4,
#         block_fn=partial(Block, attn_class=MemEffAttention),
#         **kwargs,
#     )
#     return model


# def vit_base(patch_size=16, **kwargs):
#     model = DinoVisionTransformer(
#         patch_size=patch_size,
#         embed_dim=768,
#         depth=12,
#         num_heads=12,
#         mlp_ratio=4,
#         block_fn=partial(Block, attn_class=MemEffAttention),
#         **kwargs,
#     )
#     return model


# def vit_large(patch_size=16, **kwargs):
#     model = DinoVisionTransformer(
#         patch_size=patch_size,
#         embed_dim=1024,
#         depth=24,
#         num_heads=16,
#         mlp_ratio=4,
#         block_fn=partial(Block, attn_class=MemEffAttention),
#         **kwargs,
#     )
#     return model


# def vit_giant2(patch_size=16, **kwargs):
#     """
#     Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
#     """
#     model = DinoVisionTransformer(
#         patch_size=patch_size,
#         embed_dim=1536,
#         depth=40,
#         num_heads=24,
#         mlp_ratio=4,
#         block_fn=partial(Block, attn_class=MemEffAttention),
#         **kwargs,
#     )
#     return model





############################### TRANING ########################################

if __name__ == "__main__":
    import os
    from pathlib import Path
    from typing import Optional
    import numpy as np
    import torch
    import lightning as L
    from matplotlib import pyplot as plt
    from torch.utils.data import DataLoader
    from torchmetrics import JaccardIndex

    from minerva.data.datasets.supervised_dataset import SupervisedReconstructionDataset
    from minerva.data.readers.png_reader import PNGReader
    from minerva.data.readers.tiff_reader import TiffReader
    from minerva.models.loaders import FromPretrained
    from minerva.pipelines.lightning_pipeline import SimpleLightningPipeline
    from minerva.transforms.transform import _Transform, TransformPipeline

    from lightning.pytorch.loggers.csv_logs import CSVLogger
    from lightning.pytorch.callbacks import ModelCheckpoint


    import tqdm
    
    
    class PadCrop(_Transform):
        """Transforms image and pads or crops it to the target size. 
        If the axis is larger than the target size, it will crop the image.
        If the axis is smaller than the target size, it will pad the image.
        """
        
        def __init__(
            self,
            target_h_size: int,
            target_w_size: int,
            padding_mode: str = "reflect",
            seed: int | None = None,
            constant_values: int = 0,
        ):
            """
            Initializes the transformation with target sizes, padding mode, and RNG seed.

            Parameters:
            - target_h_size (int): The target height size.
            - target_w_size (int): The target width size.
            - padding_mode (str): The padding mode to use (default is "reflect").
            - seed (int): Seed for random number generator to make cropping reproducible.
            """
            self.target_h_size = target_h_size
            self.target_w_size = target_w_size
            self.padding_mode = padding_mode
            self.rng = np.random.default_rng(
                seed
            )  # Random number generator with the provided seed
            self.constant_values = constant_values

        def __call__(self, x: np.ndarray) -> np.ndarray:
            h, w = x.shape[:2]
            # print(f"-> [{self.__class__.__name__}] x.shape={x.shape}")

            # Handle height dimension independently: pad if target_h_size > h, else crop
            if self.target_h_size > h:
                pad_h = self.target_h_size - h
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_args = {
                    "array": x,
                    "pad_width": (
                        ((pad_top, pad_bottom), (0, 0), (0, 0))
                        if len(x.shape) == 3
                        else ((pad_top, pad_bottom), (0, 0))
                    ),
                    "mode": self.padding_mode,
                }
                if self.padding_mode == "constant":
                    pad_args["constant_values"] = self.constant_values
                
                x = np.pad(**pad_args)
                
            elif self.target_h_size < h:
                crop_h_start = self.rng.integers(0, h - self.target_h_size + 1)
                x = x[crop_h_start : crop_h_start + self.target_h_size, ...]

            # Handle width dimension independently: pad if target_w_size > w, else crop
            if self.target_w_size > w:
                pad_w = self.target_w_size - w
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                
                pad_args = {
                    "array": x,
                    "pad_width": (
                        ((0, 0), (pad_left, pad_right), (0, 0))
                        if len(x.shape) == 3
                        else ((0, 0), (pad_left, pad_right))
                    ),
                    "mode": self.padding_mode,
                }
                
                if self.padding_mode == "constant":
                    pad_args["constant_values"] = self.constant_values

                x = np.pad(**pad_args)
            
            elif self.target_w_size < w:
                crop_w_start = self.rng.integers(0, w - self.target_w_size + 1)
                x = x[:, crop_w_start : crop_w_start + self.target_w_size, ...]

            # Ensure channel dimension consistency
            if len(x.shape) == 2:  # For grayscale, add a channel dimension
                x = np.expand_dims(x, axis=2)

            # Convert to torch tensor with format C x H x W
            # output = torch.from_numpy(x).float()
            x = np.transpose(x, (2, 0, 1))  # Convert to C x H x W format
            # print(f"[{self.__class__.__name__}] x.shape={x.shape}")
            # print(f"<- [{self.__class__.__name__}] x.shape={x.shape}")
            return x


    class SelectChannel(_Transform):
        """Perform a channel selection on the input image.
        """
        
        def __init__(self, channel: int, expand_channels: int = None):
            """
            Initializes the transformation with the channel to select.

            Parameters:
            - channel (int): The channel to select.
            """
            self.channel = channel
            self.expand_channels = expand_channels

        def __call__(self, x: np.ndarray) -> np.ndarray:
            x =  x[self.channel, ...]
            if self.expand_channels is not None:
                x = np.expand_dims(x, axis=self.expand_channels)
            # print(f"[{self.__class__.__name__}] x.shape={x.shape}")
            return x
        

    class CastTo(_Transform):
        def __init__(self, dtype: type):
            """
            Initializes the transformation with the target data type.

            Parameters:
            - dtype (type): The target data type.
            """
            self.dtype = dtype

        def __call__(self, x: np.ndarray) -> np.ndarray:
            # print(f"[{self.__class__.__name__}] x.shape={x.shape}")
            return x.astype(self.dtype)
        
        
    class SwapAxes(_Transform):
        def __init__(self, source_axis: int, target_axis: int):
            """
            Initializes the transformation with the source and target axes.

            Parameters:
            - source_axis (int): The source axis to swap.
            - target_axis (int): The target axis to swap.
            """
            self.source_axis = source_axis
            self.target_axis = target_axis

        def __call__(self, x: np.ndarray) -> np.ndarray:
            x = np.swapaxes(x, self.source_axis, self.target_axis)
            # print(f"[{self.__class__.__name__}] x.shape={x.shape}")
            return x
        
    class RepeatChannel(_Transform):
        def __init__(self, repeats: int, axis: int):
            """
            Initializes the transformation with the number of repeats.

            Parameters:
            - repeats (int): The number of repeats.
            - axis (int): The axis to repeat.
            """
            self.repeats = repeats
            self.axis = axis

        def __call__(self, x: np.ndarray) -> np.ndarray:
            x = np.repeat(x, self.repeats, axis=self.axis)
            # print(f"[{self.__class__.__name__}] x.shape={x.shape}")
            return x
        
        
    class ExpandDims(_Transform):
        def __init__(self, axis: int):
            """
            Initializes the transformation with the axis to expand.

            Parameters:
            - axis (int): The axis to expand.
            """
            self.axis = axis

        def __call__(self, x: np.ndarray) -> np.ndarray:
            x = np.expand_dims(x, axis=self.axis)
            # print(f"[{self.__class__.__name__}] x.shape={x.shape}")
            return x
        
    class PerSampleTransformPipeline(TransformPipeline):
        def __init__(self, transforms):
            super().__init__(transforms)
            
        def __call__(self, x: np.ndarray) -> np.ndarray:
            new_values = []
            for i, value in enumerate(x):
                # print(f"-->[{self.__class__.__name__}] value[{i}].shape={value.shape}")
                value = super().__call__(value)
                # print(f"<--[{self.__class__.__name__}] value[{i}].shape={value.shape}")
                new_values.append(value)
            
            return np.array(new_values)
        
    from minerva.data.readers.reader import _Reader


    class MultiReader(_Reader):
        def __init__(self, readers):
            self.readers = readers
            
        def __len__(self) -> int:
            return len(self.readers[0])

        def __getitem__(self, i) -> np.ndarray:
            r =  np.stack([reader[i] for reader in self.readers])
            # print(f"The shape of the reader is {r.shape}")
            return r

    class GenericDataModule(L.LightningDataModule):
        def __init__(
            self,
            root_data_dir: str,
            root_annotation_dir: str,
            transforms,
            batch_size: int = 1,
            num_workers: Optional[int] = None,
        ):
            super().__init__()
            self.root_data_dir = Path(root_data_dir)
            self.root_annotation_dir = Path(root_annotation_dir)
            self.transforms = transforms
            self.batch_size = batch_size
            self.num_workers = (
                num_workers if num_workers is not None else os.cpu_count()
            )

            self.datasets = {}

        def setup(self, stage=None):
            if stage == "fit":
                train_img_reader = TiffReader(self.root_data_dir / "train")
                train_label_reader = PNGReader(self.root_annotation_dir / "train")
                train_dataset = SupervisedReconstructionDataset(
                    readers=[
                        MultiReader([train_img_reader, train_img_reader]),
                        MultiReader([train_label_reader, train_label_reader]),
                    ],
                    transforms=self.transforms,
                )

                val_img_reader = TiffReader(self.root_data_dir / "val")
                val_label_reader = PNGReader(self.root_annotation_dir / "val")
                val_dataset = SupervisedReconstructionDataset(
                    readers=[
                        MultiReader([val_img_reader, val_img_reader]),
                        MultiReader([val_label_reader, val_label_reader]),
                    ],
                    transforms=self.transforms,
                )

                self.datasets["train"] = train_dataset
                self.datasets["val"] = val_dataset

            elif stage == "test" or stage == "predict":
                test_img_reader = TiffReader(self.root_data_dir / "test")
                test_label_reader = PNGReader(self.root_annotation_dir / "test")
                test_dataset = SupervisedReconstructionDataset(
                    readers=[
                        MultiReader([test_img_reader, test_img_reader]),
                        MultiReader([test_label_reader, test_label_reader]),
                    ],
                    transforms=self.transforms,
                )
                self.datasets["test"] = test_dataset
                self.datasets["predict"] = test_dataset

            else:
                raise ValueError(f"Invalid stage: {stage}")

        def _get_dataloader(self, partition: str, shuffle: bool):
            return DataLoader(
                self.datasets[partition],
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=shuffle,
            )

        def train_dataloader(self):
            return self._get_dataloader("train", shuffle=True)

        def val_dataloader(self):
            return self._get_dataloader("val", shuffle=False)

        def test_dataloader(self):
            return self._get_dataloader("test", shuffle=False)

        def predict_dataloader(self):
            return self._get_dataloader("predict", shuffle=False)

        def __str__(self) -> str:
            return f"""DataModule
        Data: {self.root_data_dir}
        Annotations: {self.root_annotation_dir}
        Batch size: {self.batch_size}"""

        def __repr__(self) -> str:
            return str(self)
        

    root_data_dir = "/workspaces/HIAAC-KR-Dev-Container/shared_data/seam_ai_datasets/seam_ai/images"
    root_annotation_dir = "/workspaces/HIAAC-KR-Dev-Container/shared_data/seam_ai_datasets/seam_ai/annotations"


    data_module = GenericDataModule(
        root_data_dir=root_data_dir,
        root_annotation_dir=root_annotation_dir,
        transforms=[
            PerSampleTransformPipeline([
                SwapAxes(0, -1),
                SelectChannel(0),
                SwapAxes(0, 1),
                PadCrop(1008, 784, padding_mode="reflect", seed=42, constant_values=0),
                RepeatChannel(repeats=3, axis=0),
                CastTo(np.float32),
            ]), 

            PerSampleTransformPipeline([
                # SwapAxes(0, -1),
                PadCrop(1006, 782, padding_mode="reflect", seed=42, constant_values=0),
                CastTo(np.int64),
            ]), 
        ],
        batch_size=1,
        num_workers=1
    )

    data_module
    
        
    from minerva.models.ssl.dinov2 import (
        DinoVisionTransformer,
        SETR_PUP,
        NestedTensorBlock,
        MemEffAttention,
    )
    from functools import partial

    backbone = DinoVisionTransformer(
        patch_size=14,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(NestedTensorBlock, attn_class=MemEffAttention),  # type: ignore
        init_values=1e-5,
        block_chunks=0,
    )

    head = SETR_PUP(embedding_dim=384, num_classes=6)
    
    
    model = DinoV2(
        backbone=backbone,
        head=head,
        loss_fn=torch.nn.CrossEntropyLoss(),
        n1=1006,
        n2=782
    )

    model
    
    log_dir = "./logs"
    logger = CSVLogger(log_dir, name="dinov2", version="parihaka")
    checkpoint = ModelCheckpoint(
        save_top_k=1,
        save_last=True,
    )


    trainer = L.Trainer(
        max_epochs=1,
        accelerator="gpu",
        devices=1,
        logger=logger,
        callbacks=[checkpoint],
    )

    pipeline = SimpleLightningPipeline(
        model=model,
        trainer=trainer,
        log_dir=log_dir + "/f3_segmentation",
        save_run_status=True
    )
    
    pipeline.run(data_module, task="fit")