import numpy as np
import math
from typing import Any, Dict, List, Tuple, Optional, Type
from functools import partial
import lightning as L
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchmetrics import Metric, JaccardIndex
from minerva.models.lora_adapters.lora import LoRA

"""based on: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/common.py """
class MLPBlock(nn.Module):
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
        return self.lin2(self.act(self.lin1(x)))

"""based on: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/common.py """
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

""" based on: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py"""
class ImageEncoderViT(nn.Module):
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
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
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
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.neck(x.permute(0, 3, 1, 2))

        return x

""" based on: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py"""
class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

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
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
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
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x

""" based on: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py"""
class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
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
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x

""" based on: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py"""
def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
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

""" based on: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py"""
def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
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

""" based on: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py"""
def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
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

""" based on: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py"""
def add_decomposed_rel_pos(
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
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn

""" based on: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py"""
class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """
    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x

"""
*****
PromptEncoder
*****
"""

""" based on: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/prompt_encoder.py"""
class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
          mask_in_chans (int): The number of hidden channels used for
            encoding input masks.
          activation (nn.Module): The activation to use when encoding
            input masks.
        """
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
        """Embeds point prompts."""
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
        """Embeds box prompts."""
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Embeds mask inputs."""
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

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
            and labels to embed.
          boxes (torch.Tensor or none): boxes to embed
          masks (torch.Tensor or none): masks to embed

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
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

""" based on: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/prompt_encoder.py"""
class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
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
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
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
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C

"""
*****
MaskDecoder
*****
"""

""" based on: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/mask_decoder.py"""
class MaskDecoder(nn.Module):
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
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
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
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
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
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
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
        """Predicts masks. See 'forward' for more details."""
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

""" based on: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/mask_decoder.py"""
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

""" based on: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/mask_decoder.py """
class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
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
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
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

""" based on: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/mask_decoder.py """
class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
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

""" based on: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/mask_decoder.py """
class AttentionMaskDecoder(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
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
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
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

class _SAM(nn.Module):
    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
        mask_threshold: float = 0.0
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
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
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
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
        """Normalize pixel values and pad to a square input."""
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
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
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

""" based on: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/sam.py """
class Sam(L.LightningModule):
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
            loss_fn: Optional[nn.Module] = nn.CrossEntropyLoss(), # weight=class_weights.to(self.device),
            train_metrics: Optional[Dict[str, Metric]] = {"mIoU": JaccardIndex(task="multiclass", num_classes=6)},
            val_metrics: Optional[Dict[str, Metric]] = {"mIoU": JaccardIndex(task="multiclass", num_classes=6)},
            test_metrics: Optional[Dict[str, Metric]] = {"mIoU": JaccardIndex(task="multiclass", num_classes=6)},
            apply_freeze:Optional[Dict[str, bool]] = {"image_encoder": False, "prompt_encoder": True, "mask_decoder": False},
            apply_adapter:Optional[Dict[str, LoRA]] = {},
            lora_rank:int=4,
            lora_alpha:int=1
    ):
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts (point/box).
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
          mask_threshold (float): Theshold to apply in mask, converting to binary.
          learning_rate (float): Learning rate apply in train.
        """
        super().__init__()
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.mask_threshold = mask_threshold
        self.learning_rate = learning_rate
        self.num_multimask_outputs = num_multimask_outputs
        self.iou_head_depth = iou_head_depth

        self.loss_fn = loss_fn

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
        if 'image_encoder' in apply_adapter:
            print("LoRA applied in Image Encoder!")
            for layer in self.model.image_encoder.blocks:
                layer.attn.qkv = apply_adapter['image_encoder'](original_module=layer.attn.qkv, bias=True, alpha=alpha, r=rank)
                # layer.mlp.lin1 = apply_adapter['image_encoder'](original_module=layer.mlp.lin1, bias=True, alpha=alpha, r=rank)
                # layer.mlp.lin2 = apply_adapter['image_encoder'](original_module=layer.mlp.lin2, bias=True, alpha=alpha, r=rank)
        if 'mask_decoder' in apply_adapter:
            print("LoRA applied in Mask Decoder!")
            for layer in self.model.mask_decoder.transformer.layers:
                layer.self_attn.q_proj = apply_adapter['mask_decoder'](original_module=layer.self_attn.q_proj, bias=True, alpha=alpha, r=rank)
                # layer.self_attn.k_proj = apply_adapter['mask_decoder'](original_module=layer.self_attn.k_proj, bias=True, alpha=alpha, r=rank)
                layer.self_attn.v_proj = apply_adapter['mask_decoder'](original_module=layer.self_attn.v_proj, bias=True, alpha=alpha, r=rank)
                layer.cross_attn_token_to_image.q_proj = apply_adapter['mask_decoder'](original_module=layer.cross_attn_token_to_image.q_proj, bias=True, alpha=alpha, r=rank)
                # layer.cross_attn_token_to_image.k_proj = apply_adapter['mask_decoder'](original_module=layer.cross_attn_token_to_image.k_proj, bias=True, alpha=alpha, r=rank)
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

    """ mapping weights: developed by https://github.com/hitachinsk/SAMed/blob/main/segment_anything/build_sam.py """
    def load_from_b(self, sam, state_dict, image_size, vit_patch_size):
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
    
    """ mapping weights: developed by https://github.com/hitachinsk/SAMed/blob/main/SAMed_h/segment_anything/build_sam.py """
    def load_from_h(self, sam, state_dict, image_size, vit_patch_size, encoder_global_attn_indexes):
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
            # global_rel_pos_keys = [k for k in rel_pos_keys if '2' in k or '5' in  k or '8' in k or '11' in k]
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
        # debug
        # print("y_hat shape: ", y_hat.shape) # torch.Size([B, num_classes, H, W])
        # print("y_hat shape keepdim: ", torch.argmax(y_hat, dim=1, keepdim=True).squeeze(1).shape) # torch.Size([B, H, W])
        # print("y shape: ", y.shape) # torch.Size([1, H, W])

        return {
            f"{step_name}_{metric_name}": metric.to(self.device)(
                torch.argmax(y_hat, dim=1, keepdim=True).squeeze(1), y
            )
            for metric_name, metric in self.metrics[step_name].items()
        }
    
    def training_step(self, batch, batch_idx):
        batched_input = batch
        outputs = self(batched_input, multimask_output=batched_input[0]['multimask_output'])

        # stack logits 'masks_logits' and 'labels' for loss and metrics function
        masks_logits = torch.stack([output['masks_logits'].squeeze(0) for output in outputs])  # [batch_size, 6, H, W]
        labels = torch.stack([input['label'].squeeze(0) for input in batched_input])  # [batch_size, H, W]

        # debug
        # print("batched_input type: ", type(batched_input))
        # print("outputs type: ", type(outputs))
        # print("batched_input len: ", len(batched_input)) # batch size
        # print("outputs len: ", len(outputs)) # batch size
        # print("label shape: ", type(batched_input[0]['label']), batched_input[0]['label'].shape) # torch.Size([B, H, W])
        # print("masks shape: ", type(outputs[0]['masks']), outputs[0]['masks'].shape) # torch.Size([B, num_classes, H, W])
        # print("iou_predictions shape: ", type(outputs[0]['iou_predictions']), outputs[0]['iou_predictions'].shape) # torch.Size([B, num_classes])
        # print("low_res_logits shape: ", type(outputs[0]['low_res_logits']), outputs[0]['low_res_logits'].shape) # torch.Size([B, num_classes, H_reduzido, W_reduzido])
        # print("masks_logits shape: ", type(outputs[0]['masks_logits']), outputs[0]['masks_logits'].shape) # torch.Size([B, num_classes, H, W])
        # print("shape masks_logits stack: ", masks_logits.shape) # torch.Size([B, num_classes, H, W])
        # print("shape labels stack: ", labels.shape) # torch.Size([B, H, W])
        # print("outputs: ", outputs)

        loss = self._loss(masks_logits, labels)
        metrics = self._compute_metrics(masks_logits, labels, 'train')
        
        self.log("train_loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        for metric_name, metric_value in metrics.items():
            self.log(metric_name, metric_value.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        batched_input = batch
        outputs = self(batched_input, multimask_output=batched_input[0]['multimask_output'])

        # stack logits 'masks_logits' and 'labels' for loss and metrics function
        masks_logits = torch.stack([output['masks_logits'].squeeze(0) for output in outputs])  # [batch_size, 6, H, W]
        labels = torch.stack([input['label'].squeeze(0) for input in batched_input])  # [batch_size, H, W]

        # debug
        # print("batched_input type: ", type(batched_input))
        # print("outputs type: ", type(outputs))
        # print("batched_input len: ", len(batched_input)) # batch size
        # print("outputs len: ", len(outputs)) # batch size
        # print("label shape: ", type(batched_input[0]['label']), batched_input[0]['label'].shape) # torch.Size([B, H, W])
        # print("masks shape: ", type(outputs[0]['masks']), outputs[0]['masks'].shape) # torch.Size([B, num_classes, H, W])
        # print("iou_predictions shape: ", type(outputs[0]['iou_predictions']), outputs[0]['iou_predictions'].shape) # torch.Size([B, num_classes])
        # print("low_res_logits shape: ", type(outputs[0]['low_res_logits']), outputs[0]['low_res_logits'].shape) # torch.Size([B, num_classes, H_reduzido, W_reduzido])
        # print("masks_logits shape: ", type(outputs[0]['masks_logits']), outputs[0]['masks_logits'].shape) # torch.Size([B, num_classes, H, W])
        # print("shape masks_logits stack: ", masks_logits.shape) # torch.Size([B, num_classes, H, W])
        # print("shape labels stack: ", labels.shape) # torch.Size([B, H, W])
        # print("outputs: ", outputs)

        loss = self._loss(masks_logits, labels)
        metrics = self._compute_metrics(masks_logits, labels, 'val')

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        for metric_name, metric_value in metrics.items():
            self.log(metric_name, metric_value, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return loss
    
    def test_step(self, batch: torch.Tensor, batch_idx: int):
        batched_input = batch
        
        outputs = self(batched_input, multimask_output=batched_input[0]['multimask_output'])
        
        # stack logits 'masks_logits' and 'labels' for loss and metrics function
        masks_logits = torch.stack([output['masks_logits'].squeeze(0) for output in outputs])  # [batch_size, 6, H, W]
        labels = torch.stack([input['label'].squeeze(0) for input in batched_input])  # [batch_size, H, W]

        # debug
        # print("batched_input type: ", type(batched_input))
        # print("outputs type: ", type(outputs))
        # print("batched_input len: ", len(batched_input)) # batch size
        # print("outputs len: ", len(outputs)) # batch size
        # print("label shape: ", type(batched_input[0]['label']), batched_input[0]['label'].shape) # torch.Size([B, H, W])
        # print("masks shape: ", type(outputs[0]['masks']), outputs[0]['masks'].shape) # torch.Size([B, num_classes, H, W])
        # print("iou_predictions shape: ", type(outputs[0]['iou_predictions']), outputs[0]['iou_predictions'].shape) # torch.Size([B, num_classes])
        # print("low_res_logits shape: ", type(outputs[0]['low_res_logits']), outputs[0]['low_res_logits'].shape) # torch.Size([B, num_classes, H_reduzido, W_reduzido])
        # print("masks_logits shape: ", type(outputs[0]['masks_logits']), outputs[0]['masks_logits'].shape) # torch.Size([B, num_classes, H, W])
        # print("shape masks_logits stack: ", masks_logits.shape) # torch.Size([B, num_classes, H, W])
        # print("shape labels stack: ", labels.shape) # torch.Size([B, H, W])
        # print("outputs: ", outputs)

        loss = self._loss(masks_logits, labels)
        metrics = self._compute_metrics(masks_logits, labels, 'test')

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        for metric_name, metric_value in metrics.items():
            self.log(metric_name, metric_value, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return loss

    def predict_step(self, batch: torch.Tensor, batch_idx: int, dataloader_idx: Optional[int] = None):
        batched_input = batch
        
        outputs = self(batched_input, multimask_output=batched_input[0]['multimask_output'])
        return outputs
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def _loss(self, masks, label):
        loss_ce = self.loss_fn(masks, label[:].long())
        return loss_ce