# Standard library imports
import math
from collections import OrderedDict
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

# Third-party imports
import lightning as L
import numpy as np
import timm.models.vision_transformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block, PatchEmbed

from minerva.models.nets.base import SimpleSupervisedModel

# Local imports
from minerva.utils.position_embedding import get_2d_sincos_pos_embed

###################################

############## SETR ###############

###################################


class MMAdaptivePadding(nn.Module):
    def __init__(
        self,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
        dilation: Tuple[int, int],
        padding: str = "corner",
    ):
        """
            Applies adaptive padding to the input tensor to ensure its dimensions are compatible
        with a convolutional layer using a given kernel size, stride, and dilation.

        Parameters
        ----------
        kernel_size : Tuple[int, int]
            Size of the convolution kernel.
        stride : Tuple[int, int]
            Stride of the convolution.
        dilation : Tuple[int, int]
            Dilation rate of the convolution.
        padding : str, default="corner"
            Padding mode. Options are "same" or "corner".
        """
        super().__init__()
        assert padding in ("same", "corner")
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding

    def get_pad_shape(self, input_shape):
        H, W = input_shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        dh, dw = self.dilation
        oh = math.ceil(H / sh)
        ow = math.ceil(W / sw)
        pad_h = max((oh - 1) * sh + (kh - 1) * dh + 1 - H, 0)
        pad_w = max((ow - 1) * sw + (kw - 1) * dw + 1 - W, 0)
        return pad_h, pad_w

    def forward(self, x):
        pad_h, pad_w = self.get_pad_shape(x.shape[-2:])
        if pad_h > 0 or pad_w > 0:
            if self.padding == "corner":
                x = F.pad(x, (0, pad_w, 0, pad_h))
            else:
                x = F.pad(
                    x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
                )
        return x


class MMPatchEmbed(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dims: int,
        patch_size: int,
        stride: Optional[int],
        dilation: int,
        bias: bool,
        norm_type: Optional[type],
        norm_params: Optional[dict],
        patch_norm: bool,
        padding_type: str = "corner",
    ):
        """
            Converts an image into patch embeddings using a convolutional projection layer.

        Parameters
        ----------
        in_channels : int
            Number of input image channels.
        embed_dims : int
            Dimensionality of the output patch embeddings.
        patch_size : int
            Size of the square patches.
        stride : Optional[int]
            Stride for the convolution. If None, defaults to patch size.
        dilation : int
            Dilation applied to the convolution.
        bias : bool
            Whether to include a bias term in the projection.
        norm_type : Optional[type]
            Normalization layer class (e.g., nn.LayerNorm).
        norm_params : Optional[dict]
            Parameters to initialize the normalization layer.
        patch_norm : bool
            Whether to apply normalization after patch embedding.
        padding_type : str, default="corner"
            Padding strategy for adaptive padding.
        """
        super().__init__()

        self.adapt_padding = MMAdaptivePadding(
            kernel_size=(patch_size, patch_size),
            stride=(stride, stride) if stride is not None else (patch_size, patch_size),
            dilation=(dilation, dilation),
            padding=padding_type,
        )

        self.projection = nn.Conv2d(
            in_channels,
            embed_dims,
            kernel_size=(patch_size, patch_size),
            stride=(stride, stride) if stride is not None else (patch_size, patch_size),
            dilation=(dilation, dilation),
            padding=0,
            bias=bias,
        )

        if patch_norm and norm_type is not None:
            if norm_params is None:
                self.norm = norm_type(embed_dims)
            else:
                self.norm = norm_type(embed_dims, *norm_params)
        else:
            self.norm = None

    def forward(self, x):
        x = self.adapt_padding(x)
        x = self.projection(x)  # (B, C, H', W')
        out_size = tuple(
            x.shape[2:]
        )  # force to be a tuple (H', W'), instead torch.tensor([H', W']) (mmseg return a tuple instead a tensor)
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        if self.norm:
            x = self.norm(x)
        return x, out_size


class MMMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        attn_drop: float,
        proj_drop: float,
        batch_first: bool,
        bias: bool,
    ):
        """
            Wrapper around `nn.MultiheadAttention` with support for dropout and residual connections.

        Parameters
        ----------
        embed_dims : int
            Dimensionality of each token embedding.
        num_heads : int
            Number of attention heads.
        attn_drop : float
            Dropout rate for attention weights.
        proj_drop : float
            Dropout rate for output projection.
        batch_first : bool
            Whether the input is in (B, L, C) format.
        bias : bool
            If True, add bias terms to the query, key, and value projections.
        """
        super().__init__()
        self.batch_first = batch_first
        self.attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=attn_drop, bias=bias
        )
        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = nn.Dropout(proj_drop)

    def forward(self, x, identity=None):
        if identity is None:
            identity = x

        # MMSeg do this (transpose to (seq_len, batch, dim)):
        if self.batch_first:
            x = x.transpose(0, 1)
        x = self.attn(x, x, x)[0]
        # rollback to (batch, seq_len, dim)
        if self.batch_first:
            x = x.transpose(0, 1)

        x = self.proj_drop(x)
        x = identity + self.dropout_layer(x)
        return x


class MMFFN(nn.Module):

    def __init__(
        self,
        embed_dims: int,
        feedforward_channels: int,
        dropout_type: type,
        dropout_params: Optional[dict],
        act_type: type,
        act_params: Optional[dict],
        num_fcs: int,
        ffn_drop: float,
    ):
        """
        Feed-forward network used within the Transformer encoder layer.

        Parameters
        ----------
        embed_dims : int
            Dimensionality of the token embeddings.
        feedforward_channels : int
            Number of hidden units in the feed-forward layer.
        dropout_type : type
            Dropout module class (e.g., nn.Dropout, DropPath).
        dropout_params : Optional[dict]
            Parameters for the dropout layer.
        act_type : type
            Activation function class (e.g., nn.GELU).
        act_params : Optional[dict]
            Parameters for the activation function.
        num_fcs : int
            Number of fully-connected layers. Only supports 2.
        ffn_drop : float
            Dropout rate applied after each FC layer.
        """
        super().__init__()

        if num_fcs != 2:
            raise ValueError(
                "A implementação atual do FFN suporta apenas num_fcs=2 como no MMSeg."
            )

        self.activate = act_type(**act_params) if act_params else act_type()

        layers = []

        # first block: Linear -> GELU -> Dropout
        first_block = nn.Sequential(
            nn.Linear(embed_dims, feedforward_channels),
            self.activate,
            nn.Dropout(ffn_drop),
        )
        layers.append(first_block)

        # second block: Linear -> Dropout
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))

        self.layers = nn.Sequential(*layers)

        self.dropout_layer = (
            dropout_type(**dropout_params) if dropout_params else dropout_type()
        )

    def forward(self, x, identity=None):
        if identity is None:
            identity = x
        return identity + self.dropout_layer(self.layers(x))


class MMTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        feedforward_channels: int,
        drop_rate: float,
        attn_drop_rate: float,
        drop_path_rate: float,
        num_fcs: int,
        qkv_bias: bool,
        act_type: type,
        act_params: Optional[dict],
        dropout_type: type,
        dropout_params: Optional[dict],
        norm_type: type,
        norm_params: Optional[dict],
        batch_first: bool,
        with_cp: bool,
    ):
        """
        Transformer encoder block consisting of multi-head attention and FFN.

        Parameters
        ----------
        embed_dims : int
            Token embedding dimension.
        num_heads : int
            Number of attention heads.
        feedforward_channels : int
            Hidden dimension in the FFN.
        drop_rate : float
            Dropout rate after attention and FFN.
        attn_drop_rate : float
            Dropout rate for attention weights.
        drop_path_rate : float
            Stochastic depth drop path rate.
        num_fcs : int
            Number of FC layers in FFN. Must be 2.
        qkv_bias : bool
            Whether to use bias in QKV projections.
        act_type : type
            Activation function type.
        act_params : Optional[dict]
            Activation function parameters.
        dropout_type : type
            Dropout class (e.g., nn.Dropout, DropPath).
        dropout_params : Optional[dict]
            Dropout parameters.
        norm_type : type
            Normalization layer type.
        norm_params : Optional[dict]
            Parameters for normalization layers.
        batch_first : bool
            Whether input has shape (B, L, C).
        with_cp : bool
            Whether to use checkpointing for memory savings.

        """
        super().__init__()

        self.ln1 = (
            norm_type(embed_dims, **norm_params)
            if norm_params
            else norm_type(embed_dims)
        )
        self.ln2 = (
            norm_type(embed_dims, **norm_params)
            if norm_params
            else norm_type(embed_dims)
        )

        self.attn = MMMultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            batch_first=batch_first,
            bias=qkv_bias,
        )

        dropout_params = (
            dict(drop_prob=drop_path_rate)
            if drop_path_rate > 0
            else None if dropout_params is None else dropout_params
        )

        self.ffn = MMFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=num_fcs,
            ffn_drop=drop_rate,
            dropout_type=dropout_type,
            dropout_params=dropout_params,
            act_type=act_type,
            act_params=act_params,
        )

        self.with_cp = with_cp

    def forward(self, x):
        def _inner_forward(x):
            x = self.attn(self.ln1(x), identity=x)
            x = self.ffn(self.ln2(x), identity=x)
            return x

        if self.with_cp and x.requires_grad:
            import torch.utils.checkpoint as cp

            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class SetrVitBackbone(nn.Module):

    def __init__(
        self,
        original_resolution: Optional[tuple],
        img_size: tuple,
        patch_size: int,
        embed_dims: int,
        interpolate_mode: str,
        in_channels: int,
        patch_norm: bool,
        stride: Optional[int],
        dilatation: int,
        bias: bool,
        norm_type: type,
        norm_params: Optional[dict],
        padding_type: str,
        num_layers: int,
        num_heads: int,
        out_indices: Union[int, List[int], Tuple[int, ...]],
        drop_rate: float,
        with_cls_token: bool,
        mlp_ratio: int,
        attn_drop_rate: float,
        drop_path_rate: float,
        num_fcs: int,
        qkv_bias: bool,
        output_cls_token: bool,
        act_type: type,
        act_params: dict,
        with_cp: bool,
        dropout_type: type,
        dropout_params: Optional[dict],
        batch_first: bool = True,
    ):
        """
            Vision Transformer (ViT) backbone for semantic segmentation, following the SETR architecture.

        Parameters
        ----------
        original_resolution : Optional[tuple]
            Original training image resolution (used for interpolating positional embeddings).
        img_size : tuple
            Target image size (H, W).
        patch_size : int
            Size of square patches.
        embed_dims : int
            Dimensionality of patch embeddings.
        interpolate_mode : str
            Interpolation method for resizing positional embeddings.
        in_channels : int
            Number of input channels.
        patch_norm : bool
            Whether to apply normalization after patch embedding.
        stride : Optional[int]
            Convolution stride for patch embedding.
        dilatation : int
            Dilation factor for convolution.
        bias : bool
            Whether to use bias in convolution.
        norm_type : type
            Normalization layer class.
        norm_params : Optional[dict]
            Parameters for normalization layers.
        padding_type : str
            Padding type for adaptive padding ("same" or "corner").
        num_layers : int
            Number of transformer encoder layers.
        num_heads : int
            Number of attention heads.
        out_indices : Union[int, List[int], Tuple[int, ...]]
            Indices of layers whose outputs are returned.
        drop_rate : float
            Dropout rate after positional encoding.
        with_cls_token : bool
            Whether to use a class token in the encoder.
        mlp_ratio : int
            Expansion ratio for the hidden layer in FFN.
        attn_drop_rate : float
            Dropout rate in attention.
        drop_path_rate : float
            Stochastic depth drop rate.
        num_fcs : int
            Number of FCs in FFN. Must be 2.
        qkv_bias : bool
            Whether to use bias in QKV projections.
        output_cls_token : bool
            Whether to return the class token in outputs.
        act_type : type
            Activation function class.
        act_params : dict
            Parameters for the activation function.
        with_cp : bool
            Whether to use checkpointing for memory savings.
        dropout_type : type
            Dropout class used in FFN.
        dropout_params : Optional[dict]
            Parameters for dropout.
        batch_first : bool, default=True
            If True, inputs/outputs are in shape (B, L, C).
        """
        super().__init__()

        self.original_resolution = original_resolution
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dims = embed_dims
        self.interpolate_mode = interpolate_mode

        assert (
            len(img_size) == 2
        ), f"The size of image should have length 1 or 2, but got {len(img_size)}"

        self.patch_embed = MMPatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            patch_size=patch_size,
            patch_norm=patch_norm,
            stride=stride,
            dilation=dilatation,
            bias=bias,
            norm_type=norm_type,
            norm_params=norm_params,
            padding_type=padding_type,
        )

        num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        self.with_cls_token = with_cls_token
        self.output_cls_token = output_cls_token

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dims))
        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # manipulate output indices
        if isinstance(out_indices, int):
            if out_indices == -1:
                out_indices = num_layers - 1
            self.out_indices = [out_indices]
        elif isinstance(out_indices, list) or isinstance(out_indices, tuple):
            self.out_indices = out_indices
        else:
            raise TypeError("out_indices must be type of int, list or tuple")

        """
        Generates a list of floats with num_layers elements, ranging from 0 to drop_path_rate, in a linearly spaced fashion.
        This dpr vector is used to apply the Stochastic Depth technique. 
        Instead of applying the same drop_path rate to all layers, this technique 
        linearly decays the value across layers: deeper layers tend to be more likely 
        to be "turned off" during training. This helps in regularizing the training of very deep networks.
        """
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, num_layers)
        ]  # stochastic depth decay rule

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                MMTransformerEncoderLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    feedforward_channels=mlp_ratio * embed_dims,
                    attn_drop_rate=attn_drop_rate,
                    drop_rate=drop_rate,
                    drop_path_rate=dpr[i],
                    num_fcs=num_fcs,
                    qkv_bias=qkv_bias,
                    act_params=act_params,
                    norm_params=norm_params,
                    with_cp=with_cp,
                    batch_first=batch_first,
                    dropout_type=dropout_type,
                    dropout_params=dropout_params,
                    norm_type=norm_type,
                    act_type=act_type,
                )
            )

    #     self.init_weights()

    def _pos_embeding(self, patched_img, hw_shape: tuple, pos_embed):
        """
        Positioning embeding method. Resize the pos_embed, if the input image size doesn't match the training size.

        Args:
            patched_img (torch.Tensor):
                The patched image, it should be shape of [B, L1, C].
            hw_shape (tuple):
                The downsampled image resolution. pos_embed (torch.Tensor): The pos_embed weighs, it should be shape of [B, L2, c].
        Return:
            torch.Tensor:
                The pos encoded image feature.
        """
        assert (
            patched_img.ndim == 3 and pos_embed.ndim == 3
        ), "the shapes of patched_img and pos_embed must be [B, L, C]"
        x_len, pos_len = patched_img.shape[1], pos_embed.shape[1]
        if x_len != pos_len:
            if (
                pos_len
                == (self.img_size[0] // self.patch_size)
                * (self.img_size[1] // self.patch_size)
                + 1
            ):
                pos_h = self.img_size[0] // self.patch_size
                pos_w = self.img_size[1] // self.patch_size
            else:
                raise ValueError(
                    "Unexpected shape of pos_embed, got {}.".format(pos_embed.shape)
                )
            pos_embed = self.resize_pos_embed(
                pos_embed, hw_shape, (pos_h, pos_w), self.interpolate_mode
            )
        return self.drop_after_pos(
            patched_img + pos_embed
        )  # sum of patched_embed and positional embed (PS: don't do this in MultiheadAttention() again!)

    @staticmethod
    def resize_pos_embed(pos_embed, input_shape, pos_shape, mode: str = "bicubic"):
        """
        Resize pos_embed weights. Resize pos_embed using bicubic interpolate method.

        Args:
            pos_embed (torch.Tensor):
                Position embedding weights.
            input_shape (tuple):
                Tuple for (downsampled input image height,  downsampled input image width).
            pos_shape (tuple):
                The resolution of downsampled origin training image.
            mode (str):
                Algorithm used for upsampling:
                ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` | ``'trilinear'``. Default: ``'bicubic'``
        Return:
            torch.Tensor:
                The resized pos_embed of shape [B, L_new, C]
        """
        assert pos_embed.ndim == 3, "shape of pos_embed must be [B, L, C]"
        pos_h, pos_w = pos_shape
        # keep dim for easy deployment
        cls_token_weight = pos_embed[:, 0:1]
        pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w) :]
        pos_embed_weight = pos_embed_weight.reshape(
            1, pos_h, pos_w, pos_embed.shape[2]
        ).permute(0, 3, 1, 2)
        pos_embed_weight = F.interpolate(
            input=pos_embed_weight,
            size=input_shape,
            scale_factor=None,
            mode=mode,
            align_corners=False,
        )
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
        pos_embed = torch.cat((cls_token_weight, pos_embed_weight), dim=1)
        return pos_embed

    # def init_weights(self):
    #     print("iniciando os pesos")
    #     self.patch_embed.init_weights()

    def interpolate_pos_embeddings(
        self, pretrained_pos_embed, new_img_size, patch_size=16
    ):
        h, w = new_img_size[0] // patch_size, new_img_size[1] // patch_size
        if self.original_resolution is None:
            raise ValueError(
                "original_resolution must be set to interpolate pos_embed."
            )
        original_h, original_w = self.original_resolution
        pos_embed_reshaped = pretrained_pos_embed[:, 1:].reshape(
            1, original_h // patch_size, original_w // patch_size, -1
        )
        pos_embed_interpolated = (
            F.interpolate(
                pos_embed_reshaped.permute(0, 3, 1, 2),
                size=(h, w),
                mode=self.interpolate_mode,
                align_corners=False,
            )
            .permute(0, 2, 3, 1)
            .reshape(1, -1, pos_embed_reshaped.shape[-1])
        )

        cls_token = pretrained_pos_embed[:, :1]
        return torch.cat((cls_token, pos_embed_interpolated), dim=1)

    def load_backbone(self, path: str):
        """Loads pretrained weights and handles positional embedding resizing
        if necessary."""
        state_dict = torch.load(path)

        # Caso os pesos venham de um checkpoint do Lightning ou similar
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # Interpola pos_embed, se necessário
        if "pos_embed" in state_dict:
            image_size = (
                (self.img_size, self.img_size)
                if isinstance(self.img_size, int)
                else self.img_size
            )
            expected_shape = (
                1,
                (image_size[0] // self.patch_size) * (image_size[1] // self.patch_size)
                + 1,
                self.embed_dims,
            )

            if state_dict["pos_embed"].shape != expected_shape:
                print("🔄 Interpolando pos_embed para nova resolução...")
                with torch.no_grad():
                    state_dict["pos_embed"] = self.interpolate_pos_embeddings(
                        state_dict["pos_embed"],
                        new_img_size=image_size,
                        patch_size=self.patch_size,
                    )
        else:
            print("⚠️ Arquivo .pth não tem pos_embed; pulando interpolação.")

        # Filtra apenas os pesos que estão presentes no modelo
        model_keys = self.state_dict().keys()
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}

        # Carrega com warning de chaves faltantes/inesperadas
        missing, unexpected = self.load_state_dict(filtered_state_dict, strict=False)
        print(f"🔍 Missing keys: {missing}")
        print(f"🚫 Unexpected keys: {unexpected}")

    def forward(self, x):
        # apply patch embed
        x, hw_shape = self.patch_embed(x)
        # print(x.shape, hw_shape)

        # apply cls token to embed
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, 1024)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 1025, 1024)
        # apply droptout
        x = self._pos_embeding(x, hw_shape, self.pos_embed)

        if not self.with_cls_token:
            x = x[:, 1:]  # remove class token for transformer encoder input

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)

            # process output indices (default: return all indices)
            if i in self.out_indices:
                if self.with_cls_token:
                    out = x[
                        :, 1:
                    ]  # remove class token and reshape token for decoder head
                else:
                    out = x
                B, _, C = out.shape
                out = (
                    out.reshape(B, hw_shape[0], hw_shape[1], C)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                if self.output_cls_token:
                    out = [out, x[:, 0]]
                outs.append(out)

        # final return: the output of last layer, converted like intermediate layers
        if self.with_cls_token:
            x_final = x[:, 1:]
        else:
            x_final = x

        B, _, C = x_final.shape
        x_final = (
            x_final.reshape(B, hw_shape[0], hw_shape[1], C)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        return x_final, tuple(outs)


###################################

############### SFM ###############

###################################


class MaskedAutoencoderViT(L.LightningModule):
    """
    Masked Autoencoder with VisionTransformer backbone.

    Args:
        img_size (int): Size of input image.
        patch_size (int): Size of image patch.
        in_chans (int): Number of input channels.
        embed_dim (int): Dimension of token embeddings.
        depth (int): Number of transformer blocks.
        num_heads (int): Number of attention heads.
        decoder_embed_dim (int): Dimension of decoder embeddings.
        decoder_depth (int): Number of decoder transformer blocks.
        decoder_num_heads (int): Number of decoder attention heads.
        mlp_ratio (float): Ratio of MLP hidden layer size to embedding size.
        norm_layer (torch.nn.LayerNorm): Normalization layer.
        norm_pix_loss (bool): Whether to normalize pixel loss.

    References:
        - timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
        - DeiT: https://github.com/facebookresearch/deit
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=1,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
    ):
        super().__init__()

        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding
        self.in_chans = in_chans
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim),
            requires_grad=False,
        )  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**2 * in_chans, bias=True
        )  # decoder to patch

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # Initialization
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):  # input: (32, 1, 224, 224)
        """
        Extract patches from input images.

        Args:
            imgs (torch.Tensor): Input images of shape (N, C, H, W).

        Returns:
            torch.Tensor: Patches of shape (N, num_patches, patch_size^2 * in_chans).
        """
        p = self.patch_embed.patch_size[0]
        assert (
            imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        )  # only square images are supported, and the size must be divisible by the patch size

        h = w = imgs.shape[2] // p
        x = imgs.reshape(
            (imgs.shape[0], self.in_chans, h, p, w, p)
        )  # Transform images into (32, 1, 14, 16, 14, 16)
        x = torch.einsum("nchpwq->nhwpqc", x)  # reshape into (32, 14, 14, 16, 16, 1)
        x = x.reshape(
            (imgs.shape[0], h * w, p**2 * self.in_chans)
        )  # Transform into (32, 196, 256)
        return x

    def unpatchify(self, x):
        """
        Reconstruct images from patches.

        Args:
            x (torch.Tensor): Patches of shape (N, L, patch_size^2 * in_chans).

        Returns:
            torch.Tensor: Reconstructed images of shape (N, C, H, W).
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape((x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape((x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.

        Args:
            x (torch.Tensor): Input tensor of shape (N, L, D).
            mask_ratio (float): Ratio of values to mask.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Masked input,
            binary mask, shuffled indices.
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones(N, L, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        """
        Forward pass through the encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).
            mask_ratio (float): Ratio of values to mask.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Encoded
            representation, binary mask, shuffled indices.
        """
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]

        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        """
        Forward pass through the decoder.

        Args:
            x (torch.Tensor): Input tensor of shape (N, L, D).
            ids_restore (torch.Tensor): Indices to restore the original order
            of patches.

        Returns:
            torch.Tensor: Decoded output tensor of shape (N, L, patch_size^2 * in_chans).
        """
        x = self.decoder_embed(x)

        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )
        x = torch.cat([x[:, :1, :], x_], dim=1)

        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        x = self.decoder_pred(x)
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        Calculate the loss.

        Args:
            imgs (torch.Tensor): Input images of shape (N, C, H, W).
            pred (torch.Tensor): Predicted output of shape (N, L, patch_size^2 * in_chans).
            mask (torch.Tensor): Binary mask of shape (N, L).

        Returns:
            torch.Tensor: Computed loss value.
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        """
        Forward pass.

        Args:
            imgs (torch.Tensor): Input images of shape (N, C, H, W).
            mask_ratio (float): Ratio of values to mask.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Loss value,
            predicted output, binary mask.
        """
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

    def training_step(self, batch, batch_idx):
        """
        Training step.

        Args:
            batch (Tuple[torch.Tensor]): Input batch of images and corresponding labels.
            batch_idx (int): Index of the current batch.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the loss value for
            the current step.
        """
        imgs, _ = batch
        loss, _, _ = self(imgs)
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        """
        Validation step.

        Args:
            batch (Tuple[torch.Tensor]): Input batch of images and
            corresponding labels.
            batch_idx (int): Index of the current batch.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the loss value for
            the current step.
        """
        imgs, _ = batch
        loss, _, _ = self(imgs)
        self.log("val_loss", loss)
        return {"val_loss": loss}

    def configure_optimizers(self):
        """
        Configure optimizer.

        Returns:
            torch.optim.Optimizer: Optimizer.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# Define model architectures

# mae_vit_small_patch16_dec512d8b
# decoder: 512 dim, 8 blocks, depth: 6
mae_vit_small_patch16 = partial(
    MaskedAutoencoderViT,
    patch_size=16,
    embed_dim=768,
    depth=6,
    num_heads=12,
    decoder_embed_dim=512,
    decoder_depth=4,
    decoder_num_heads=16,
    mlp_ratio=4,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
)

# mae_vit_base_patch16_dec512d8b
# decoder: 512 dim, 8 blocks,
mae_vit_base_patch16 = partial(
    MaskedAutoencoderViT,
    patch_size=16,
    embed_dim=768,
    depth=12,
    num_heads=12,
    decoder_embed_dim=512,
    decoder_depth=8,
    decoder_num_heads=16,
    mlp_ratio=4,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
)

# mae_vit_large_patch16_dec512d8b
# decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = partial(
    MaskedAutoencoderViT,
    patch_size=16,
    embed_dim=1024,
    depth=24,
    num_heads=16,
    decoder_embed_dim=512,
    decoder_depth=8,
    decoder_num_heads=16,
    mlp_ratio=4,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
)

# mae_vit_huge_patch14_dec512d8b
# decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = partial(
    MaskedAutoencoderViT,
    patch_size=14,
    embed_dim=1280,
    depth=32,
    num_heads=16,
    decoder_embed_dim=512,
    decoder_depth=8,
    decoder_num_heads=16,
    mlp_ratio=4,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
)

# mae_vit_large_patch16_dec256d4b
# decoder: 256 dim, 8 blocks
mae_vit_large_patch16D4d256 = partial(
    MaskedAutoencoderViT,
    patch_size=16,
    embed_dim=1024,
    depth=24,
    num_heads=16,
    decoder_embed_dim=256,
    decoder_depth=4,
    decoder_num_heads=16,
    mlp_ratio=4,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
)


# mae_vit_base_patch16_dec256d4b
mae_vit_base_patch16D4d256 = partial(
    MaskedAutoencoderViT,
    patch_size=16,
    embed_dim=768,
    depth=12,
    num_heads=12,
    decoder_embed_dim=256,
    decoder_depth=4,
    decoder_num_heads=16,
    mlp_ratio=4,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
)


################################################################################
# SFM DOWNSTREAM TASKS
################################################################################


class VisionTransformer(
    timm.models.vision_transformer.VisionTransformer, L.LightningModule
):
    """Vision Transformer with support for global average pooling"""

    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        self.decoder = VIT_MLAHead(
            mla_channels=self.embed_dim, num_classes=self.num_classes
        )

        self.segmentation_head = SegmentationHead(
            in_channels=16,
            out_channels=self.num_classes,
            kernel_size=3,
        )
        if self.global_pool:
            norm_layer = kwargs["norm_layer"]
            embed_dim = kwargs["embed_dim"]
            self.fc_norm = norm_layer(embed_dim)
            del self.norm  # remove the original norm

        self.loss_fn = nn.CrossEntropyLoss()

    def forward_features(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        _H, _W = (
            H // self.patch_embed.patch_size[0],
            W // self.patch_embed.patch_size[0],
        )
        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        featureskip = []
        featureskipnum = 1
        for blk in self.blocks:
            x = blk(x)
            if featureskipnum % (len(self.blocks) // 4) == 0:
                featureskip.append(x[:, 1:, :])
                # print(featureskipnum)
            featureskipnum += 1

        x = self.decoder(
            featureskip[0],
            featureskip[1],
            featureskip[2],
            featureskip[3],
            h=_H,
            w=_W,
        )
        return x

    def forward(self, x):
        x = x.float()
        x = self.forward_features(x)
        return x


class Conv2dReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        skip_channels=0,
        use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.up(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        upsampling = (
            nn.UpsamplingBilinear2d(scale_factor=upsampling)
            if upsampling > 1
            else nn.Identity()
        )
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self):
        super().__init__()
        # self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            1024,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )

        decoder_channels = (256, 128, 64, 16)

        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        # if self.config.n_skip != 0:
        #     skip_channels = self.config.skip_channels
        #     for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
        #         skip_channels[3-i]=0
        # else:
        #     skip_channels=[0,0,0,0]
        skip_channels = [512, 256, 128, 64]
        self.conv_feature1 = Conv2dReLU(
            1024, skip_channels[0], kernel_size=3, padding=1, use_batchnorm=True
        )
        self.conv_feature2 = Conv2dReLU(
            1024, skip_channels[1], kernel_size=3, padding=1, use_batchnorm=True
        )
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv_feature3 = Conv2dReLU(
            1024, skip_channels[2], kernel_size=3, padding=1, use_batchnorm=True
        )
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.conv_feature4 = Conv2dReLU(
            1024, skip_channels[3], kernel_size=3, padding=1, use_batchnorm=True
        )
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=8)

        # skip_channels=[128,64,32,8]
        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch)
            for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def TransShape(self, x, head_channels=512, up=0):
        B, n_patch, hidden = (
            x.size()
        )  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)

        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        if up == 0:
            x = self.conv_feature1(x)
        elif up == 1:
            x = self.conv_feature2(x)
            x = self.up2(x)
        elif up == 2:
            x = self.conv_feature3(x)
            x = self.up3(x)
        elif up == 3:
            x = self.conv_feature4(x)
            x = self.up4(x)
        return x

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = (
            hidden_states.size()
        )  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        skip_channels = [512, 256, 128, 64]
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = self.TransShape(
                    features[i], head_channels=skip_channels[i], up=i
                )
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x


class MLAHead(nn.Module):
    def __init__(self, mla_channels=256, mlahead_channels=128, norm_cfg=None):
        super(MLAHead, self).__init__()
        self.head2 = nn.Sequential(
            nn.Conv2d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mlahead_channels),
            nn.ReLU(),
            nn.Conv2d(mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mlahead_channels),
            nn.ReLU(),
        )
        self.head3 = nn.Sequential(
            nn.Conv2d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mlahead_channels),
            nn.ReLU(),
            nn.Conv2d(mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mlahead_channels),
            nn.ReLU(),
        )
        self.head4 = nn.Sequential(
            nn.Conv2d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mlahead_channels),
            nn.ReLU(),
            nn.Conv2d(mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mlahead_channels),
            nn.ReLU(),
        )
        self.head5 = nn.Sequential(
            nn.Conv2d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mlahead_channels),
            nn.ReLU(),
            nn.Conv2d(mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mlahead_channels),
            nn.ReLU(),
        )

    def forward(self, mla_p2, mla_p3, mla_p4, mla_p5):
        head2 = F.interpolate(
            self.head2(mla_p2),
            (4 * mla_p2.shape[-2], 4 * mla_p2.shape[-1]),
            mode="bilinear",
            align_corners=True,
        )
        head3 = F.interpolate(
            self.head3(mla_p3),
            (4 * mla_p3.shape[-2], 4 * mla_p3.shape[-1]),
            mode="bilinear",
            align_corners=True,
        )
        head4 = F.interpolate(
            self.head4(mla_p4),
            (4 * mla_p4.shape[-2], 4 * mla_p4.shape[-1]),
            mode="bilinear",
            align_corners=True,
        )
        head5 = F.interpolate(
            self.head5(mla_p5),
            (4 * mla_p5.shape[-2], 4 * mla_p5.shape[-1]),
            mode="bilinear",
            align_corners=True,
        )
        return torch.cat([head2, head3, head4, head5], dim=1)


class VIT_MLAHead(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        img_size=768,
        mla_channels=256,
        mlahead_channels=128,
        num_classes=6,
        norm_layer=nn.BatchNorm2d,
        norm_cfg=None,
        **kwargs,
    ):
        super(VIT_MLAHead, self).__init__(**kwargs)
        self.img_size = img_size
        self.norm_cfg = norm_cfg
        self.mla_channels = mla_channels
        self.BatchNorm = norm_layer
        self.mlahead_channels = mlahead_channels
        self.num_classes = num_classes
        self.mlahead = MLAHead(
            mla_channels=self.mla_channels,
            mlahead_channels=self.mlahead_channels,
            norm_cfg=self.norm_cfg,
        )
        self.cls = nn.Conv2d(4 * self.mlahead_channels, self.num_classes, 3, padding=1)

    def forward(self, x1, x2, x3, x4, h=14, w=14):
        B, n_patch, hidden = x1.size()
        if h == w:
            h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x1 = x1.permute(0, 2, 1)
        x1 = x1.contiguous().view(B, hidden, h, w)
        x2 = x2.permute(0, 2, 1)
        x2 = x2.contiguous().view(B, hidden, h, w)
        x3 = x3.permute(0, 2, 1)
        x3 = x3.contiguous().view(B, hidden, h, w)
        x4 = x4.permute(0, 2, 1)
        x4 = x4.contiguous().view(B, hidden, h, w)
        x = self.mlahead(x1, x2, x3, x4)
        x = self.cls(x)
        x = F.interpolate(x, size=(h * 16, w * 16), mode="bilinear", align_corners=True)
        return x


def vit_base_patch16_downstream_regression(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_large_patch16_downstream_regression(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_huge_patch14_downstream_regression(**kwargs):
    model = VisionTransformer(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def interpolate_pos_embed(model, checkpoint_model, newsize1=None, newsize2=None):
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches**0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            if newsize1 == None:
                newsize1, newsize2 = new_size, new_size
            print(
                "Position interpolate from %dx%d to %dx%d"
                % (orig_size, orig_size, newsize1, newsize2)
            )
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(
                -1, orig_size, orig_size, embedding_size
            ).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens,
                size=(newsize1, newsize2),
                mode="bicubic",
                align_corners=False,
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model["pos_embed"] = new_pos_embed


class SFM_BasePatch16_Downstream(SimpleSupervisedModel):
    def __init__(
        self,
        img_size: Union[int, Tuple[int, ...]] = (512, 512),
        num_classes: int = 6,
        in_chans: int = 1,
        loss_fn: Optional[torch.nn.Module] = None,
        learning_rate: float = 1e-3,
        **kwargs,
    ):
        """Create a SFM model with a ViT base backbone. The ViT-Base-16 backbone
        has the following configuration:
        - Patch size: 16
        - Embedding dimension: 768
        - Depth: 12
        - Number of heads: 12

        Parameters
        ----------
        img_size : Union[int, Tuple[int, ...]]
            Size of the input image. Note that, to use default pre-trained SFM
            model, the size should be (512, 512).
        num_classes : int
            Number of classes for segmentation head. Default is 6.
        in_chans : int
            Number of input channels. Default is 1.
        loss_fn : Optional[torch.nn.Module], optional
            Loss function, by default None
        learning_rate : float, optional
            Learning rate value, by default 1e-3
        """
        super().__init__(
            backbone=vit_base_patch16_downstream_regression(
                img_size=img_size,
                num_classes=num_classes,
                in_chans=in_chans,
            ),
            fc=torch.nn.Identity(),
            loss_fn=loss_fn or torch.nn.CrossEntropyLoss(),
            learning_rate=learning_rate,
            flatten=False,
            **kwargs,
        )

    def _single_step(
        self, batch: torch.Tensor, batch_idx: int, step_name: str
    ) -> torch.Tensor:
        x, y = batch
        x = x.float()
        if x.shape[1] > 1:
            x = x[:, 0:1, :, :]
        if y.ndim == 4:
            y = y[:, 0, :, :].long()
        return super()._single_step((x, y), batch_idx, step_name)

    def predict_step(
        self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        x, _ = batch
        x = x.float()
        if x.shape[1] > 1:
            x = x[:, 0:1, :, :]
        logits = self.backbone.model(x)
        return logits
