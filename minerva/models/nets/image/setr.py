import math
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import lightning.pytorch as L
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch import nn
from torch.optim.adam import Adam
from torchmetrics import Metric

from minerva.models.nets.base import SimpleSupervisedModel
from minerva.models.nets.image.vit import SetrVitBackbone


class ConvModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        norm_type: type,
        act_type: type,
        norm_params: Optional[dict] = None,
        act_params: Optional[dict] = None,
    ):
        """
        Convolutional module with normalization and activation.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        kernel_size : int
            Size of the convolution kernel.
        padding : int
            Padding added to both sides of the input.
        norm_type : type
            Type of normalization layer (e.g., nn.BatchNorm2d).
        act_type : type
            Type of activation function (e.g., nn.ReLU).
        norm_params : dict, optional
            Optional parameters for normalization.
        act_params : dict, optional
            Optional parameters for activation.
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, bias=False
        )
        self.bn = (
            norm_type(out_channels, **norm_params)
            if norm_params
            else norm_type(out_channels)
        )
        self.activate = act_type(**act_params) if act_params else act_type()
        self.init_weights()

    def init_weights(self):
        """Initialize convolution weights."""
        init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
        if self.conv.bias is not None:
            init.constant_(self.conv.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ConvModule."""
        return self.activate(self.bn(self.conv(x)))


class _SETRUPHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        num_classes: int,
        in_index: int,
        num_convs: int,
        up_scale: int,
        kernel_size: int,
        align_corners: bool,
        dropout: float,
        norm_type: type,
        act_type: type,
        norm_params: Optional[dict] = None,
        act_params: Optional[dict] = None,
        interpolate_mode: str = "bilinear",
    ):
        """
        Lightweight decoder head with LayerNorm and upsampling for SETR.

        Parameters
        ----------
        in_channels : int
            Number of input channels from encoder.
        channels : int
            Number of internal intermediate channels.
        num_classes : int
            Number of target output classes.
        in_index : int
            Index to select feature from encoder outputs.
        num_convs : int
            Number of upsampling convolutional layers.
        up_scale : int
            Upsample factor per layer.
        kernel_size : int
            Convolution kernel size.
        align_corners : bool
            Align corners in bilinear upsampling.
        dropout : float
            Dropout probability.
        norm_type : type
            Normalization layer type.
        act_type : type
            Activation function type.
        norm_params : dict, optional
            Additional parameters for normalization.
        act_params : dict, optional
            Additional parameters for activation.
        interpolate_mode : str, default="bilinear"
            Interpolation mode for upsampling.
        """
        super().__init__()
        self.in_index = in_index
        self.align_corners = align_corners
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout, inplace=False)
        self.norm = nn.LayerNorm(in_channels, eps=1e-6, elementwise_affine=True)

        self.up_convs = nn.ModuleList()
        current_in = in_channels
        for _ in range(num_convs):
            self.up_convs.append(
                nn.Sequential(
                    ConvModule(
                        in_channels=current_in,
                        out_channels=channels,
                        kernel_size=kernel_size,
                        padding=(kernel_size - 1) // 2,
                        norm_type=norm_type,
                        norm_params=norm_params,
                        act_type=act_type,
                        act_params=act_params,
                    ),
                    nn.Upsample(
                        scale_factor=up_scale,
                        mode=interpolate_mode,
                        align_corners=align_corners,
                    ),
                )
            )
            current_in = channels

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass of SETR decoder head."""
        x = xs[self.in_index]
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).transpose(1, 2).contiguous()
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, C, H, W).contiguous()

        for conv in self.up_convs:
            x = conv(x)

        x = self.conv_seg(x)
        return x


class MMDropPath(nn.Module):
    def __init__(self, drop_prob: float):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        output = x.div(keep_prob) * random_tensor
        return output


class _SetR_PUP(nn.Module):

    def __init__(
        self,
        # Encoder
        original_resolution: Optional[tuple],
        img_size: tuple,
        patch_size: int,
        in_channels: int,
        embed_dims: int,
        num_layers: int,
        num_heads: int,
        out_indices: Optional[tuple],
        stride: int,
        patch_norm: bool,
        dilatation: int,
        bias: bool,
        padding_type: str,
        mlp_ratio: int,
        attn_drop_rate: float,
        drop_path_rate: float,
        num_fcs: int,
        qkv_bias: bool,
        output_cls_token: bool,
        act_type: type,
        with_cp: bool,
        encoder_dropout: float,
        encoder_norm_type: type,
        dropout_type: type,
        cls_token: bool,
        interpolate_mode: str,
        act_params: Optional[dict],
        dropout_params: Optional[dict],
        encoder_norm_params: Optional[dict],
        # Decoder
        decoder_channels: int,
        decoder_in_index: int,
        num_classes: int,
        decoder_dropout: float,
        decoder_norm_type: type,
        decoder_num_convs: int,
        decoder_up_scale: int,
        decoder_kernel_size: int,
        decoder_align_corners: bool,
        decoder_norm_params: Optional[dict],
        # Aux heads
        aux_heads_in_index: tuple[int, int, int],
        aux_head_num_convs: int,
        aux_head_up_scale: int,
    ):
        """
        Full SETR_PUP model with encoder and decoder.

        Parameters
        ----------
        original_resolution : tuple
            Resolution of original input image.
        img_size : tuple
            Input image size used for patch embedding.
        patch_size : int
            Patch size for Vision Transformer.
        in_channels : int
            Number of input image channels.
        embed_dims : int
            Dimensionality of embeddings.
        num_layers : int
            Number of transformer encoder layers.
        num_heads : int
            Number of attention heads.
        out_indices : tuple or None
            Indices of intermediate outputs for decoding.
        stride : int
            Patch stride.
        patch_norm : bool
            Apply normalization to patches.
        dilatation : int
            Dilation for patch embedding.
        bias : bool
            Use bias in conv layers.
        padding_type : str
            Padding type used for patch embedding.
        mlp_ratio : int
            MLP expansion ratio.
        attn_drop_rate : float
            Attention dropout rate.
        drop_path_rate : float
            Stochastic depth dropout rate.
        num_fcs : int
            Number of fully connected layers in FFN.
        qkv_bias : bool
            Use bias in QKV projections.
        output_cls_token : bool
            Output class token with final features.
        act_type : type
            Activation function type.
        with_cp : bool
            Use gradient checkpointing.
        encoder_dropout : float
            Dropout rate after patch embedding.
        encoder_norm_type : type
            Type of normalization used in encoder.
        dropout_type : type
            Type of residual dropout layer.
        cls_token : bool
            Use class token in transformer.
        interpolate_mode : str
            Mode for interpolating positional embeddings.
        act_params : dict, optional
            Params for activation function.
        dropout_params : dict, optional
            Params for dropout module.
        encoder_norm_params : dict, optional
            Params for encoder normalization.
        decoder_channels : int
            Number of intermediate decoder channels.
        decoder_in_index : int
            Which encoder layer to use in decoder.
        num_classes : int
            Number of classes for segmentation.
        decoder_dropout : float
            Dropout rate in decoder.
        decoder_norm_type : type
            Normalization type in decoder.
        decoder_num_convs : int
            Number of conv blocks in decoder.
        decoder_up_scale : int
            Upsample scale factor.
        decoder_kernel_size : int
            Decoder conv kernel size.
        decoder_align_corners : bool
            Use align_corners in bilinear upsample.
        decoder_norm_params : dict, optional
            Parameters for decoder normalization.
        aux_heads_in_index : tuple of int
            Which layers to use in auxiliary decoders.
        aux_head_num_convs : int
            Number of convs in each auxiliary head.
        aux_head_up_scale : int
            Upsample factor in each auxiliary head.
        """
        super().__init__()
        if out_indices is None:
            assert out_indices is not None, "encoder_out_indices must be provided."
            # assert (
            #     len(encoder_out_indices) == 3
            # ), "encoder_out_indices must have 3 values. Only 3 aux heads are supported."

        self.encoder_out_indices = out_indices

        # definindo encoder (ViT do MMSegmentation)
        self.encoder = SetrVitBackbone(
            original_resolution=original_resolution,
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dims,
            num_layers=num_layers,
            num_heads=num_heads,
            out_indices=out_indices,
            drop_rate=encoder_dropout,
            norm_type=encoder_norm_type,
            norm_params=encoder_norm_params,
            with_cls_token=cls_token,
            interpolate_mode=interpolate_mode,
            stride=stride,
            patch_norm=patch_norm,
            dilatation=dilatation,
            bias=bias,
            padding_type=padding_type,
            mlp_ratio=mlp_ratio,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            num_fcs=num_fcs,
            qkv_bias=qkv_bias,
            output_cls_token=output_cls_token,
            act_type=act_type,
            act_params=act_params if act_params is not None else dict(),
            with_cp=with_cp,
            dropout_type=dropout_type,
            dropout_params=(
                dropout_params if dropout_params is not None else dict(drop_prob=0.0)
            ),
        )

        # definindo decoder SETR (do MMSegmentation)
        self.decoder = _SETRUPHead(
            in_channels=embed_dims,
            channels=decoder_channels,
            in_index=decoder_in_index,
            num_classes=num_classes,
            dropout=decoder_dropout,
            norm_type=decoder_norm_type,
            norm_params=decoder_norm_params,
            num_convs=decoder_num_convs,
            up_scale=decoder_up_scale,
            kernel_size=decoder_kernel_size,
            align_corners=decoder_align_corners,
            act_type=act_type,
            act_params=act_params if act_params is not None else dict(),
            interpolate_mode=interpolate_mode,
        )

        # definindo aux_heads decoder do SETR (do MMSegmentation). PS: a diferença é o in_index, num_convs e up_scale
        self.aux_head1 = _SETRUPHead(
            in_channels=embed_dims,
            channels=decoder_channels,
            in_index=aux_heads_in_index[0],
            num_classes=num_classes,
            dropout=decoder_dropout,
            norm_type=decoder_norm_type,
            norm_params=decoder_norm_params,
            num_convs=aux_head_num_convs,
            up_scale=aux_head_up_scale,
            kernel_size=decoder_kernel_size,
            align_corners=decoder_align_corners,
            act_type=act_type,
            act_params=act_params if act_params is not None else dict(),
            interpolate_mode=interpolate_mode,
        )
        self.aux_head2 = _SETRUPHead(
            in_channels=embed_dims,
            channels=decoder_channels,
            in_index=aux_heads_in_index[1],
            num_classes=num_classes,
            dropout=decoder_dropout,
            norm_type=decoder_norm_type,
            norm_params=decoder_norm_params,
            num_convs=aux_head_num_convs,
            up_scale=aux_head_up_scale,
            kernel_size=decoder_kernel_size,
            align_corners=decoder_align_corners,
            act_type=act_type,
            act_params=act_params if act_params is not None else dict(),
            interpolate_mode=interpolate_mode,
        )
        self.aux_head3 = _SETRUPHead(
            in_channels=embed_dims,
            channels=decoder_channels,
            in_index=aux_heads_in_index[2],
            num_classes=num_classes,
            dropout=decoder_dropout,
            norm_type=decoder_norm_type,
            norm_params=decoder_norm_params,
            num_convs=aux_head_num_convs,
            up_scale=aux_head_up_scale,
            kernel_size=decoder_kernel_size,
            align_corners=decoder_align_corners,
            act_type=act_type,
            act_params=act_params if act_params is not None else dict(),
            interpolate_mode=interpolate_mode,
        )

    def forward(self, x: torch.Tensor):
        if self.encoder_out_indices is not None:
            x, aux_results = self.encoder(
                x
            )  # x é a ultima camada e aux_results são as camadas definidas por out_indices
            # PS: no forward() do decoder busca o in_index definido para cada aux_head
            x_aux1 = self.aux_head1(aux_results)  # usa in_index 0
            x_aux2 = self.aux_head2(aux_results)  # usa in_index 1
            x_aux3 = self.aux_head3(aux_results)  # usa in_index 2
            x = self.decoder(
                aux_results
            )  # usa in_index 3 (no caso a ultima camada, que é o proprio x)
            return x, x_aux1, x_aux2, x_aux3

        x, aux_results = self.encoder(x)
        x = self.decoder(x)
        return x


class SETR_PUP(SimpleSupervisedModel):

    def __init__(
        self,
        # encoder params
        original_resolution: Optional[tuple] = None,
        img_size: tuple = (512, 512),
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dims: int = 1024,
        num_layers: int = 24,
        num_heads: int = 16,
        out_indices: Optional[tuple] = (9, 14, 19, 23),
        encoder_stride: Optional[int] = None,
        patch_norm: bool = False,
        dilatation: int = 1,
        bias: bool = True,
        padding_type: str = "corner",
        mlp_ratio: int = 4,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        num_fcs: int = 2,
        qkv_bias: bool = True,
        output_cls_token: bool = False,
        act_type: type = nn.GELU,
        with_cp: bool = False,
        encoder_dropout: float = 0.0,
        encoder_norm_type: type = nn.LayerNorm,
        dropout_type: type = MMDropPath,
        cls_token: bool = True,
        interpolate_mode: str = "bilinear",
        act_params: Optional[dict] = None,
        dropout_params: Optional[dict] = None,
        encoder_norm_params: Optional[dict] = None,
        # Decoder
        decoder_channels: int = 256,
        decoder_in_index: int = 3,
        num_classes: int = 6,
        decoder_dropout: float = 0.0,
        decoder_norm_type: type = nn.SyncBatchNorm,
        decoder_num_convs: int = 4,
        decoder_up_scale: int = 2,
        decoder_kernel_size: int = 3,
        decoder_align_corners: bool = False,
        decoder_norm_params: Optional[dict] = None,
        # Aux heads
        aux_heads_in_index: tuple[int, int, int] = (0, 1, 2),
        aux_head_num_convs: int = 2,
        aux_head_up_scale: int = 4,
        # Training
        aux_weights: Optional[list[float]] = None,
        loss_fn: Optional[nn.Module] = None,
        optimizer: type = Adam,
        optimizer_kwargs: Optional[Dict] = None,
        train_metrics: Optional[Dict[str, Metric]] = None,
        val_metrics: Optional[Dict[str, Metric]] = None,
        test_metrics: Optional[Dict[str, Metric]] = None,
        freeze_backbone: bool = False,
        learning_rate: float = 1e-3,
        loss_weights: Optional[list[float]] = None,
        lr_scheduler: Optional[type] = None,
        lr_scheduler_kwargs: Optional[Dict[str, Any]] = None,
        head_lr_factor: float = 1.0,
        use_sliding_inference: bool = True,
        sliding_window_stride: Tuple[int, int] = (341, 341),
    ):
        """
        LightningModule implementation for SETR_PUP (SEgmentation TRansformer with Progressive UPsampling).

        Parameters
        ----------
        original_resolution : tuple
            Resolution of the original images used to pretrain the backbone.
        img_size : tuple
            Input image size (height, width) used during training and patch embedding.
        patch_size : int
            Size of each image patch extracted in the ViT encoder.
        in_channels : int
            Number of input channels (usually 3 for RGB).
        embed_dims : int
            Embedding dimension for each patch.
        num_layers : int
            Number of transformer encoder layers.
        num_heads : int
            Number of attention heads in each transformer layer.
        out_indices : tuple, optional
            Indices of the encoder layers to use as features for decoding.
        encoder_stride : int
            Stride used in patch embedding.
        patch_norm : bool
            Whether to apply normalization to patch embeddings.
        dilatation : int
            Dilation factor for patch embedding.
        bias : bool
            Whether to include bias in the projection layers.
        padding_type : str
            Padding mode used in patch embedding ("same" or "corner").
        mlp_ratio : int
            Expansion ratio for the MLP block inside transformer layers.
        attn_drop_rate : float
            Dropout rate applied to attention weights.
        drop_path_rate : float
            Probability of dropping entire residual paths (stochastic depth).
        num_fcs : int
            Number of linear layers in the feed-forward MLP of the transformer.
        qkv_bias : bool
            Whether to include bias in QKV projections.
        output_cls_token : bool
            Whether to include class token in encoder output.
        act_type : type
            Activation function class to use (e.g., nn.GELU).
        with_cp : bool
            Whether to enable checkpointing to save memory.
        encoder_dropout : float
            Dropout rate after positional embedding in the encoder.
        encoder_norm_type : type
            Normalization type used in the encoder.
        dropout_type : type
            Type of stochastic path dropout layer.
        cls_token : bool
            Whether to use a class token in the ViT.
        interpolate_mode : str
            Interpolation mode used for resizing positional embeddings.
        act_params : dict, optional
            Additional parameters for the activation function.
        dropout_params : dict, optional
            Additional parameters for the dropout layer.
        encoder_norm_params : dict, optional
            Additional parameters for the encoder normalization layer.

        decoder_channels : int
            Number of channels in intermediate layers of the decoder.
        decoder_in_index : int
            Index into encoder outputs to be used as decoder input.
        num_classes : int
            Number of segmentation classes.
        decoder_dropout : float
            Dropout probability in the decoder.
        decoder_norm_type : type
            Type of normalization in decoder conv blocks.
        decoder_num_convs : int
            Number of conv+upsample blocks in the decoder.
        decoder_up_scale : int
            Upsample scale factor for decoder blocks.
        decoder_kernel_size : int
            Convolution kernel size in decoder blocks.
        decoder_align_corners : bool
            Whether to align corners when using bilinear interpolation.
        decoder_norm_params : dict, optional
            Additional arguments for decoder normalization.

        aux_heads_in_index : tuple of int
            Indices of encoder layers to feed into each auxiliary decoder head.
        aux_head_num_convs : int
            Number of conv blocks in each auxiliary head.
        aux_head_up_scale : int
            Upsample factor for auxiliary heads.

        aux_weights : list of float, optional
            Weights for auxiliary losses [aux1, aux2, aux3].
        loss_fn : nn.Module, optional
            Loss function module (defaults to CrossEntropy).
        optimizer_type : type, optional
            Optimizer class (e.g., torch.optim.Adam).
        optimizer_params : dict, optional
            Parameters to pass to the optimizer.
        train_metrics : dict, optional
            Dictionary of training metrics.
        val_metrics : dict, optional
            Dictionary of validation metrics.
        test_metrics : dict, optional
            Dictionary of test metrics.
        learning_rate : float
            Learning rate for training.
        loss_weights : list of float, optional
            Class-wise weights for the loss function.
        head_lr_factor : float
            Learning rate multiplier for decoder heads.
        lr_scheduler : type, optional
            Learning rate scheduler class to be instantiated. By default, it is
            set to None, which means no scheduler will be used. Should be a
            subclass of `torch.optim.lr_scheduler.LRScheduler` (e.g.,
            `torch.optim.lr_scheduler.StepLR`).
        lr_scheduler_kwargs : dict, optional
            Additional kwargs passed to the scheduler constructor.
        use_sliding_inference : bool
            Whether to use sliding window inference for large images on validation and test.
        sliding_window_stride : tuple of int
            Stride for sliding window inference (height, width).
        """
        # definindo out_indices padrao do MMSegmentation

        super().__init__(
            backbone=SetrVitBackbone(
                original_resolution=original_resolution,
                img_size=img_size,
                patch_size=patch_size,
                in_channels=in_channels,
                embed_dims=embed_dims,
                num_layers=num_layers,
                num_heads=num_heads,
                out_indices=out_indices if out_indices is not None else (9, 14, 19, 23),
                drop_rate=encoder_dropout,
                norm_type=encoder_norm_type,
                norm_params=encoder_norm_params,
                with_cls_token=cls_token,
                interpolate_mode=interpolate_mode,
                stride=encoder_stride,
                patch_norm=patch_norm,
                dilatation=dilatation,
                bias=bias,
                padding_type=padding_type,
                mlp_ratio=mlp_ratio,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rate,
                num_fcs=num_fcs,
                qkv_bias=qkv_bias,
                output_cls_token=output_cls_token,
                act_type=act_type,
                act_params=act_params if act_params is not None else dict(),
                with_cp=with_cp,
                dropout_type=dropout_type,
                dropout_params=(
                    dropout_params
                    if dropout_params is not None
                    else dict(drop_prob=0.0)
                ),
            ),
            fc=_SETRUPHead(
                in_channels=embed_dims,
                channels=decoder_channels,
                in_index=decoder_in_index,
                num_classes=num_classes,
                dropout=decoder_dropout,
                norm_type=decoder_norm_type,
                norm_params=decoder_norm_params,
                num_convs=decoder_num_convs,
                up_scale=decoder_up_scale,
                kernel_size=decoder_kernel_size,
                align_corners=decoder_align_corners,
                act_type=act_type,
                act_params=act_params if act_params is not None else dict(),
                interpolate_mode=interpolate_mode,
            ),
            loss_fn=(
                loss_fn
                if loss_fn is not None
                else nn.CrossEntropyLoss(
                    weight=(
                        torch.tensor(loss_weights) if loss_weights is not None else None
                    )
                )
            ),
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            freeze_backbone=freeze_backbone,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            learning_rate=learning_rate,
        )

        self.decoder_num_classes = num_classes
        self.img_size = img_size

        if head_lr_factor != 1:
            self.automatic_optimization = False
            self.multiple_optimizers = True
        else:
            self.automatic_optimization = True
            self.multiple_optimizers = False

        # definingo pesos nas losses do aux_head usado no MMSegmentation
        if aux_weights is None:
            aux_weights = [0.3, 0.3, 0.3]
            warnings.warn(f"aux_weights using values [{aux_weights}].")

        self.num_classes = num_classes
        self.aux_weights = aux_weights
        self.head_lr_factor = head_lr_factor

        self.use_sliding_inference = use_sliding_inference

        if use_sliding_inference:
            assert (
                sliding_window_stride is not None
            ), "sliding_window_stride must be provided when use_sliding_inference is True."
            self.sliding_window_stride = sliding_window_stride

        if out_indices is None:
            assert out_indices is not None, "encoder_out_indices must be provided."

        self.encoder_out_indices = out_indices

        # definindo aux_heads decoder do SETR (do MMSegmentation). PS: a diferença é o in_index, num_convs e up_scale
        self.aux_head1 = _SETRUPHead(
            in_channels=embed_dims,
            channels=decoder_channels,
            in_index=aux_heads_in_index[0],
            num_classes=num_classes,
            dropout=decoder_dropout,
            norm_type=decoder_norm_type,
            norm_params=decoder_norm_params,
            num_convs=aux_head_num_convs,
            up_scale=aux_head_up_scale,
            kernel_size=decoder_kernel_size,
            align_corners=decoder_align_corners,
            act_type=act_type,
            act_params=act_params if act_params is not None else dict(),
            interpolate_mode=interpolate_mode,
        )
        self.aux_head2 = _SETRUPHead(
            in_channels=embed_dims,
            channels=decoder_channels,
            in_index=aux_heads_in_index[1],
            num_classes=num_classes,
            dropout=decoder_dropout,
            norm_type=decoder_norm_type,
            norm_params=decoder_norm_params,
            num_convs=aux_head_num_convs,
            up_scale=aux_head_up_scale,
            kernel_size=decoder_kernel_size,
            align_corners=decoder_align_corners,
            act_type=act_type,
            act_params=act_params if act_params is not None else dict(),
            interpolate_mode=interpolate_mode,
        )
        self.aux_head3 = _SETRUPHead(
            in_channels=embed_dims,
            channels=decoder_channels,
            in_index=aux_heads_in_index[2],
            num_classes=num_classes,
            dropout=decoder_dropout,
            norm_type=decoder_norm_type,
            norm_params=decoder_norm_params,
            num_convs=aux_head_num_convs,
            up_scale=aux_head_up_scale,
            kernel_size=decoder_kernel_size,
            align_corners=decoder_align_corners,
            act_type=act_type,
            act_params=act_params if act_params is not None else dict(),
            interpolate_mode=interpolate_mode,
        )

    def forward(self, x: torch.Tensor):
        if self.encoder_out_indices is not None:
            # x é a ultima camada e aux_results são as camadas definidas por out_indices
            # PS: no forward() do decoder busca o in_index definido para cada aux_head
            x, aux_results = self.backbone(x)
            x_aux1 = self.aux_head1(aux_results)  # usa in_index 0
            x_aux2 = self.aux_head2(aux_results)  # usa in_index 1
            x_aux3 = self.aux_head3(aux_results)  # usa in_index 2
            # usa in_index 3 (no caso a ultima camada, que é o proprio x)
            x = self.fc(aux_results)
            return x, x_aux1, x_aux2, x_aux3

        x, aux_results = self.backbone(x)
        x = self.fc(x)
        return x

    def _loss_func(
        self,
        y_hat: Union[
            torch.Tensor,
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ],
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the loss between the output and the input data.

        Parameters
        ----------
        y_hat : torch.Tensor
            The output data from the forward pass.
        y : torch.Tensor
            The input data/label.

        Returns
        -------
        torch.Tensor
            The loss value.
        """

        y = y.squeeze(1) if y.ndim == 4 else y

        if isinstance(y_hat, tuple):
            y_hat, y_aux1, y_aux2, y_aux3 = y_hat
            loss = self.loss_fn(y_hat, y.long())
            loss_aux1 = self.loss_fn(y_aux1, y.long())
            loss_aux2 = self.loss_fn(y_aux2, y.long())
            loss_aux3 = self.loss_fn(y_aux3, y.long())
            return (
                loss
                + (loss_aux1 * self.aux_weights[0])
                + (loss_aux2 * self.aux_weights[1])
                + (loss_aux3 * self.aux_weights[2])
            )
        loss = self.loss_fn(y_hat, y.long())

        return loss

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        if self.multiple_optimizers:
            optimizers_list = self.optimizers()

            for opt in optimizers_list:
                opt.zero_grad()

            loss = self._single_step(batch, batch_idx, "train")

            self.manual_backward(loss)

            for opt in optimizers_list:
                opt.step()
        else:
            return self._single_step(batch, batch_idx, "train")

    def _slide_inference(
        self,
        image: np.ndarray,
        crop_size=(512, 512),
        stride=(341, 341),
        ori_shape: Optional[Tuple[int, int]] = None,
    ):
        """Realiza inferência por janelamento (sliding window) com reconstrução e resize final opcional."""
        h, w, _ = image.shape
        stride_h, stride_w = stride
        crop_h, crop_w = crop_size

        num_classes = self.decoder_num_classes  # precisa estar definido no modelo
        preds = torch.zeros((num_classes, h, w), dtype=torch.float32).to(self.device)
        count_mat = torch.zeros((h, w), dtype=torch.float32).to(self.device)

        for y in range(0, h, stride_h):
            for x in range(0, w, stride_w):
                y1 = min(y, h - crop_h)
                x1 = min(x, w - crop_w)
                y2 = y1 + crop_h
                x2 = x1 + crop_w

                patch = image[y1:y2, x1:x2, :]
                patch = patch.astype(np.float32)
                patch = (
                    torch.from_numpy(patch)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .to(self.device)
                )  # (1, C, H, W)

                with torch.no_grad():
                    logits_x, logits_aux1, logits_aux2, logits_aux3 = self.forward(
                        patch
                    )  # (1, num_classes, H, W)

                preds[:, y1:y2, x1:x2] += logits_x.squeeze(0)
                count_mat[y1:y2, x1:x2] += 1

        preds = preds / count_mat.unsqueeze(0)  # média dos logits

        # Redimensionar para forma original, se ori_shape for fornecido
        if ori_shape is not None:
            preds = preds.unsqueeze(0)  # (1, C, H, W)

            preds = F.interpolate(
                preds, size=ori_shape[-2:], mode="bilinear", align_corners=False
            )
            preds = preds.squeeze(0)

        pred_mask = preds.argmax(dim=0).cpu().numpy().astype(np.uint8)
        return pred_mask

    def _eval_step_with_slide(self, batch, step_name: str):
        img, gt = batch

        preds = []
        for i in range(img.shape[0]):
            img_np = img[i].permute(1, 2, 0).cpu().numpy()
            pred = self._slide_inference(
                img_np,
                crop_size=self.img_size,
                stride=self.sliding_window_stride,
                ori_shape=gt.shape,
            )
            preds.append(torch.from_numpy(pred))

        preds = torch.stack(preds, dim=0).to(self.device)
        gt = gt.squeeze(1).long()

        metrics = self._compute_metrics(preds, gt, step_name)
        for metric_name, metric_value in metrics.items():
            self.log(
                metric_name,
                metric_value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

    def _compute_metrics(
        self, y_hat: torch.Tensor, y: torch.Tensor, step_name: str
    ) -> Dict[str, torch.Tensor]:
        """Calculate the metrics for the given step.

        Parameters
        ----------
        y_hat : torch.Tensor
            The output data from the forward pass.
        y : torch.Tensor
            The input data/label.
        step_name : str
            Name of the step. It will be used to get the metrics from the
            `self.metrics` attribute.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary with the metrics values.
        """
        if self.metrics[step_name] is None:
            return {}

        if isinstance(y_hat, (tuple, list)):
            y_hat = y_hat[0]  # Keep only logits for metrics

        # Convert logits to predicted class indices
        y_hat_classes = torch.argmax(y_hat, dim=1)  # [N, H, W]

        # Remove extra channel from target if present
        if y.ndim == 4 and y.shape[1] == 1:
            y = y.squeeze(1)

        return {
            f"{step_name}_{metric_name}": metric.to(self.device)(y_hat, y)
            for metric_name, metric in self.metrics[step_name].items()
        }

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        return (
            self._eval_step_with_slide(batch, "val")
            if self.use_sliding_inference
            else self._single_step(batch, batch_idx, "val")
        )

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        return (
            self._eval_step_with_slide(batch, "test")
            if self.use_sliding_inference
            else self._single_step(batch, batch_idx, "test")
        )

    def predict_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ):
        x, _ = batch
        return self.forward(x)[0]

    def load_backbone(self, path: str, freeze: bool = False):
        """Loads pretrained ViT backbone optionally freezing its weights."""
        self.backbone.load_backbone(path)
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def configure_optimizers(self):
        # Freeze or not the backbone model
        for param in self.backbone.parameters():
            param.requires_grad = not self.freeze_backbone
        # Unfreeze the fc model
        for param in self.fc.parameters():
            param.requires_grad = True

        if self.multiple_optimizers:
            optimizers = [
                self.optimizer(
                    self.backbone.parameters(),
                    lr=self.learning_rate,
                    **self.optimizer_kwargs,
                ),
                self.optimizer(
                    list(self.fc.parameters())
                    + list(self.aux_head1.parameters())
                    + list(self.aux_head2.parameters())
                    + list(self.aux_head3.parameters()),
                    lr=self.learning_rate * self.head_lr_factor,
                    **self.optimizer_kwargs,
                ),
            ]
            if self.lr_scheduler is None:
                return optimizers
            schedulers = [
                self.lr_scheduler(optimizers[0], **self.lr_scheduler_kwargs),
                self.lr_scheduler(optimizers[1], **self.lr_scheduler_kwargs),
            ]
            return optimizers, schedulers
        else:
            optimizer = self.optimizer_type(
                self.parameters(),
                lr=self.learning_rate,
                **self.optimizer_params,
            )
            if self.lr_scheduler is None:
                return optimizer
            scheduler = self.lr_scheduler(optimizer, **self.lr_scheduler_kwargs)
            return optimizer, scheduler

    @staticmethod
    def create_from_dict(config: Dict) -> "SETR_PUP":
        return SETR_PUP(**config)
