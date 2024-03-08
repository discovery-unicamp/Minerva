import warnings
from typing import Optional, Tuple

import torch
from torch import nn

from sslt.models.nets.base import SimpleSupervisedModel
from sslt.models.nets.vit import _VisionTransformerBackbone
from sslt.utils.upsample import Upsample, resize


class _SETRUPHead(nn.Module):
    """Naive upsampling head and Progressive upsampling head of SETR.

    Naive or PUP head of `SETR  <https://arxiv.org/pdf/2012.15840.pdf>`_.

    """

    def __init__(
        self,
        channels: int,
        norm_layer: Optional[nn.Module],
        conv_norm: Optional[nn.Module],
        conv_act: Optional[nn.Module],
        in_channels: int,
        num_classes: int,
        num_convs: int = 1,
        up_scale: int = 4,
        kernel_size: int = 3,
        align_corners: bool = True,
        dropout: float = 0.1,
        threshold: Optional[float] = None,
    ):

        assert kernel_size in [1, 3], "kernel_size must be 1 or 3."

        super().__init__()

        self.num_classes = num_classes
        self.out_channels = channels
        self.threshold = threshold
        self.cls_seg = nn.Conv2d(channels, self.num_classes, 1)
        self.norm = norm_layer if norm_layer is not None else nn.LayerNorm(in_channels)
        conv_norm = (
            conv_norm if conv_norm is not None else nn.SyncBatchNorm(self.out_channels)
        )
        conv_act = conv_act if conv_act is not None else nn.ReLU()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 != None else None

        self.up_convs = nn.ModuleList()

        for _ in range(num_convs):
            self.up_convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        self.out_channels,
                        kernel_size,
                        padding=kernel_size // 2,
                        bias=False,
                    ),
                    conv_norm,
                    conv_act,
                    Upsample(
                        scale_factor=up_scale,
                        mode="bilinear",
                        align_corners=align_corners,
                    ),
                )
            )
            in_channels = self.out_channels

    def forward(self, x):

        x = self.norm(x)

        for up_conv in self.up_convs:
            print(x.shape)
            x = up_conv(x)

        if self.dropout is not None:
            x = self.dropout(x)
        out = self.cls_seg(x)

        return out


class _SETRMLAHead(nn.Module):
    """Multi level feature aggretation head of SETR.

    MLA head of `SETR  <https://arxiv.org/pdf/2012.15840.pdf>`_.
    """

    def __init__(
        self,
        channels: int,
        conv_norm: Optional[nn.Module],
        conv_act: Optional[nn.Module],
        in_channels: list[int],
        out_channels: int,
        num_classes: int,
        mla_channels: int = 128,
        up_scale: int = 4,
        kernel_size: int = 3,
        align_corners: bool = True,
        dropout: float = 0.1,
        threshold: Optional[float] = None,
    ):
        super().__init__()

        if out_channels is None:
            if num_classes == 2:
                warnings.warn(
                    "For binary segmentation, we suggest using"
                    "`out_channels = 1` to define the output"
                    "channels of segmentor, and use `threshold`"
                    "to convert `seg_logits` into a prediction"
                    "applying a threshold"
                )
            out_channels = num_classes

        if out_channels != num_classes and out_channels != 1:
            raise ValueError(
                "out_channels should be equal to num_classes,"
                "except binary segmentation set out_channels == 1 and"
                f"num_classes == 2, but got out_channels={out_channels}"
                f"and num_classes={num_classes}"
            )

        if out_channels == 1 and threshold is None:
            threshold = 0.3
            warnings.warn("threshold is not defined for binary, and defaults to 0.3")

        self.num_classes = num_classes
        self.out_channels = out_channels
        self.threshold = threshold
        conv_norm = (
            conv_norm if conv_norm is not None else nn.SyncBatchNorm(mla_channels)
        )
        conv_act = conv_act if conv_act is not None else nn.ReLU()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 != None else None
        self.cls_seg = nn.Conv2d(channels, out_channels, 1)

        num_inputs = len(in_channels)

        self.up_convs = nn.ModuleList()
        for i in range(num_inputs):
            self.up_convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels[i],
                        mla_channels,
                        kernel_size,
                        padding=kernel_size // 2,
                        bias=False,
                    ),
                    conv_norm,
                    conv_act,
                    nn.Conv2d(
                        mla_channels,
                        mla_channels,
                        kernel_size,
                        padding=kernel_size // 2,
                        bias=False,
                    ),
                    conv_norm,
                    conv_act,
                    Upsample(
                        scale_factor=up_scale,
                        mode="bilinear",
                        align_corners=align_corners,
                    ),
                )
            )

    def forward(self, x):
        outs = []
        for x, up_conv in zip(x, self.up_convs):
            outs.append(up_conv(x))
        out = torch.cat(outs, dim=1)
        if self.dropout is not None:
            out = self.dropout(out)
        out = self.cls_seg(out)
        return out


class _SetR_PUP(nn.Module):

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        norm_layer: Optional[nn.Module] = None,
        interpolate_mode: str = "bilinear",
    ):
        super().__init__()
        self.encoder = _VisionTransformerBackbone(
            image_size=image_size,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

        self.decoder = _SETRUPHead(
            channels=256,
            in_channels=hidden_dim,
            num_classes=6,
            num_convs=4,
            up_scale=2,
            kernel_size=3,
            align_corners=False,
            dropout=0,
            norm_layer=norm_layer,
            conv_norm=None,  # Add default value for conv_norm
            conv_act=None,  # Add default value for conv_act
        )

        self.aux_head1 = _SETRUPHead(
            channels=1024,
            in_channels=hidden_dim,
            num_classes=6,
            num_convs=2,
            up_scale=2,
            kernel_size=3,
            align_corners=False,
            dropout=0,
            norm_layer=norm_layer,
            conv_norm=None,  # Add default value for conv_norm
            conv_act=None,  # Add default value for conv_act
        )

        self.aux_head2 = _SETRUPHead(
            channels=256,
            in_channels=hidden_dim,
            num_classes=6,
            num_convs=2,
            up_scale=2,
            kernel_size=3,
            align_corners=False,
            dropout=0,
            norm_layer=norm_layer,
            conv_norm=None,  # Add default value for conv_norm
            conv_act=None,  # Add default value for conv_act
        )

        self.aux_head3 = _SETRUPHead(
            channels=256,
            in_channels=hidden_dim,
            num_classes=6,
            num_convs=2,
            up_scale=2,
            kernel_size=3,
            align_corners=False,
            dropout=0,
            norm_layer=norm_layer,
            conv_norm=None,  # Add default value for conv_norm
            conv_act=None,  # Add default value for conv_act
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.encoder(x)
        x_aux1 = self.aux_head1(x)
        x_aux2 = self.aux_head2(x)
        x_aux3 = self.aux_head3(x)
        x = self.decoder(x)
        return x, x_aux1, x_aux2, x_aux3
