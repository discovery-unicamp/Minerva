from typing import Optional, Tuple

import torch
from torch import nn

from sslt.utils.upsample import Upsample


class SETRUPHead(nn.Module):
    """Naive upsampling head and Progressive upsampling head of SETR.

    Naive or PUP head of `SETR  <https://arxiv.org/pdf/2012.15840.pdf>`_.

    """

    def __init__(
        self,
        norm_layer: nn.Module,
        conv_norm: nn.Module,
        conv_act: nn.Module,
        in_channels: int,
        out_channels: int,
        size: Optional[Tuple[int, int]] = None,
        num_convs: int = 1,
        up_scale: int = 4,
        kernel_size: int = 3,
        align_corners: bool = False,
    ):

        assert kernel_size in [1, 3], "kernel_size must be 1 or 3."

        super().__init__()

        self.size = size
        self.norm = norm_layer
        self.conv_norm = conv_norm
        self.conv_act = conv_act
        self.in_channels = in_channels
        self.channels = out_channels
        self.align_corners = align_corners

        self.up_convs = nn.ModuleList()
        in_channels = self.in_channels
        out_channels = self.channels
        for _ in range(num_convs):
            self.up_convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size,
                        padding=kernel_size // 2,
                        bias=False,
                    ),
                    self.conv_norm,
                    self.conv_act,
                    Upsample(
                        scale_factor=up_scale,
                        mode="bilinear",
                        align_corners=self.align_corners,
                    ),
                )
            )
            in_channels = out_channels

    def forward(self, x):
        x = self._transform_inputs(x)

        n, c, h, w = x.shape
        x = x.reshape(n, c, h * w).transpose(2, 1).contiguous()
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(n, c, h, w).contiguous()

        for up_conv in self.up_convs:
            x = up_conv(x)
        out = self.cls_seg(x)
        return out


class SETRMLAHead(nn.Module):
    """Multi level feature aggretation head of SETR.

    MLA head of `SETR  <https://arxiv.org/pdf/2012.15840.pdf>`_.

    Args:
        mlahead_channels (int): Channels of conv-conv-4x of multi-level feature
            aggregation. Default: 128.
        up_scale (int): The scale factor of interpolate. Default:4.
    """

    def __init__(
        self,
        mla_channels=128,
        up_scale=4,
    ):
        super().__init__(input_transform="multiple_select")
        self.mla_channels = mla_channels

        num_inputs = len(self.in_channels)

        # Refer to self.cls_seg settings of BaseDecodeHead
        assert self.channels == num_inputs * mla_channels

        self.up_convs = nn.ModuleList()
        for i in range(num_inputs):
            self.up_convs.append(
                nn.Sequential(
                    ConvModule(
                        in_channels=self.in_channels[i],
                        out_channels=mla_channels,
                        kernel_size=3,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                    ),
                    ConvModule(
                        in_channels=mla_channels,
                        out_channels=mla_channels,
                        kernel_size=3,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                    ),
                    Upsample(
                        scale_factor=up_scale,
                        mode="bilinear",
                        align_corners=self.align_corners,
                    ),
                )
            )

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        outs = []
        for x, up_conv in zip(inputs, self.up_convs):
            outs.append(up_conv(x))
        out = torch.cat(outs, dim=1)
        out = self.cls_seg(out)
        return out
