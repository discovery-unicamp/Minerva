"""Full assembly of the parts to form the complete network"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from minerva.models.nets.base import SimpleSupervisedModel

""" -------------- Parts of the U-Net model --------------"""


class _DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    """
    Performs two convolutions with the same number
    of input and output channels, followed by batch normalization and ReLU activation
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        """

        Parameters
        ----------
        in_channels : int
            Number of input channels, i.e. the number of channels in the input image (1 for grayscale, 3 for RGB)
        out_channels : int
            Number of output channels, i.e. the number of channels produced by the convolution
        mid_channels : int, optional
            Number of channels in the middle, by default None

        """
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, padding=1, bias=False
            ),  # no need to add bias since BatchNorm2d will do that
            nn.BatchNorm2d(mid_channels),  # normalize the output of the previous layer
            nn.ReLU(
                inplace=True
            ),  # inplace=True will modify the input directly instead of allocating new memory
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class _Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), _DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class _Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = _DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = _DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW (channel, height, width)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # pad the input tensor on all sides with the given "pad" value
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class _OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


""" -------------- The U-Net model --------------"""


class _UNet(torch.nn.Module):
    """Implementation of U-Net model."""

    def __init__(
        self,
        n_channels: int = 1,
        bilinear: bool = False,
    ):
        """Implementation of U-Net model.

        Parameters
        ----------
        n_channels : int, optional
            Number of input channels, by default 1
        bilinear : bool, optional
            If `True` use bilinear interpolation for upsampling, by default
            False.
        """
        super().__init__()
        factor = 2 if bilinear else 1

        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = _DoubleConv(n_channels, 64)
        self.down1 = _Down(64, 128)
        self.down2 = _Down(128, 256)
        self.down3 = _Down(256, 512)
        self.down4 = _Down(512, 1024 // factor)
        self.up1 = _Up(1024, 512 // factor, bilinear)
        self.up2 = _Up(512, 256 // factor, bilinear)
        self.up3 = _Up(256, 128 // factor, bilinear)
        self.up4 = _Up(128, 64, bilinear)
        # self.outc = (OutConv(64, n_classes))
        self.outc = _OutConv(64, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class UNet(SimpleSupervisedModel):
    """This class is a simple implementation of the U-Net model, which is a
    convolutional neural network used for image segmentation. The model consists
    of a contracting path (encoder) and an expansive path (decoder). The
    contracting path follows the typical architecture of a convolutional neural
    network, with repeated applications of convolutions and max pooling layers.
    The expansive path consists of up-convolutions and concatenation of feature
    maps from the contracting path. The model also has skip connections, which
    allows the expansive path to use information from the contracting path at
    multiple resolutions. The U-Net model was originally proposed by
    Ronneberger, Fischer, and Brox in 2015.

    This architecture, handles arbitrary input sizes, and returns an output of
    the same size as the input. The expected input size is (N, C, H, W), where N
    is the batch size, C is the number of channels, H is the height of the input
    image, and W is the width of the input image.

    Note that, for this implementation, the input batch is a single tensor and
    not a tuple of tensors (e.g., data and label).

    Note that this class wrappers the `_UNet` class, which is the actual
    implementation of the U-Net model, into a `SimpleReconstructionNet` class,
    which is a simple autoencoder pipeline for reconstruction tasks.

    References
    ----------
    Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional
    networks for biomedical image segmentation." Medical Image Computing and
    Computer-Assisted Intervention-MICCAI 2015: 18th International Conference,
    Munich, Germany, October 5-9, 2015, Proceedings, Part III 18. Springer
    International Publishing, 2015.
    """

    def __init__(
        self,
        n_channels: int = 1,
        bilinear: bool = False,
        learning_rate: float = 1e-3,
        loss_fn: Optional[torch.nn.Module] = None,
        **kwargs,
    ):
        """Wrapper implementation of the U-Net model.

        Parameters
        ----------
        n_channels : int, optional
            The number of channels of the input, by default 1
        bilinear : bool, optional
            If `True` use bilinear interpolation for upsampling, by default
            False.
        learning_rate : float, optional
            The learning rate to Adam optimizer, by default 1e-3
        loss_fn : torch.nn.Module, optional
            The function used to compute the loss. If `None`, it will be used
            the MSELoss, by default None.
        kwargs : Dict
            Additional arguments to be passed to the `SimpleSupervisedModel`
            class.
        """
        super().__init__(
            backbone=_UNet(n_channels=n_channels, bilinear=bilinear),
            fc=torch.nn.Identity(),
            loss_fn=loss_fn or torch.nn.MSELoss(),
            learning_rate=learning_rate,
            flatten=False,
            **kwargs,
        )
