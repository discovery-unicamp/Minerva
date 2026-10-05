import torch.nn as nn


class Conv(nn.Module):
    """1D Convolutional layer with weight normalization.

    This module wraps a `Conv1d` operation with appropriate padding for dilations
    and applies Kaiming initialization and weight normalization for stability.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int, optional
        Size of the convolution kernel. Default is 3.
    dilation : int, optional
        Dilation rate of the convolution. Default is 1.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(Conv, self).__init__()
        self.padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=self.padding,
        )
        self.conv = nn.utils.parametrizations.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        """Apply the convolution to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C_in, L).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, C_out, L).
        """
        out = self.conv(x)
        return out


class ZeroConv1d(nn.Module):
    """1x1 Convolution layer initialized with zeros.

    Used as the final layer in diffusion models to ensure
    the network initially outputs zeros, stabilizing training.

    Parameters
    ----------
    in_channel : int
        Number of input channels.
    out_channel : int
        Number of output channels.
    """

    def __init__(self, in_channel, out_channel):
        super(ZeroConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=1, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

    def forward(self, x):
        out = self.conv(x)
        return out
