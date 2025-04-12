from typing import Literal
import lightning as L
import torch
import torch.nn.functional as F
import numpy as np
from minerva.models.nets.time_series.resnet import _ResNet1D

# RNN encoder used by tonekaboni


class RnnEncoder(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        in_channel: int,
        encoding_size: int,
        cell_type: str = "GRU",
        num_layers: int = 1,
        device: str = "cpu",
        dropout: int = 0,
        bidirectional: bool = True,
        permute: bool = False,
        squeeze: bool = True,
    ):
        """
        Initializes an RnnEncoder instance.

        This encoder utilizes a recurrent neural network (RNN) to encode sequential data,
        such as accelerometer and gyroscope readings from human activity recognition tasks.

        Parameters
        ----------
        hidden_size : int
            Size of the hidden state in the RNN.
        in_channel : int
            Number of input channels (e.g., dimensions of accelerometer and gyroscope data).
        encoding_size : int
            Desired size of the output encoding.
        cell_type : str, optional
            Type of RNN cell to use (default is 'GRU'). Options include 'GRU', 'LSTM', etc.
        num_layers : int, optional
            Number of RNN layers (default is 1).
        device : str, optional
            Device to run the model on (default is 'cpu'). Options include 'cpu' and 'cuda'.
        dropout : float, optional
            Dropout probability (default is 0.0).
        bidirectional : bool, optional
            Whether the RNN is bidirectional (default is True).
        permute: bool, optional
            If `True` the input data will be permuted before passing through the model, by default False.
        squeeze: bool, optional
            If `True`, the outputs of RNN states is squeezed before passed to Linear layer.
            By default True.

        Examples
        --------
        >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        >>> encoder = RnnEncoder(hidden_size=100, in_channel=6, encoding_size=320,
                                 cell_type='GRU', num_layers=1, device=device,
                                 dropout=0.0, bidirectional=True).to(device)
        >>> element1 = torch.randn(32, 50, 6)  # Batch size: 32, Time steps: 50, Input channels: 6
        >>> encoding = encoder(element1.to(device))
        >>> print(encoding.shape)
        torch.Size([32, 320])

        Notes
        -----
        - The input tensor should have the shape (batch_size, time_steps, in_channel).
        - The output tensor will have the shape (batch_size, encoding_size).
        """
        super(RnnEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.in_channel = in_channel
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.encoding_size = encoding_size
        self.bidirectional = bidirectional
        self.device = device
        self.permute = permute
        self.squeeze = squeeze

        self.nn = torch.nn.Sequential(
            torch.nn.Linear(
                self.hidden_size * (int(self.bidirectional) + 1),
                self.encoding_size,
            )
        ).to(self.device)
        self.rnn = torch.nn.GRU(
            input_size=in_channel,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False,
            dropout=dropout,
            bidirectional=bidirectional,
        ).to(self.device)

    def forward(self, x):
        """
        Forward pass for the RnnEncoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, time_steps, in_channel).

        Returns
        -------
        torch.Tensor
            Encoded tensor of shape (batch_size, encoding_size).
        """
        if self.permute:
            x = x.permute(2, 0, 1)
        else:
            x = x.permute(1, 0, 2)
        past = torch.zeros(
            self.num_layers * (int(self.bidirectional) + 1),
            x.shape[1],
            self.hidden_size,
        ).to(self.device)
        # print(f"Input tensor shape before passing to RNN \n: {x.shape}")  # Print the shape of x
        out, _ = self.rnn(x.to(self.device), past)
        # print(f"Input tensor shape after passing to RNN :\n {out.shape}")
        if self.squeeze:
            encodings = self.nn(out[-1].squeeze(0))  # Process the output of the RNN
        else:
            encodings = self.nn(out[-1])
        # print(f"4-Output encodings shape after passing to RNN :\n {encodings.shape}")
        return encodings


# TS2Vec encoder used by xu


class ResNetEncoder(_ResNet1D):
    def forward(self, x):
        return super().forward(x.permute(0, 2, 1))


class DilatedConvEncoder(torch.nn.Module):
    def __init__(self, in_channels: int, channels: list, kernel_size: int):
        """
        This module implements a stack of dilated convolutional blocks for feature extraction
        from sequential data.

        Parameters:
        -----------
        - in_channels (int):
            Number of input channels to the first convolutional layer.
        - channels (list):
            List of integers specifying the number of output channels for each convolutional layer.
        - kernel_size (int):
            Size of the convolutional kernel.
        """
        super().__init__()
        self.net = torch.nn.Sequential(
            *[
                ConvBlock(
                    channels[i - 1] if i > 0 else in_channels,
                    channels[i],
                    kernel_size=kernel_size,
                    dilation=2**i,
                    final=(i == len(channels) - 1),
                )
                for i in range(len(channels))
            ]
        )

    def forward(self, x):
        return self.net(x)


class ConvBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        final: bool = False,
    ):
        """
        A single block of dilated convolutional layers followed by a residual connection and activation.

        Parameters:
        -----------
        - in_channels (int):
            Number of input channels to the first convolutional layer.
        - out_channels (int):
            Number of output channels from the final convolutional layer.
        - kernel_size (int):
            Size of the convolutional kernel.
        - dilation (int):
            Dilation factor for the convolutional layers.
        - final (bool, optional):
            Whether this is the final block in the sequence (default: False).

        """
        super().__init__()
        self.conv1 = SamePadConv(
            in_channels, out_channels, kernel_size, dilation=dilation
        )
        self.conv2 = SamePadConv(
            out_channels, out_channels, kernel_size, dilation=dilation
        )
        self.projector = (
            torch.nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels or final
            else None
        )

    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = torch.nn.functional.gelu(x)
        x = self.conv1(x)
        x = torch.nn.functional.gelu(x)
        x = self.conv2(x)
        return x + residual


class SamePadConv(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        groups: int = 1,
    ):
        """
        Purpose:
        -------
        Implements a convolutional layer with padding to maintain the same output size as the input.

        Parameters:
        -----------
        - in_channels (int):
            Number of input channels to the convolutional layer.
        - out_channels (int):
            Number of output channels from the convolutional layer.
        - kernel_size (int):
            Size of the convolutional kernel.
        - dilation (int, optional):
            Dilation factor for the convolutional layer (default: 1).
        - groups (int, optional):
            Number of blocked connections from input channels to output channels (default: 1).
        """
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0

    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out


# Discriminator aka projection head


class TSEncoder(torch.nn.Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        hidden_dims: int = 64,
        depth: int = 10,
        permute: bool = False,
        encoder_cls: type = DilatedConvEncoder,
        encoder_cls_kwargs: dict = {},
    ):
        """
        Encoder utilizing dilated convolutional layers for encoding sequential data.

        Parameters
        ----------
        input_dims : int
            Dimensionality of the input features.
        output_dims : int
            Desired dimensionality of the output features.
        hidden_dims : int, optional
            Number of hidden dimensions in the convolutional layers (default is 64).
        depth : int, optional
            Number of convolutional layers (default is 10).
        - permute : bool, optional
            If `True` the input data will be permuted before passing through
            the model, by default False. This should be removed after the encoder
            receives data in the shape (bs, channels, timesteps)

        Examples
        --------
        >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        >>> encoder = TSEncoder(input_dims=6, output_dims=320, hidden_dims=64, depth=10).to(device)
        >>> element1 = torch.randn(12, 128, 6)  # Batch size: 12, Time steps: 128, Input channels: 6
        >>> encoded_features = encoder(element1.to(device))
        >>> print(encoded_features.shape)
        torch.Size([12, 128, 320])

        Notes
        -----
        - The input tensor should have the shape (batch_size, seq_len, input_dims).
        - The output tensor will have the shape (batch_size, seq_len, output_dims).
        - If the expected output tensor is of shape (batch_size, output_dims), consider using a pooling layer.
        One option is to use the `MaxPoolingTransposingSqueezingAdapter` adapter. at minerva/models/adapters.py
        """
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.input_fc = torch.nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = encoder_cls(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3,
            **encoder_cls_kwargs,
        )
        self.repr_dropout = torch.nn.Dropout(p=0.1)
        self.permute = permute

    def forward(self, x, mask=None):
        """
        Forward pass of the encoder.

        Parameters:
        -----------
        - x (torch.Tensor):
            Input tensor of shape (batch_size, seq_len, input_dims).
        - mask (str, optional):
            Type of masking to apply (default: None).

        Returns:
        --------
        - torch.Tensor:
            Encoded features of shape (batch_size, seq_len, output_dims).
        """
        if self.permute:
            x = x.permute(0, 2, 1)
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0

        x = self.input_fc(x)

        if mask == "binomial":
            mask = torch.from_numpy(
                np.random.binomial(1, 0.5, size=(x.size(0), x.size(1)))
            ).to(x.device)
            mask &= nan_mask
            x[~mask] = 0

        x = x.transpose(1, 2)  # B x Ch x T
        # print("shape of x before feature extractor",x.shape)
        x = self.repr_dropout(self.feature_extractor(x))  # B x Co x T
        # print("shape of x after feature extractor",x.shape)
        x = x.transpose(1, 2)  # B x T x Co

        return x


class Discriminator_TNC(torch.nn.Module):
    def __init__(self, input_size: int, max_pool: bool = False):
        """
        A discriminator model used for contrastive learning tasks, predicting whether two inputs belong
        to the same neighborhood in the feature space.

        Parameters
        ----------
        input_size : int
            Dimensionality of each input.
        max_pool : bool, optional
            Whether to apply max pooling before feeding into the projection head (default is False).
            If using TS2Vec encoder, set to True; if using RNN, set to False.

        Examples
        --------
        >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        >>> discriminator = Discriminator_TNC(input_size=320, max_pool=True).to(device)
        >>> forward_ts2vec1 = torch.randn(12, 128, 320)  # Example tensor with shape (batch_size, timesteps, encoding_size)
        >>> forward_ts2vec3 = torch.randn(12, 128, 320)  # Another example tensor with shape (batch_size, timesteps, encoding_size)
        >>> output = discriminator(forward_ts2vec1, forward_ts2vec3)
        >>> print(output.shape)
        torch.Size([12])

        >>> # Example with RNN encoder
        >>> rnn_encoder = RnnEncoder(hidden_size=100, in_channel=6, encoding_size=320, cell_type='GRU', num_layers=1, device=device, dropout=0.0, bidirectional=True).to(device)
        >>> element1 = torch.randn(12, 128, 6)  # Batch size: 12, Time steps: 128, Input channels: 6
        >>> forward_rnn1 = rnn_encoder(element1.to(device))
        >>> forward_rnn2 = rnn_encoder(element1.to(device))
        >>> discriminator = Discriminator_TNC(input_size=320, max_pool=False).to(device)
        >>> output = discriminator(forward_rnn1, forward_rnn2)
        >>> print(output.shape)
        torch.Size([12])

        Notes
        -----
        - The input tensors should have the shape (batch_size, input_size).
        - The output tensor will have the shape (batch_size,) representing the predicted probabilities.
        """
        super(Discriminator_TNC, self).__init__()
        self.input_size = input_size
        self.max_pool = max_pool

        self.model = torch.nn.Sequential(
            torch.nn.Linear(2 * self.input_size, 4 * self.input_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4 * self.input_size, 1),
        )

        torch.nn.init.xavier_uniform_(self.model[0].weight)
        torch.nn.init.xavier_uniform_(self.model[3].weight)

    def forward(self, x, x_tild):
        """
        Predict the probability of the two inputs belonging to the same neighborhood.

        Parameters:
        -----------
        - x (torch.Tensor):
            Input tensor of shape (batch_size, input_size).
        - x_tild (torch.Tensor):
            Input tensor of shape (batch_size, input_size).

        Returns:
        --------
        - p (torch.Tensor):
            Output tensor of shape (batch_size,) representing the predicted probabilities.
        """
        x_all = torch.cat([x, x_tild], -1)
        if self.max_pool:
            x_all = F.max_pool1d(
                x_all.transpose(1, 2).contiguous(), kernel_size=x_all.size(1)
            ).transpose(1, 2)

        p = self.model(x_all)
        return p.view((-1,))
