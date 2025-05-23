import torch
from numpy import linspace
from typing import Callable, List, Optional


class ConvTAEEncoder(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        time_steps: int = 60,
        encoding_size: int = 256,
        fc_num_layers: int = 3,
        conv_num_layers: int = 3,
        conv_mid_channels: int = 12,
        conv_kernel: int = 5,
        conv_padding: int = 0,
    ):
        """
        An encoder for a simple convolutional autoencoder.

        Parameters
        ----------
        in_channels : int, optional
            Number of channels of the input that the model receives, by default 2
        time_steps : int, optional
            Number of time steps of the input that the model receives, by default 60
        encoding_size : int, optional
            Size of the data representation generated by the model, by default 256
        fc_num_layers : int, optional
            Number of fully connected layers, by default 3
        conv_num_layers : int, optional
            Number of convolutional layers, by default 3
        conv_mid_channels : int, optional
            Number of channels used for in_channels and out_channels in the convolutional layers, except in the first, by default 12
        conv_kernel : int, optional
            Size of the convolutional kernel, by default 5
        conv_padding : int, optional
            Padding used in the convolutional layers, by default 0
        """
        super(ConvTAEEncoder, self).__init__()
        # Saving parameters
        self.in_channels = in_channels
        self.time_steps = time_steps
        self.encoding_size = encoding_size
        self.fc_num_layers = fc_num_layers
        self.conv_num_layers = conv_num_layers
        self.conv_mid_channels = conv_mid_channels
        self.conv_kernel = conv_kernel
        self.conv_padding = conv_padding

        # Simulation function
        def l_out(l_in, kernel, padding, stride=1):
            return (l_in - kernel + 2 * padding) // stride + 1

        # Defining layers
        layers = []
        # Convolutional layers
        conv_input_channels = in_channels
        current_time_steps = self.time_steps
        for _ in range(conv_num_layers):
            layers.append(
                torch.nn.Conv1d(conv_input_channels, conv_mid_channels, conv_kernel)
            )
            layers.append(torch.nn.ReLU())
            conv_input_channels = conv_mid_channels
            # Updating L_out
            current_time_steps = l_out(
                current_time_steps, self.conv_kernel, self.conv_padding
            )
        # Fully connected layers
        layers.append(torch.nn.Flatten())
        current_size = current_time_steps * conv_mid_channels
        linear_dimensions = linspace(
            current_size, encoding_size, fc_num_layers + 1
        ).astype(int)
        for i in range(fc_num_layers):
            layers.append(
                torch.nn.Linear(linear_dimensions[i], linear_dimensions[i + 1])
            )
            layers.append(torch.nn.ReLU())
        # Removing last ReLU
        layers.pop()
        # Creating model
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ConvTAEDecoder(torch.nn.Module):
    def __init__(
        self,
        target_channels: int = 6,
        target_time_steps: int = 60,
        encoding_size: int = 256,
        fc_num_layers: int = 3,
        conv_num_layers: int = 3,
        conv_mid_channels: int = 12,
        conv_kernel: int = 5,
        conv_padding: int = 0,
    ):
        """
        A decoder for a simple convolutional autodecoder.

        Parameters
        ----------
        target_channels : int, optional
            Number of channels of the output that the model should target to, by default 6
        target_time_steps : int, optional
            Number of time steps of the output that the model should target to, by default 60
        encoding_size : int, optional
            Size of the data representation received by the model, by default 256
        fc_num_layers : int, optional
            Number of fully connected layers, by default 3
        conv_num_layers : int, optional
            Number of convolutional layers, by default 3
        conv_mid_channels : int, optional
            Number of channels used for in_channels and out_channels in the convolutional layers, except in the last, by default 12
        conv_kernel : int, optional
            Size of the convolutional kernel, by default 5
        conv_padding : int, optional
            Padding used in the convolutional layers, by default 0
        """
        super(ConvTAEDecoder, self).__init__()
        # Saving parameters
        self.target_channels = target_channels
        self.target_time_steps = target_time_steps
        self.encoding_size = encoding_size
        self.fc_num_layers = fc_num_layers
        self.conv_num_layers = conv_num_layers
        self.conv_mid_channels = conv_mid_channels
        self.conv_kernel = conv_kernel
        self.conv_padding = conv_padding

        # Simulation function
        def convtranspose1d_input_required(target_output, kernel, padding):
            return target_output + 2 * padding - kernel + 1

        # Defining layers
        layers = []
        required_output = self.target_time_steps
        required_input = required_output
        output_channels = self.target_channels
        for _ in range(conv_num_layers):
            required_input = convtranspose1d_input_required(
                required_output, conv_kernel, conv_padding
            )
            layers.append(
                torch.nn.ConvTranspose1d(
                    self.conv_mid_channels,
                    output_channels,
                    kernel_size=conv_kernel,
                    padding=conv_padding,
                )
            )
            layers.append(torch.nn.ReLU())
            output_channels = self.conv_mid_channels
            required_output = required_input
        # Layer to resize it to the correct shape
        layers.append(torch.nn.Unflatten(1, (self.conv_mid_channels, required_output)))
        # Fully connected layers
        linear_dimensions = linspace(
            self.encoding_size,
            required_input * self.conv_mid_channels,
            fc_num_layers + 1,
        ).astype(int)
        # Reverse the list
        linear_dimensions = linear_dimensions[::-1]
        for i in range(fc_num_layers):
            layers.append(
                torch.nn.Linear(linear_dimensions[i + 1], linear_dimensions[i])
            )
            layers.append(torch.nn.ReLU())
        # Removing last ReLU
        layers.pop()
        # Inverting layers
        layers = layers[::-1]
        # Creating model
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
