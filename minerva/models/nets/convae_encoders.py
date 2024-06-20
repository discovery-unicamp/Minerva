import torch
from numpy import linspace

class ConvTAEEncoder(torch.nn.Module):
    def __init__(
            self,
            in_channels=6,
            time_steps=60,
            encoding_size=256,
            fc_num_layers=3,
            conv_num_layers=3,
            conv_mid_channels=12,
            conv_kernel=5,
            conv_padding=0,
    ):
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
            layers.append(torch.nn.Conv1d(conv_input_channels, conv_mid_channels, conv_kernel))
            layers.append(torch.nn.ReLU())
            conv_input_channels = conv_mid_channels
            # Updating L_out
            current_time_steps = l_out(current_time_steps, self.conv_kernel, self.conv_padding)
        # Fully connected layers
        layers.append(torch.nn.Flatten())
        current_size = current_time_steps * conv_mid_channels
        linear_dimensions = linspace(current_size, encoding_size, fc_num_layers + 1).astype(int)
        for i in range(fc_num_layers):
            layers.append(torch.nn.Linear(linear_dimensions[i], linear_dimensions[i + 1]))
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
            target_channels=6,
            target_time_steps=60,
            encoding_size=256,
            fc_num_layers=3,
            conv_num_layers=3,
            conv_mid_channels=12,
            conv_kernel=5,
            conv_padding=0,
    ):
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
            required_input = convtranspose1d_input_required(required_output, conv_kernel, conv_padding)
            layers.append(torch.nn.Conv1d(self.conv_mid_channels, output_channels, kernel_size=conv_kernel, padding=conv_padding))
            layers.append(torch.nn.ReLU())
            output_channels = self.conv_mid_channels
            required_output = required_input
        # Fully connected layers
        linear_dimensions = linspace(self.encoding_size, required_input * self.conv_mid_channels, fc_num_layers + 1).astype(int)
        # Reverse the list
        linear_dimensions = linear_dimensions[::-1]
        for i in range(fc_num_layers):
            layers.append(torch.nn.Linear(linear_dimensions[i], linear_dimensions[i + 1]))
            layers.append(torch.nn.ReLU())
        # Removing last ReLU
        layers.pop()
        # Inverting layers
        layers = layers[::-1]
        # Creating model
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)