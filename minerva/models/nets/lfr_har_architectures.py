from torch import nn


class HARSCnnEncoder(nn.Module):
    """
    A convolutional encoder used with the LFR technique, adapted from
    https://github.com/layer6ai-labs/lfr/blob/main/ssl_models/models/encoders.py
    to work with our HAR dataset.
    """

    def __init__(
        self,
        dim: int = 128,
        input_channel: int = 9,
        inner_conv_output_dim: int = 128 * 18,
    ):
        """
        Parameters
        ----------
        dim : int
            The dimension of the latent space, by default 128.
        input_channel : int
            The number of input channels, by default 9. In the LFR paper, the shape of
            the input data was (batch, 9, 128), which required an input_channel value
            of 9. However, for data in the shape (batch, 6, 60), a value of 6 is required.
        inner_conv_output_dim : int
            The output dimension of the inner convolutional layers, by default 128*18.
            In the LFR paper, the shape of the input data was (batch, 9, 128), which
            required an inner_conv_output_dim value of 128*18. However, for data in the
            shape (batch, 6, 60), a value of 128*10 is required.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                input_channel, 32, kernel_size=8, stride=1, bias=False, padding=(8 // 2)
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.35),
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        if dim == inner_conv_output_dim:
            self.mlp = nn.Identity()
        else:
            # use a linear layer to reach the latent shape
            self.mlp = nn.Linear(inner_conv_output_dim, dim)

    def forward(self, xb):
        # Flatten images into vectors
        out = self.conv(xb)
        out = out.view(out.size(0), -1)
        out = self.mlp(out)
        return out


class LFR_HAR_Projector(nn.Module):
    def __init__(self, encoding_size: int = 512, input_channel: int = 9):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(
                input_channel, 16, kernel_size=8, stride=1, bias=False, padding=(8 // 2)
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Conv1d(16, 32, kernel_size=8, stride=1, bias=False, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, encoding_size),
        )

    def forward(self, x):
        return self.model(x)


class LFR_HAR_Predictor(nn.Module):
    def __init__(self, encoding_size: int, middle_dim: int, num_layers: int):
        super().__init__()
        # If we have 1 layer, we just use a linear layer
        if num_layers == 1:
            self.model = nn.Linear(encoding_size, encoding_size)
        # If we have 2 layers, we use a 2-layer predictor
        elif num_layers == 2:
            self.model = nn.Sequential(
                nn.Linear(encoding_size, middle_dim, bias=False),
                nn.BatchNorm1d(middle_dim),
                nn.ReLU(inplace=True),
                nn.Linear(middle_dim, encoding_size),
            )
        # If we have 3 layers, we use a 3-layer predictor
        elif num_layers == 3:
            self.model = nn.Sequential(
                nn.Linear(encoding_size, middle_dim, bias=False),
                nn.BatchNorm1d(middle_dim),
                nn.ReLU(inplace=True),
                nn.Linear(middle_dim, middle_dim, bias=False),
                nn.BatchNorm1d(middle_dim),
                nn.ReLU(inplace=True),
                nn.Linear(middle_dim, encoding_size),
            )
        else:
            raise ValueError("Invalid number of layers")

    def forward(self, z):
        return self.model(z)
