from torch import nn
from minerva.models.ssl.lfr import RepeatedModuleList


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
    """
    A projector module for LFR in HAR tasks that projects the input data into a random
    latent space.
    """

    def __init__(
        self, encoding_size: int = 512, input_channel: int = 9, middle_dim: int = 1088
    ):
        """
        Initializes a projector module.

        Parameters
        ----------
        encoding_size: int
            The output dimensionality of the projector module.
        input_channel: int
            The number of channels in the input data.
        middle_dim: int
            The expected dimensionality after the convolution module, by default 1088. The
            original paper, where the input has 9 channels and 128 timestamps, requires 1088.
            For data with 6 channels and 60 timestamps, 544 should be used.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                input_channel, 16, kernel_size=8, stride=1, bias=False, padding=(8 // 2)
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Conv1d(16, 32, kernel_size=8, stride=1, bias=False, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.mlp = nn.Sequential(
            nn.Linear(middle_dim, 256), nn.Linear(256, encoding_size)
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.mlp(out)
        return out


class LFR_HAR_Predictor(nn.Module):
    """
    A predictor module for LFR in HAR tasks that maps latent embeddings to a randomly
    projected data representation.
    """

    def __init__(self, encoding_size: int, middle_dim: int, num_layers: int):
        """
        Initializes a predictor module.

        Parameters
        ----------
        encoding_size: int
            The input and output dimensionality of the predictor module.
        middle_dim: int
            Dimensionality of the hidden layers in the predictor.
        num_layers: int
            Number of layers in the predictor. If set to 1, the predictor becomes a single
            linear layer and 'middle_dim' is ignored.
        """
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


class LFR_HAR_Projector_List(RepeatedModuleList):
    """
    A repeated list of projector modules for LFR in HAR tasks. Each one projects the
    input data into a random latent space.
    """

    def __init__(
        self, size: int, encoding_size: int, input_channel: int, middle_dim: int
    ):
        """
        Initializes a list of projector modules.

        Parameters
        ----------
        size: int
            Number of projector modules to instantiate in the list.
        encoding_size: int
            The output dimensionality of each projector module.
        input_channel: int
            The number of channels in the input data.
        """
        super().__init__(
            size=size,
            cls=LFR_HAR_Projector,
            encoding_size=encoding_size,
            input_channel=input_channel,
            middle_dim=middle_dim,
        )


class LFR_HAR_Predictor_List(RepeatedModuleList):
    """
    A repeated list of predictor modules for LFR in HAR tasks. Each predictor maps latent
    embeddings to a randomly projected data representation.
    """

    def __init__(self, size: int, encoding_size: int, middle_dim: int, num_layers: int):
        """
        Initializes a list of predictor modules.

        Parameters
        ----------
        size: int
            Number of predictor modules to instantiate in the list.
        encoding_size: int
            The input and output dimensionality of each predictor module.
        middle_dim: int
            Dimensionality of the hidden layers in each predictor.
        num_layers: int
            Number of layers in each predictor. If set to 1, the predictors become single
            linear layers and 'middle_dim' is ignored.
        """
        super().__init__(
            size=size,
            cls=LFR_HAR_Predictor,
            encoding_size=encoding_size,
            middle_dim=middle_dim,
            num_layers=num_layers,
        )
