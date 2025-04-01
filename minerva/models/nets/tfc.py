import torch
import torch.nn as nn
from typing import Tuple, Optional
from typing import Callable
from minerva.transforms.transform import _Transform
from minerva.transforms.tfc import TFC_Transforms


class TFC_Backbone(nn.Module):
    """
    A convolutional version of backbone of the Temporal-Frequency Convolutional (TFC) model.
    The backbone is composed of two convolutional neural networks that extract features from the input data in the time domain and frequency domain.
    The features are then projected to a latent space.
    This class implements the forward method that receives the input data and returns the features extracted in the time domain and frequency domain.
    """

    def _calculate_fc_input_features(
        self,
        encoder: torch.nn.Module,
        input_shape: Tuple[int, int],
        adapter: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> int:
        """
        Calculate the input features of the fully connected layer after the encoders (conv blocks).

        Parameters
        ----------
        - encoder: torch.nn.Module
            The encoder to calculate the input features
        - input_shape: Tuple[int, int]
            The input shape of the data
        - adapter : Callable[[torch.Tensor], torch.Tensor], optional
            An adapter to be used from the backbone to the head, by default None.

        Returns
        -------
        - int
            The number of features to be passed to the fully connected layer
        """
        random_input = torch.randn(1, *input_shape)
        with torch.no_grad():
            # print("\n0- random input shape:",random_input.shape)
            out = encoder(random_input)
            # print("\n1- out shape:",out.shape)
            if self.adapter is not None:
                out = self.adapter(out)
        # Handle cases where the output has only 1 dimension apart from the batch size
        if out.dim() == 1:
            calc_fc_input_features = out.size(0)  # Take the second dimension
        else:
            calc_fc_input_features = out.reshape(out.size(0), -1).size(1)

        # print(f"calc_fc_input_features: {calc_fc_input_features}, input_shape: {input_shape}")
        return calc_fc_input_features

    def __init__(
        self,
        input_channels: int,
        TS_length: int,
        single_encoding_size: int = 128,
        transform: _Transform = None,
        time_encoder: Optional[nn.Module] = None,
        frequency_encoder: Optional[nn.Module] = None,
        time_projector: Optional[nn.Module] = None,
        frequency_projector: Optional[nn.Module] = None,
        adapter: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        batch_1_correction: bool = False,
    ):
        """
        Constructor of the TFC_Backbone class.

        Parameters
        ----------
        - input_channels: int
            The number of channels in the input data
        - TS_length: int
            The number of time steps in the input data
        - single_encoding_size: int
            The size of the encoding in the latent space of frequency or time domain individually
        - transform: _Transform
            The transformation to be applied to the input data. If None, a default transformation is applied that includes data augmentation and frequency domain transformation
        - time_encoder: Optional[nn.Module]
            The encoder for the time domain. If None, a default encoder is used
        - frequency_encoder: Optional[nn.Module]
            The encoder for the frequency domain. If None, a default encoder is used
        - time_projector: Optional[nn.Module]
            The projector for the time domain. If None, a default projector is used. If passing, make sure to correct calculate the input features by backbone
        - frequency_projector: Optional[nn.Module]
            The projector for the frequency domain. If None, a default projector is used. If passing, make sure to correct calculate the input features by backbone
        - adapter : Callable[[torch.Tensor], torch.Tensor], optional
            An adapter to be used from the backbone to the head, by default None.
        - batch_1_correction: bool
            If True, the batch normalization is ignored when the batch size is 1,
            If False, a runtime error is raised when the batch size is 1
            Standard is False

        """
        super(TFC_Backbone, self).__init__()
        self.adapter = adapter
        self.transform = transform

        if transform is None:
            self.transform = TFC_Transforms()

        self.time_encoder = time_encoder
        if time_encoder is None:
            self.time_encoder = TFC_Conv_Block(
                input_channels, batch_1_correction=batch_1_correction
            )

        self.frequency_encoder = frequency_encoder
        if frequency_encoder is None:
            self.frequency_encoder = TFC_Conv_Block(
                input_channels, batch_1_correction=batch_1_correction
            )

        self.time_projector = time_projector
        if time_projector is None:
            self.time_projector = TFC_Standard_Projector(
                self._calculate_fc_input_features(
                    self.time_encoder, (input_channels, TS_length)
                ),
                single_encoding_size,
                batch_1_correction=batch_1_correction,
            )

        self.frequency_projector = frequency_projector
        if frequency_projector is None:
            self.frequency_projector = TFC_Standard_Projector(
                self._calculate_fc_input_features(
                    self.frequency_encoder, (input_channels, TS_length)
                ),
                single_encoding_size,
                batch_1_correction=batch_1_correction,
            )
        self.h_time, self.z_time, self.h_freq, self.z_freq = (
            None,
            None,
            None,
            None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward method of the backbone. It receives the input data in the time domain and frequency domain and returns the features extracted in the time domain and frequency domain.

        Parameters
        ----------
        - x: torch.Tensor
            The input data

        Returns
        -------
        - tuple
            A tuple with the features extracted in the time domain and frequency domain, h_time, z_time, h_freq, z_freq respectively
        """

        x_in_t, _, x_in_f, _ = self.transform(x)

        x = self.time_encoder(x_in_t)
        if self.adapter is not None:
            x = self.adapter(x)

        h_time = x.reshape(x.shape[0], -1)
        z_time = self.time_projector(h_time)

        f = self.frequency_encoder(x_in_f)
        if self.adapter is not None:
            f = self.adapter(f)

        h_freq = f.reshape(f.shape[0], -1)
        z_freq = self.frequency_projector(h_freq)

        self.h_time, self.z_time, self.h_freq, self.z_freq = (
            h_time,
            z_time,
            h_freq,
            z_freq,
        )

        return torch.cat((z_time, z_freq), dim=1)

    def get_representations(self):
        """
        This function returns the representations of the time and frequency domain extracted by the backbone.
        The h and z representations, after ther encoder and after the projector, respectively.
        This function must be called after the forward method.


        Returns
        -------
        - Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

        """
        return self.h_time, self.z_time, self.h_freq, self.z_freq


class TFC_PredicionHead(nn.Module):
    """
    A simple prediction head for the Temporal-Frequency Convolutional (TFC) model.
    The prediction head is composed of a linear layer that receives the features extracted by the backbone and returns the prediction of the model.
    This class implements the forward method that receives the features extracted by the backbone and returns the prediction of the model.
    """

    def __init__(
        self,
        num_classes: int,
        connections: int = 2,
        single_encoding_size: int = 128,
        argmax_output: bool = False,
    ):
        """
        Constructor of the TFC_PredicionHead class.

        Parameters
        ----------
        - num_classes: int
            The number of classes in the classification task
        - connections: int
            The number of pipelines in the backbone. If 1, only the time or frequency domain is used. If 2, both domains are used. Other values are treated as 1.
        - single_encoding_size: int
            The size of the encoding in the latent space of frequency or time domain individually
        - argmax: bool
            If True, the argmax function is applied to the prediction. If False, the prediction returns the logits
        """
        super(TFC_PredicionHead, self).__init__()
        if connections != 2:
            print(f"Only one pipeline is on: {connections} connection.")
        self.logits = nn.Linear(connections * single_encoding_size, 64)
        self.logits_simple = nn.Linear(64, num_classes)
        self.argmax_output = argmax_output

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """
        The forward method of the prediction head. It receives the features extracted by the backbone and returns the prediction of the model.

        Parameters
        ----------
        - emb: torch.Tensor
            The features extracted by the backbone

        Returns
        -------
        - torch.Tensor
            The prediction of the model

        """
        emb_flat = emb.reshape(emb.shape[0], -1)
        emb = torch.sigmoid(self.logits(emb_flat))
        pred = self.logits_simple(emb)
        if self.argmax_output:
            pred = pred.argmax(dim=1)
        return pred


class TFC_Conv_Block(nn.Module):
    """
    A standart convolutional block for the Temporal-Frequency Convolutional (TFC) model.

    This class implements the forward method that receives the input data and returns the features extracted by the block.
    """

    def __init__(self, input_channels: int, batch_1_correction: bool = False):
        """
        Constructor of the TFC_Conv_Block class.

        Parameters
        ----------
        - input_channels: int
            The number of channels in the input data
        - batch_1_correction: bool
            If True, the batch normalization is ignored when the batch size is 1,
            If False, a runtime error is raised when the batch size is 1
            Standard is False
        """
        super(TFC_Conv_Block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                input_channels,
                32,
                kernel_size=8,
                stride=1,
                bias=False,
                padding=4,
            ),
            IgnoreWhenBatch1(nn.BatchNorm1d(32), active=batch_1_correction),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.35),
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            IgnoreWhenBatch1(nn.BatchNorm1d(64), active=batch_1_correction),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Conv1d(64, 60, kernel_size=8, stride=1, bias=False, padding=4),
            IgnoreWhenBatch1(nn.BatchNorm1d(60), active=batch_1_correction),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor):
        """
        The forward method of the convolutional block. It receives the input data and returns the features extracted by the block.

        Parameters
        ----------
        - x: torch.Tensor
            The input data

        Returns
        -------
        - torch.Tensor
            The features extracted by the block
        """
        return self.block(x)


class TFC_Standard_Projector(nn.Module):
    """
    A standart projector for the Temporal-Frequency Convolutional (TFC) model.

    This class implements the forward method that receives the input data and returns the features extracted by the projector.
    """

    def __init__(
        self,
        input_channels: int,
        single_encoding_size: int,
        batch_1_correction: bool = False,
    ):
        """
        Constructor of the TFC_Standard_Projector class.

        Parameters
        ----------
        - input_channels: int
            The number of channels in the input data
        - single_encoding_size: int
            The size of the encoding in the latent space of frequency or time domain individually
        - batch_1_correction: bool
            If True, the batch normalization is ignored when the batch size is 1,
            If False, a runtime error is raised when the batch size is 1
            Standard is False

        """
        super(TFC_Standard_Projector, self).__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_channels, 256),
            IgnoreWhenBatch1(nn.BatchNorm1d(256), active=batch_1_correction),
            nn.ReLU(),
            nn.Linear(256, single_encoding_size),
        )

    def forward(self, x: torch.Tensor):
        """
        The forward method of the projector. It receives the input data and returns the features extracted by the projector.

        Parameters
        ----------
        - x: torch.Tensor
            The input data

        Returns
        -------
        - torch.Tensor
            The features extracted by the projector
        """
        try:
            return self.projector(x)
        except ValueError as e:
            # mostra o tipo do erro
            if "Expected more than 1 value per channel" in e.args[0]:
                raise ValueError(
                    "The batch size is 1, which is not supported by this convolutional backbone. If you really want to use a batch size of 1, set the batch_1_correction parameter on constructor of projector or the TF-C backbone to True."
                )
            else:
                raise e


class IgnoreWhenBatch1(nn.Module):
    """
    This class is used to ignore some processes when the batch size is 1. It is necessary in Batch Normalization.

    """

    def __init__(self, module: nn.Module, active: bool = False):
        """
        Parameters
        ----------
        - module: nn.Module
            The module to be used in the forward method that will be ignored when the batch size is 1.
        - active: bool
            If True, the module is only used in the forward method if batch size is different from 1. If False, the module is always used.

        """
        super().__init__()  # necessary to instantiate backward hooks
        self.module = module
        self.active = active
        if active:
            print(f"{nn.Module} will be ignored when batch size is 1.")

    def forward(self, x):
        """
        The forward method of the IgnoreWhenBatch1 class. It receives the input data and returns the
        output of the module if the batch size is greater than 1. Otherwise, it returns the input data.

        Parameters
        ----------
        - x: torch.Tensor
            The input data

        Returns
        -------
        - torch.Tensor
            The output of the module if the batch size is greater than 1. Otherwise, the input data.

        """
        if x.shape[0] == 1 and self.active:
            return x
        return self.module(x)
