import torch
import torch.nn as nn
from typing import Tuple, Optional
from minerva.models.nets.tnc import TSEncoder
from minerva.models.adapters import MaxPoolingTransposingSqueezingAdapter
from typing import Callable

class TFC_Backbone(nn.Module):
    """
    A convolutional version of backbone of the Temporal-Frequency Convolutional (TFC) model.
    The backbone is composed of two convolutional neural networks that extract features from the input data in the time domain and frequency domain.
    The features are then projected to a latent space.
    This class implements the forward method that receives the input data and returns the features extracted in the time domain and frequency domain.
    """
    def _calculate_fc_input_features(self, encoder: torch.nn.Module, input_shape: Tuple[int, int],
                                     adapter: Optional[Callable[[torch.Tensor], torch.Tensor]] = None) -> int:
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
            out = encoder(random_input)
            if self.adapter is not None:
                out = self.adapter(out)
        try:
            out = out.view(out.size(0), -1).size(1)
        except :
            out = out.reshape(out.size(0), -1).size(1)
        return out
    
    def __init__(self, input_channels: int, TS_length: int, single_encoding_size: int = 128,
                time_encoder: Optional[nn.Module] = None, frequency_encoder: Optional[nn.Module] = None,
                time_projector: Optional[nn.Module] = None, frequency_projector: Optional[nn.Module] = None,
                adapter: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
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
        """
        super(TFC_Backbone, self).__init__()
        self.adapter = adapter

        self.time_encoder = time_encoder
        if time_encoder is None:
            self.time_encoder = TFC_Conv_Block(input_channels)

        self.frequency_encoder = frequency_encoder
        if frequency_encoder is None:
            self.frequency_encoder = TFC_Conv_Block(input_channels)

        self.time_projector = time_projector
        if time_projector is None:
            self.time_projector = TFC_Standard_Projector(self._calculate_fc_input_features(self.time_encoder, (input_channels, TS_length)), single_encoding_size)

        self.frequency_projector = frequency_projector
        if frequency_projector is None:
            self.frequency_projector = TFC_Standard_Projector(self._calculate_fc_input_features(self.frequency_encoder, (input_channels, TS_length)), single_encoding_size)
        
    def forward(self, x_in_t: torch.Tensor, x_in_f: torch.Tensor) -> torch.Tensor:
        """
        The forward method of the backbone. It receives the input data in the time domain and frequency domain and returns the features extracted in the time domain and frequency domain.

        Parameters
        ----------
        - x_in_t: torch.Tensor
            The input data in the time domain
        - x_in_f: torch.Tensor
            The input data in the frequency domain

        Returns
        -------
        - tuple
            A tuple with the features extracted in the time domain and frequency domain, h_time, z_time, h_freq, z_freq respectively
        """

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

        return h_time, z_time, h_freq, z_freq
    

class TFC_PredicionHead(nn.Module):
    """
    A simple prediction head for the Temporal-Frequency Convolutional (TFC) model.
    The prediction head is composed of a linear layer that receives the features extracted by the backbone and returns the prediction of the model.
    This class implements the forward method that receives the features extracted by the backbone and returns the prediction of the model.
    """
    def __init__(self, num_classes: int, connections:int =2, single_encoding_size:int =128):
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
        """
        super(TFC_PredicionHead, self).__init__()
        if connections != 2:
            print(f"Only one pipeline is on: {connections} connection.")
        self.logits = nn.Linear(connections*single_encoding_size, 64)
        self.logits_simple = nn.Linear(64, num_classes)

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
        return pred

class TFC_Conv_Block(nn.Module):
    """
    A standart convolutional block for the Temporal-Frequency Convolutional (TFC) model.

    This class implements the forward method that receives the input data and returns the features extracted by the block.
    """
    def __init__(self, input_channels: int):
        """
        Constructor of the TFC_Conv_Block class.

        Parameters
        ----------
        - input_channels: int
            The number of channels in the input data
        """
        super(TFC_Conv_Block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.35),
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Conv1d(64, 60, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(60),
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
    def __init__(self, input_channels: int, single_encoding_size: int):
        """
        Constructor of the TFC_Standard_Projector class.

        Parameters
        ----------
        - input_channels: int
            The number of channels in the input data
        - single_encoding_size: int
            The size of the encoding in the latent space of frequency or time domain individually
        """
        super(TFC_Standard_Projector, self).__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, single_encoding_size)
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
        return self.projector(x)