from typing import Literal, Tuple
from minerva.models.nets.tnc import TSEncoder, DilatedConvEncoder
from minerva.models.nets.base import SimpleSupervisedModel
from minerva.models.nets.mlp import MLP
import torch


class TS2VecClassifier(SimpleSupervisedModel):
    def __init__(
        self,
        input_shape: Tuple[int, int],
        ts_input_dims: int,
        ts_output_dims: int,
        ts_hidden_dims: int = 64,
        ts_depth: int = 10,
        hidden_dims: int = 128,
        num_classes: int = 6,
    ):
        encoder = TSEncoder(
            input_dims=ts_input_dims,
            hidden_dims=ts_hidden_dims,
            output_dims=ts_output_dims,
            depth=ts_depth,
            permute=True,
            encoder_cls=DilatedConvEncoder,
        )
        self.fc_input_features = self._calculate_fc_input_features(encoder, input_shape)

        super().__init__(
            backbone=encoder,
            fc=MLP([self.fc_input_features, hidden_dims, num_classes]),
            loss_fn=torch.nn.CrossEntropyLoss(),
            flatten=True,
        )

    def _calculate_fc_input_features(
        self, backbone: torch.nn.Module, input_shape: Tuple[int, int]
    ) -> int:
        """Run a single forward pass with a random input to get the number of
        features after the convolutional layers.

        Parameters
        ----------
        backbone : torch.nn.Module
            The backbone of the network
        input_shape : Tuple[int, int, int]
            The input shape of the network.

        Returns
        -------
        int
            The number of features after the convolutional layers.
        """
        random_input = torch.randn(1, *input_shape)
        with torch.no_grad():
            out = backbone(random_input)
        return out.reshape(out.size(0), -1).size(1)
