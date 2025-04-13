from torch import nn
import torch
from typing import Union


class SiameseNetworkWrapper(nn.Module):
    """
    A simple wrapper for a Siamese Network. The code was inspired by the tutorial in the Pytorch
    website https://github.com/pytorch/examples/blob/main/siamese_network/main.py. It passes the
    inputs (namely x1 and x2) through the same backbone, and concatenates the representations
    obtained.
    """

    def __init__(self, backbone: nn.Module) -> None:
        """
        Initializes the wrapper.

        Parameters
        ----------
        backbone : nn.Module
            The backbone of the Siamese Network.
        """
        super(SiameseNetworkWrapper, self).__init__()
        self.backbone = backbone

    def forward_once(self, x):
        """
        Passes the input through the backbone and flattens the output.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output data from the forward pass through the backbone.
        """

        output = self.backbone(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, x: Union[list, tuple]) -> torch.Tensor:
        """
        Passes the inputs through the backbone and concatenates the representations.
        x must be a list or a tuple containing two inputs, namely x1 and x2.

        Parameters
        ----------
        x : Union[list, tuple]
            A list or a tuple containing the two inputs.

        Returns
        -------
        torch.Tensor
            The concatenated representations.
        """
        # Asserting the input is a list or a tuple
        if not isinstance(x, (list, tuple)):
            raise TypeError("The input must be a list or a tuple")
        # Asserting the input has two elements
        if len(x) != 2:
            raise ValueError("The input must have two elements")
        # Extracting the two inputs
        x1 = x[0]
        x2 = x[1]
        # Passing the inputs through the backbone
        x1 = self.forward_once(x1)
        x2 = self.forward_once(x2)
        # Concatenating the representations
        output = torch.cat((x1, x2), 1)
        return output
