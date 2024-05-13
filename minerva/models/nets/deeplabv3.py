import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .base import SimpleSupervisedModel


class Resnet50Backbone(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.resnet50 = models.resnet50()
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-2])

    def forward(self, x):
        return self.resnet50(x)


class DeepLabV3_Head(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError(
            "DeepLabV3's head has not yet been implemented"
        )

    def forward(self, x):
        raise NotImplementedError(
            "DeepLabV3's head has not yet been implemented"
        )


class DeepLabV3(SimpleSupervisedModel):
    """A DeeplabV3 with a ResNet50 backbone

    References
    ----------
    Liang-Chieh Chen, George Papandreou, Florian Schroff, Hartwig Adam. 
    "Rethinking Atrous Convolution for Semantic Image Segmentation", 2017
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        loss_fn: torch.nn.Module = None,
        **kwargs,
    ):
        """Wrapper implementation of the DeepLabv3 model.

        Parameters
        ----------
        learning_rate : float, optional
            The learning rate to Adam optimizer, by default 1e-3
        loss_fn : torch.nn.Module, optional
            The function used to compute the loss. If `None`, it will be used
            the MSELoss, by default None.
        kwargs : Dict
            Additional arguments to be passed to the `SimpleSupervisedModel`
            class.
        """
        super().__init__(
            backbone=Resnet50Backbone(),
            fc=DeepLabV3_Head(),
            loss_fn=loss_fn or torch.nn.MSELoss(),
            learning_rate=learning_rate,
            **kwargs,
        )
