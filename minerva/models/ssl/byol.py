import copy
import torch
import torchvision
import lightning as L
import numpy as np
import warnings

from torch import nn
from torch import Tensor
from collections import OrderedDict
from typing import Optional, Sequence

from minerva.losses.negative_cossine_similatiry import NegativeCosineSimilarity
from minerva.models.nets.mlp import MLP
from torch.optim import Optimizer
from minerva.models.nets.image.deeplabv3 import DeepLabV3Backbone


class BYOL(L.LightningModule):
    """Bootstrap Your Own Latent (BYOL) model for self-supervised learning.

    References
    ----------
    Grill, J., Strub, F., Altché, F., Tallec, C., Richemond, P. H., Buchatskaya, E., ... & Valko, M. (2020).
    "Bootstrap your own latent - a new approach to self-supervised learning." Advances in Neural Information Processing Systems, 33, 21271-21284.
    """

    def __init__(
        self,
        backbone: Optional[nn.Module] = None,
        projection_head: Optional[nn.Module] = None,
        prediction_head: Optional[nn.Module] = None,
        learning_rate: Optional[float] = 1e-3,
        schedule: Optional[int] = 90000,
        criterion: Optional[Optimizer] = None,
    ):
        """
        Initializes the BYOL model.

        Parameters
        ----------
        backbone : Optional[nn.Module]
            The backbone network for feature extraction. Defaults to DeepLabV3Backbone.
        projection_head : Optional[nn.Module]
            Optional custom projection head module. If None, a default MLP-based projection head is used.
        prediction_head : Optional[nn.Module]
            Optional custom prediction head module. If None, a default MLP-based prediction head is used.
        learning_rate : float
            The learning rate for the optimizer. Defaults to 1e-3.
        schedule : int
            The total number of steps for cosine decay scheduling. Defaults to 90000.
        criterion : Optional[Optimizer]
            Loss function to use. Defaults to NegativeCosineSimilarity.
        """
        super().__init__()
        self.backbone = backbone or DeepLabV3Backbone()
        self.learning_rate = learning_rate
        self.projection_head = projection_head or self._default_projection_head()
        self.prediction_head = prediction_head or self._default_prediction_head()
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        self.deactivate_requires_grad(self.backbone_momentum)
        self.deactivate_requires_grad(self.projection_head_momentum)
        self.criterion = criterion or NegativeCosineSimilarity()
        self.schedule_length = schedule

    def _default_projection_head(self) -> nn.Module:
        """Creates the default projection head used in BYOL."""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            MLP(
                layer_sizes=[2048, 4096, 256],
                activation_cls=nn.ReLU,
                intermediate_ops=[nn.BatchNorm1d(4096), None],
            ),
        )

    def _default_prediction_head(self) -> nn.Module:
        """Creates the default prediction head used in BYOL."""
        return MLP(
            layer_sizes=[256, 4096, 256],
            activation_cls=nn.ReLU,
            intermediate_ops=[nn.BatchNorm1d(4096), None],
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the BYOL model.

        Parameters
        ----------
        x : Tensor
            Input image tensor.

        Returns
        -------
        Tensor
            Output tensor after passing through the backbone, projection, and prediction heads.
        """
        y = self.backbone(x)
        if isinstance(y, OrderedDict):
            y = y["out"]
        z = self.projection_head(y)
        p = self.prediction_head(z)
        return p

    def forward_momentum(self, x: Tensor) -> Tensor:
        """
        Forward pass using momentum encoder.

        Parameters
        ----------
        x : Tensor
            Input image tensor.

        Returns
        -------
        Tensor
            Output tensor after passing through the momentum backbone and projection head.
        """
        y = self.backbone_momentum(x)
        if isinstance(y, OrderedDict):
            y = y["out"]
        z = self.projection_head_momentum(y)
        return z.detach()

    def training_step(self, batch: Sequence[Tensor], batch_idx: int) -> Tensor:
        """
        Performs a training step for BYOL.

        Parameters
        ----------
        batch : Sequence[Tensor]
            A batch of input pairs (x0, x1).
        batch_idx : int
            Batch index.

        Returns
        -------
        Tensor
            The computed loss for the current batch.
        """
        momentum = self.cosine_schedule(
            self.current_epoch, self.schedule_length, 0.996, 1
        )
        self.update_momentum(self.backbone, self.backbone_momentum, m=momentum)
        self.update_momentum(
            self.projection_head, self.projection_head_momentum, m=momentum
        )
        (x0, x1) = batch
        p0 = self.forward(x0)
        z0 = self.forward_momentum(x0)
        p1 = self.forward(x1)
        z1 = self.forward_momentum(x1)
        loss = 0.5 * (self.criterion(p0, z1) + self.criterion(p1, z0))
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def configure_optimizers(self):
        """
        Configures the optimizer for the BYOL model.

        Returns
        -------
        torch.optim.SGD
            Optimizer with configured learning rate.
        """
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate)

    @torch.no_grad()
    def update_momentum(self, model: nn.Module, model_ema: nn.Module, m: float):
        """
        Updates model weights using momentum.

        Parameters
        ----------
        model : nn.Module
            Original model.
        model_ema : nn.Module
            Momentum model.
        m : float
            Momentum factor.
        """
        for model_ema_param, model_param in zip(
            model_ema.parameters(), model.parameters()
        ):
            model_ema_param.data = model_ema_param.data * m + model_param.data * (
                1.0 - m
            )

    @torch.no_grad()
    def deactivate_requires_grad(self, model: nn.Module):
        """
        Freezes the weights of the model.

        Parameters
        ----------
        model : nn.Module
            Model to freeze.
        """
        for param in model.parameters():
            param.requires_grad = False

    def cosine_schedule(
        self,
        step: int,
        max_steps: int,
        start_value: float,
        end_value: float,
        period: Optional[int] = None,
    ) -> float:
        """
        Uses cosine decay to gradually modify `start_value` to reach `end_value`.

        Parameters
        ----------
        step : int
            Current step number.
        max_steps : int
            Total number of steps.
        start_value : float
            Starting value.
        end_value : float
            Target value.
        period : Optional[int]
            Steps over which cosine decay completes a full cycle. Defaults to max_steps.

        Returns
        -------
        float
            Cosine decay value.
        """
        if step < 0:
            raise ValueError(f"Current step number {step} can't be negative")
        if max_steps < 1:
            raise ValueError(f"Total step number {max_steps} must be >= 1")
        if period is not None and period <= 0:
            raise ValueError(f"Period {period} must be >= 1")
        decay = (
            end_value
            - (end_value - start_value)
            * (np.cos(np.pi * step / (max_steps - 1)) + 1)
            / 2
        )
        return decay
