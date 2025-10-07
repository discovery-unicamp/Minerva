import copy
import torch
import numpy as np
from torch import nn, Tensor
from collections import OrderedDict
from typing import Optional, Sequence, Dict, Any

from minerva.losses.negative_cossine_similatiry import NegativeCosineSimilarity
from minerva.models.nets.mlp import MLP
from minerva.models.nets.image.deeplabv3 import DeepLabV3Backbone
from minerva.models.nets.base import SimpleSupervisedModel


class BYOL(SimpleSupervisedModel):
    """
    Bootstrap Your Own Latent (BYOL) model for self-supervised representation learning.

    This class implements the BYOL framework [1], built on top of :class:`SimpleSupervisedModel`
    to reuse its optimizer, logging, and training utilities. Unlike typical supervised models,
    BYOL does not require labeled data; instead, it learns representations by predicting one
    augmented view of an image from another, using both an online and a momentum encoder.

    The model consists of:
        - An **online encoder**: backbone + projection head + prediction head.
        - A **momentum encoder**: backbone + projection head (no prediction head),
          updated using an exponential moving average of the online encoder parameters.

    Key features:
        - Self-supervised loss via :class:`~minerva.losses.negative_cossine_similatiry.NegativeCosineSimilarity`
        - Momentum update schedule using cosine decay.
        - Default optimizer: Adam with ``weight_decay=1e-6``.
        - Built-in hooks for momentum update and loss computation.

    Parameters
    ----------
    backbone : nn.Module, optional
        Feature extractor network. Defaults to :class:`~minerva.models.nets.image.deeplabv3.DeepLabV3Backbone`.
    projection_head : nn.Module, optional
        Projection head mapping encoder features to latent space.
        If None, a default 3-layer MLP is used.
    prediction_head : nn.Module, optional
        Prediction head mapping projected features to target space.
        If None, a default 2-layer MLP is used.
    learning_rate : float, default=1e-3
        Learning rate for optimizer.
    schedule : int, default=90000
        Number of training steps over which to apply cosine momentum schedule.
    criterion : nn.Module, optional
        Loss function. Defaults to :class:`~minerva.losses.negative_cossine_similatiry.NegativeCosineSimilarity`.
    optimizer : type, optional
        Optimizer class. Defaults to :class:`torch.optim.Adam` if not provided.
    optimizer_kwargs : dict, optional
        Extra keyword arguments for the optimizer. By default, uses ``{"weight_decay": 1e-6}``.

    Notes
    -----
    - Metrics are disabled by default since BYOL is self-supervised.
    - The ``fc`` layer from :class:`SimpleSupervisedModel` is replaced with ``nn.Identity()``
      because BYOL uses its own projection/prediction heads.
    - The forward pass returns predictions from the online encoder; the momentum encoder is
      used internally for target computation only.

    References
    ----------
    [1] Grill, J.B., Strub, F., Altché, F., Tallec, C., Richemond, P.H., Buchatskaya, E.,
        Doersch, C., Pires, B.A., Guo, Z.D., Azar, M.G., Piot, B., Kavukcuoglu, K.,
        Munos, R., & Valko, M. (2020).
        Bootstrap Your Own Latent - A New Approach to Self-Supervised Learning.
        Advances in Neural Information Processing Systems, 33, 21271–21284.
    """

    def __init__(
        self,
        backbone: Optional[nn.Module] = None,
        projection_head: Optional[nn.Module] = None,
        prediction_head: Optional[nn.Module] = None,
        learning_rate: float = 1e-3,
        schedule: int = 90000,
        criterion: Optional[nn.Module] = None,
        optimizer: type = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        backbone_model = backbone or DeepLabV3Backbone()
        projection_head_model = projection_head or self._default_projection_head()
        prediction_head_model = prediction_head or self._default_prediction_head()
        loss_criterion = criterion or NegativeCosineSimilarity()

        optimizer = optimizer or torch.optim.Adam
        default_optimizer_kwargs = {"lr": learning_rate, "weight_decay": 1e-6}
        if optimizer_kwargs:
            default_optimizer_kwargs = optimizer_kwargs

        super().__init__(
            backbone=backbone_model,
            fc=nn.Identity(),
            loss_fn=loss_criterion,
            adapter=None,
            learning_rate=learning_rate,
            flatten=False,
            train_metrics=None,
            val_metrics=None,
            test_metrics=None,
            freeze_backbone=False,
            optimizer=optimizer,
            optimizer_kwargs=default_optimizer_kwargs,
        )

        self.backbone = backbone_model
        self.projection_head = projection_head_model
        self.prediction_head = prediction_head_model

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        self.deactivate_requires_grad(self.backbone_momentum)
        self.deactivate_requires_grad(self.projection_head_momentum)

        self.criterion = loss_criterion
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

    def _loss_func(self, outputs, targets=None) -> torch.Tensor:

        (x0, x1) = outputs
        p0 = self.forward(x0)
        z0 = self.forward_momentum(x0)
        p1 = self.forward(x1)
        z1 = self.forward_momentum(x1)
        return 0.5 * (self.criterion(p0, z1) + self.criterion(p1, z0))

    def training_step(self, batch: Sequence[Tensor], batch_idx: int) -> torch.Tensor:
        """Overrides SimpleSupervisedModel's step for BYOL."""
        momentum = self.cosine_schedule(
            self.current_epoch, self.schedule_length, 0.996, 1
        )
        self.update_momentum(self.backbone, self.backbone_momentum, m=momentum)
        self.update_momentum(
            self.projection_head, self.projection_head_momentum, m=momentum
        )

        loss = self._loss_func(batch)
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
