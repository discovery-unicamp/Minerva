from typing import Any, Optional, Tuple

import lightning as L
import torch
import torch.nn as nn
from torch import Tensor

from minerva.losses.xtent_loss import NTXentLoss
from minerva.optimizers.lars import LARS


class SimCLR(L.LightningModule):
    def __init__(
        self,
        backbone: nn.Module,
        projection_head: nn.Module,
        flatten: bool = True,
        temperature: float = 0.5,
        lr: float = 1e-3,
    ):
        """
        Initializes the SimCLR model.

        Parameters
        ----------
        backbone : nn.Module
            Backbone model for feature extraction.
        projection_head : nn.Module
            Projection head model.
        flatten : bool, optional, default=True
            Whether to flatten the output of the backbone model, by default True
        temperature : float, optional, default=0.5
            Temperature for the NT-Xent loss, by default 0.5
        lr : float, optional, default=1e-3
            Learning rate for the optimizer, by default 1e-3
        """
        super(SimCLR, self).__init__()
        self.backbone = backbone
        self.projector = projection_head
        self.lr = lr
        self.flatten = flatten
        self.loss = NTXentLoss(temperature)

    def forward(self, x: Tuple[Tensor, Tensor]):
        """
        Forward pass through the SimCLR model.

        Parameters
        ----------
        x : Tuple[Tensor, Tensor]
            Input tensor of features with shape (batch_size, input_dim).

        Returns
        -------
        Tensor
            Output tensor of projected features with shape (batch_size, output_dim).
        """
        x0, x1 = x
        x0 = self.backbone(x0)
        x1 = self.backbone(x1)
        if self.flatten:
            x0 = torch.flatten(x0, 1)
            x1 = torch.flatten(x1, 1)
        x0 = self.projector(x0)
        x1 = self.projector(x1)
        return x0, x1

    def _single_step(self, batch: Tuple[Tuple[Tensor, Tensor], Any]) -> Tensor:
        """
        Performs a single forward and loss computation step.

        Parameters
        ----------
        batch : Tuple[Tuple[Tensor, Tensor], Any]
            Input batch containing images and optional labels.

        Returns
        -------
        Tensor
            Computed loss for the batch.
        """
        images, _ = batch  # Labels are not used for contrastive loss
        y0, y1 = self.forward(images)
        loss = self.loss(y0, y1)
        return loss

    def training_step(
        self, batch: Tuple[Tuple[Tensor, Tensor], Any], batch_idx: int
    ) -> Tensor:
        """
        Training step.

        Parameters
        ----------
        batch : Tuple[Tuple[Tensor, Tensor], Any]
            Input batch containing images and optional labels.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        Tensor
            Computed loss for the batch.
        """
        loss = self._single_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(
        self, batch: Tuple[Tuple[Tensor, Tensor], Any], batch_idx: int
    ) -> Tensor:
        """
        Validation step.

        Parameters
        ----------
        batch : Tuple[Tuple[Tensor, Tensor], Any]
            Input batch containing images and optional labels.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        Tensor
            Computed loss for the batch.
        """
        loss = self._single_step(batch)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(
        self,
        batch: Tuple[Tuple[Tensor, Tensor], Any],
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ):
        """
        Predict step.

        Parameters
        ----------
        batch : Tuple[Tuple[Tensor, Tensor], Any]
            Input batch containing images and optional labels.
        batch_idx : int
            Index of the current batch.
        dataloader_idx : Optional[int], optional
            Index of the dataloader, by default None

        Returns
        -------
        Tensor
            Computed loss for the batch.
        """
        images, _ = batch
        return self.forward(images)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configures the optimizer for training.

        Returns
        -------
        torch.optim.Optimizer
            Optimizer instance.
        """
        return LARS(self.parameters(), lr=self.lr)
