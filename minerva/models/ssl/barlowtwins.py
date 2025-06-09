from torch import nn
from typing import Optional, Tuple, Any
from torch import Tensor
import torch
import lightning as L
from minerva.losses.batchwise_barlowtwins_loss import BarlowTwinsLoss


class BarlowTwins(L.LightningModule):
    """
    PyTorch Lightning module implementing the Barlow Twins self-supervised learning framework.

    It accepts a backbone and projection head for feature encoding and embedding,
    uses a contrastive loss (defaulting to BarlowTwinsLoss if none is provided),
    supports standard training and validation loops in PyTorch Lightning,
    and optimizes using the Adam optimizer.
    """

    def __init__(
        self,
        backbone: nn.Module,
        projection_head: nn.Module,
        loss_fn: Optional[nn.Module] = None,
        learning_rate: float = 0.0001,
        weight_decay: float = 1e-6,
    ):
        """
        Initialize the BarlowTwins module.

        Parameters
        ----------
        backbone : nn.Module
            Neural network used to extract features from input images.
        projection_head : nn.Module
            Network that maps backbone outputs to a latent space.
        loss_fn : nn.Module, optional
            Custom loss function. Defaults to BarlowTwinsLoss.
        learning_rate : float, optional
            Learning rate for the optimizer. Default is 0.0001.
        weight_decay : float, optional
            Weight decay (L2 regularization). Default is 1e-6.
        """
        super().__init__()
        self.backbone = backbone
        self.projection_head = projection_head
        if loss_fn:
            self.loss_fn = loss_fn
        else:
            self.loss_fn = BarlowTwinsLoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, x: Tensor):
        """
        Forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (e.g., batch of images).

        Returns
        -------
        torch.Tensor
            Projected features in the embedding space.
        """
        x = self.backbone(x)
        z = self.projection_head(x)
        return z

    def _single_step(self, batch: Tuple[Tuple[Tensor, Tensor], Any]) -> Tensor:
        """
        Compute the loss for a single batch.

        Parameters
        ----------
        batch : tuple of ((torch.Tensor, torch.Tensor), Any)
            A tuple containing a pair of augmented views (x0, x1) and labels (unused).

        Returns
        -------
        torch.Tensor
            The computed loss for the batch.
        """
        (x0, x1), _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.loss_fn(z0, z1)
        return loss

    def training_step(
        self, batch: Tuple[Tuple[Tensor, Tensor], Any], batch_idx: int
    ) -> Tensor:
        """
        Defines one training step.

        Parameters
        ----------
        batch : tuple of ((torch.Tensor, torch.Tensor), Any)
            Batch containing two augmented views and labels (unused).
        batch_idx : int
            Index of the batch.

        Returns
        -------
        torch.Tensor
            Training loss for the batch.
        """
        loss = self._single_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(
        self, batch: Tuple[Tuple[Tensor, Tensor], Any], batch_idx: int
    ) -> Tensor:
        """
        Defines one validation step.

        Parameters
        ----------
        batch : tuple of ((torch.Tensor, torch.Tensor), Any)
            Batch containing two augmented views and labels (unused).
        batch_idx : int
            Index of the batch.

        Returns
        -------
        torch.Tensor
            Validation loss for the batch.
        """
        loss = self._single_step(batch)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """
        Configures the Adam optimizer with provided learning rate and weight decay.

        Returns
        -------
        torch.optim.Optimizer
            Configured Adam optimizer.
        """
        return torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
