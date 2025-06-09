from torch import nn
from typing import Optional, Tuple, Any
from torch import Tensor
import torch
import lightning as L


class SimSiam(L.LightningModule):
    """
    SimSiam implementation using PyTorch Lightning.

    This class implements the SimSiam self-supervised learning framework, which is designed
    to learn useful representations without using negative samples. It employs a backbone
    encoder, a projection head, and a prediction head to train the backbone.
    """

    def __init__(
        self,
        backbone: nn.Module,
        projection_head: Optional[nn.Module] = None,
        prediction_head: Optional[nn.Module] = None,
        loss_fn: Optional[nn.Module] = None,
        learning_rate: float = 0.0001,
        weight_decay: float = 1e-6,
    ):
        """
        Initialize the SimSiam module.

        Parameters
        ----------
        backbone : nn.Module
            The feature extractor network (e.g., a ResNet encoder).
        projection_head : nn.Module, optional
            The network that maps backbone outputs to the projection space.
            If None, a default 3-layer MLP designed to work with ResNet50 is used.
        prediction_head : nn.Module, optional
            The network that maps projection vectors to the prediction space.
            If None, a default 2-layer MLP is used.
        loss_fn : Callable, optional
            Loss function used for training. Default is cosine similarity loss.
        learning_rate : float, optional
            Learning rate for the optimizer. Default is 0.0001.
        weight_decay : float, optional
            Weight decay for the optimizer. Default is 1e-6.
        """
        super(SimSiam, self).__init__()
        self.backbone = backbone

        if projection_head:
            self.projection_head = projection_head
        else:
            input_dim = 2048
            hidden_dim = 2048
            output_dim = 2048
            self.projection_head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
                nn.BatchNorm1d(output_dim, affine=False),
            )

        if prediction_head:
            self.prediction_head = prediction_head
        else:
            input_dim = 2048
            hidden_dim = 512
            output_dim = 2048
            self.prediction_head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )

        if loss_fn:
            self.loss_fn = loss_fn
        else:
            self.loss_fn = torch.nn.CosineSimilarity(dim=1, eps=1e-8)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, x):
        """
        Forward pass through the backbone, projection, and prediction heads.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The detached projection vector `z` and prediction vector `p`.
        """
        f = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(f)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p

    def _single_step(self, batch: Tuple[Tuple[Tensor, Tensor], Any]) -> Tensor:
        """
        Compute the loss for a single batch.

        Parameters
        ----------
        batch : Tuple[Tuple[Tensor, Tensor], Any]
            A tuple containing a pair of augmented views (x0, x1) and labels (unused).

        Returns
        -------
        torch.Tensor
            The computed loss for the batch.
        """
        (x0, x1), _ = batch
        z0, p0 = self.forward(x0)
        z1, p1 = self.forward(x1)
        loss = -0.5 * (self.loss_fn(z0, p1).mean() + self.loss_fn(z1, p0).mean())
        return loss

    def training_step(
        self, batch: Tuple[Tuple[Tensor, Tensor], Any], batch_idx: int
    ) -> Tensor:
        """
        Defines one training step.

        Parameters
        ----------
        batch : Tuple[Tuple[Tensor, Tensor], Any]
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
        batch : Tuple[Tuple[Tensor, Tensor], Any]
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
            The optimizer used for training.
        """
        return torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
