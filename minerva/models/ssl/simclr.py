from typing import Any, Callable, Optional, Sequence, Tuple
import torch
import torch.nn as nn
from torch.nn.functional import normalize
import lightning as L

class ProjectionHead(nn.Module):
    """
    Projection head for SimCLR that maps features to a latent space for contrastive learning.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.
    hidden_dim : int
        Dimensionality of the hidden layer.
    output_dim : int
        Dimensionality of the output projections.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(ProjectionHead, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """
        Performs a forward pass through the projection head.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of features with shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of projected features with shape (batch_size, output_dim).
        """

        return self.layers(x)

class LinearEvalHead(nn.Module):
    """
    Linear evaluation head for supervised learning tasks.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.
    num_classes : int
        Number of classes for classification.
    """
    def __init__(self, input_dim: int, num_classes: int):
        """
        Initializes the LinearEvalHead module.`__init__`: Initializes the 
        module with the specified input and output dimensions,
        and defines the linear layer.
        """
        super(LinearEvalHead, self).__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the linear evaluation head.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of features with shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of logits with shape (batch_size, num_classes).
        """
        return self.classifier(x)

class SimCLR(L.LightningModule):
    """
    SimCLR model for self-supervised contrastive learning.

    Parameters
    ----------
    backbone : nn.Module
        Backbone model for feature extraction.
    projector_dim : int
        Input dimensionality for the projection head.
    hidden_dim : int
        Hidden layer dimensionality for the projection head.
    output_dim : int
        Output dimensionality for the projection head.
    temperature : float, optional, default=0.5
        Temperature parameter for contrastive loss.
    lr : float, optional, default=1e-3
        Learning rate for the optimizer.
    test_metric : Callable, optional
        Metric function for evaluating performance on the test set.
    num_classes : int, optional
        Number of classes for supervised classification tasks.
    """
    def __init__(
        self,
        backbone: nn.Module,
        projector_dim: int,
        hidden_dim: int,
        output_dim: int,
        temperature: float = 0.5,
        lr: float = 1e-3,
        test_metric: Optional[Callable] = None,
        num_classes: Optional[int] = None,
    ):
        """
        Initializes the SimCLR model.

        Parameters
        ----------
        backbone : nn.Module
            Backbone model for feature extraction.
        projector_dim : int
            Input dimensionality for the projection head.
        hidden_dim : int
            Hidden layer dimensionality for the projection head.
        output_dim : int
            Output dimensionality for the projection head.
        temperature : float, optional, default=0.5
            Temperature parameter for contrastive loss.
        lr : float, optional, default=1e-3
            Learning rate for the optimizer.
        test_metric : Callable, optional
            Metric function for evaluating performance on the test set.
        num_classes : int, optional
            Number of classes for supervised classification tasks.
        """
        super(SimCLR, self).__init__()
        self.backbone = backbone
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.projector = ProjectionHead(projector_dim, hidden_dim, output_dim)
        self.temperature = temperature
        self.lr = lr
        self.test_metric = test_metric
        self.num_classes = num_classes
        if num_classes is not None:
            self.classifier = LinearEvalHead(output_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the SimCLR model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of features with shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of projected features with shape (batch_size, output_dim).
        """
        features = self.backbone(x)
        pooled = self.avgpool(features)
        flattened = torch.flatten(pooled, 1)
        projections = self.projector(flattened)
        return normalize(projections, dim=1)

    def nt_xent_loss(self, projections: torch.Tensor) -> torch.Tensor:
        """
        Computes the normalized temperature-scaled cross-entropy loss.

        Parameters
        ----------
        projections : torch.Tensor
            Projections of the input batch.

        Returns
        -------
        torch.Tensor
            Contrastive loss value.
        """
        batch_size = projections.size(0)
        similarity_matrix = torch.mm(projections, projections.T) / self.temperature
        labels = torch.arange(batch_size).to(projections.device)
        loss = nn.CrossEntropyLoss()(similarity_matrix, labels)
        return loss

    def _single_step(self, batch: Tuple[torch.Tensor, Any]) -> torch.Tensor:
        """
        Performs a single forward and loss computation step.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, Any]
            Input batch containing images and optional labels.

        Returns
        -------
        torch.Tensor
            Computed loss for the batch.
        """
        images, _ = batch  # Labels are not used for contrastive loss
        projections = self(images)
        loss = self.nt_xent_loss(projections)
        return loss

    def training_step(self, batch: Tuple[torch.Tensor, Any], batch_idx: int) -> torch.Tensor:
        """
        Training step.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, Any]
            Input batch containing images and optional labels.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            Computed loss for the batch.
        """
        loss = self._single_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, Any], batch_idx: int) -> torch.Tensor:
        """
        Validation step.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, Any]
            Input batch containing images and optional labels.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            Computed loss for the batch.
        """
        loss = self._single_step(batch)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: Tuple[torch.Tensor, Any], batch_idx: int) -> torch.Tensor:
        """
        Test step.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, Any]
            Input batch containing images and optional labels.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            Computed loss for the batch.
        """
        loss = self._single_step(batch)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch: Tuple[torch.Tensor, Any], batch_idx: int, dataloader_idx: Optional[int] = None) -> torch.Tensor:
        """
        Predict step.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, Any]
            Input batch containing images and optional labels.
        batch_idx : int
            Index of the current batch.
        dataloader_idx : Optional[int], optional
            Index of the dataloader, by default None

        Returns
        -------
        torch.Tensor
            Computed loss for the batch.
        """
        images, _ = batch
        return self(images)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configures the optimizer for training.

        Returns
        -------
        torch.optim.Optimizer
            Optimizer instance.
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)
