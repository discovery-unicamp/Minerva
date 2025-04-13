from typing import Any, Dict, Optional, Union, Callable

import lightning as L
import torch
from torchmetrics import Metric
from minerva.models.loaders import LoadableModule


class SimpleSupervisedModel(L.LightningModule):
    """A modular Lightning model wrapper for supervised learning tasks.

    This class enables the construction of supervised models by combining a
    backbone (feature extractor), an optional adapter, and a fully connected
    (FC) head. It provides a clean interface for setting up custom training,
    validation, and testing pipelines with pluggable loss functions, metrics,
    optimizers, and learning rate schedulers.

    The architecture is structured as follows:

        +------------------+
        |  Backbone Model  |
        +------------------+
                |
                v
        +------------------------+
        |  Adapter (Optional)    |
        +------------------------+
                |
         (Flatten if needed)
                v
        +------------------------+
        |  Fully Connected Head  |
        +------------------------+
                |
                v
        +------------------+
        |  Loss Function   |
        +------------------+

    Training and validation steps comprise the following steps:

    1. Forward pass input through the backbone.
    2. Pass through adapter (if provided).
    3. Flatten the output (if `flatten` is True) before the FC head.
    4. Forward through the FC head.
    5. Compute loss with respect to targets.
    6. Backpropagate and update parameters.
    7. Compute metrics and log them.
    8. Return loss. `train_loss`, `val_loss`, and `test_loss` are always
       logged, along with any additional metrics specified in the
       `train_metrics`, `val_metrics`, and `test_metrics` dictionaries.

    This wrapper is especially useful to quickly set up supervised models for
    various tasks, such as image classification, object detection, and
    segmentation. It is designed to be flexible and extensible, allowing users
    to easily swap out components like the backbone, adapter, and FC head as
    needed. The model is built with a focus on simplicity and modularity, making
    it easy to adapt to different use cases and requirements.
    The model is designed to be used with PyTorch Lightning and is compatible
    with its training loop.

    **Note**: For more complex architectures that does not follow the above
    structure should not inherit from this class.

    **Note**: Input batches must be tuples (input_tensor, target_tensor).
    """

    def __init__(
        self,
        backbone: Union[torch.nn.Module, LoadableModule],
        fc: Union[torch.nn.Module, LoadableModule],
        loss_fn: torch.nn.Module,
        adapter: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        learning_rate: float = 1e-3,
        flatten: bool = True,
        train_metrics: Optional[Dict[str, Metric]] = None,
        val_metrics: Optional[Dict[str, Metric]] = None,
        test_metrics: Optional[Dict[str, Metric]] = None,
        freeze_backbone: bool = False,
        optimizer: type = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        lr_scheduler: Optional[type] = None,
        lr_scheduler_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initializes the supervised model with training components and configs.

        Parameters
        ----------
        backbone : torch.nn.Module or LoadableModule
            The backbone (feature extractor) model.
        fc : torch.nn.Module or LoadableModule
            The fully connected head. Use nn.Identity() if not required.
        loss_fn : torch.nn.Module
            Loss function to optimize during training.
        adapter : Callable, optional
            Function to transform backbone outputs before feeding into `fc`.
        learning_rate : float, default=1e-3
            Learning rate used for optimization.
        flatten : bool, default=True
            If True, flattens backbone outputs before `fc`.
        train_metrics : dict, optional
            TorchMetrics dictionary for training evaluation.
        val_metrics : dict, optional
            TorchMetrics dictionary for validation evaluation.
        test_metrics : dict, optional
            TorchMetrics dictionary for test evaluation.
        freeze_backbone : bool, default=False
            If True, backbone parameters are frozen during training.
        optimizer: type
            Optimizer class to be instantiated. By default, it is set to
            `torch.optim.Adam`. Should be a subclass of
            `torch.optim.Optimizer` (e.g., `torch.optim.SGD`).
        optimizer_kwargs : dict, optional
            Additional kwargs passed to the optimizer constructor.
        lr_scheduler : type, optional
            Learning rate scheduler class to be instantiated. By default, it is
            set to None, which means no scheduler will be used. Should be a
            subclass of `torch.optim.lr_scheduler.LRScheduler` (e.g.,
            `torch.optim.lr_scheduler.StepLR`).
        lr_scheduler_kwargs : dict, optional
            Additional kwargs passed to the scheduler constructor.
        """

        super().__init__()
        self.backbone = backbone
        self.fc = fc
        self.loss_fn = loss_fn
        self.adapter = adapter
        self.learning_rate = learning_rate
        self.flatten = flatten
        self.freeze_backbone = freeze_backbone
        self.metrics = {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        }
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs or {}

    def _loss_func(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculate the loss between the output and the input data.

        Parameters
        ----------
        y_hat : torch.Tensor
            The output data from the forward pass.
        y : torch.Tensor
            The input data/label.

        Returns
        -------
        torch.Tensor
            The loss value.
        """
        loss = self.loss_fn(y_hat, y)
        return loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass with the input data on the backbone model.

        Parameters
        ----------
        x : torch.Tensor
            The input data.

        Returns
        -------
        torch.Tensor
            The output data from the forward pass.
        """
        x = self.backbone(x)
        if self.flatten:
            x = x.reshape(x.size(0), -1)
        if self.adapter is not None:
            x = self.adapter(x)
        x = self.fc(x)
        return x

    def _compute_metrics(
        self, y_hat: torch.Tensor, y: torch.Tensor, step_name: str
    ) -> Dict[str, torch.Tensor]:
        """Calculate the metrics for the given step.

        Parameters
        ----------
        y_hat : torch.Tensor
            The output data from the forward pass.
        y : torch.Tensor
            The input data/label.
        step_name : str
            Name of the step. It will be used to get the metrics from the
            `self.metrics` attribute.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary with the metrics values.
        """
        if self.metrics[step_name] is None:
            return {}

        return {
            f"{step_name}_{metric_name}": metric.to(self.device)(y_hat, y)
            for metric_name, metric in self.metrics[step_name].items()
        }

    def _single_step(
        self, batch: torch.Tensor, batch_idx: int, step_name: str
    ) -> torch.Tensor:
        """Perform a single train/validation/test step. It consists in making a
        forward pass with the input data on the backbone model, computing the
        loss between the output and the input data, and logging the loss.

        Parameters
        ----------
        batch : torch.Tensor
            The input data. It must be a 2-element tuple of tensors, where the
            first tensor is the input data and the second tensor is the mask.
        batch_idx : int
            The index of the batch.
        step_name : str
            The name of the step. It will be used to log the loss. The possible
            values are: "train", "val" and "test". The loss will be logged as
            "{step_name}_loss".

        Returns
        -------
        torch.Tensor
            A tensor with the loss value.
        """
        x, y = batch
        y_hat = self.forward(x)
        loss = self._loss_func(y_hat, y)
        self.log(
            f"{step_name}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        metrics = self._compute_metrics(y_hat, y, step_name)
        for metric_name, metric_value in metrics.items():
            self.log(
                metric_name,
                metric_value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        return loss

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        return self._single_step(batch, batch_idx, step_name="train")

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        return self._single_step(batch, batch_idx, step_name="val")

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        return self._single_step(batch, batch_idx, step_name="test")

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, _ = batch
        y_hat = self.forward(x)
        return y_hat

    def configure_optimizers(self):
        # Freeze or not the backbone model
        for param in self.backbone.parameters():
            param.requires_grad = not self.freeze_backbone
        # Unfreeze the fc model
        for param in self.fc.parameters():
            param.requires_grad = True

        optimizer = self.optimizer(
            self.parameters(), lr=self.learning_rate, **self.optimizer_kwargs
        )
        if self.lr_scheduler is None:
            return optimizer

        scheduler = self.lr_scheduler(optimizer, **self.lr_scheduler_kwargs)

        return [optimizer], [scheduler]
