from typing import Dict, Optional, Union

import lightning as L
import torch
from torchmetrics import Metric
from minerva.models.loaders import FromPretrained


class SimpleSupervisedModel(L.LightningModule):
    """Simple pipeline for supervised models.

    This class implements a very common deep learning pipeline, which is
    composed by the following steps:

    1. Make a forward pass with the input data on the backbone model;
    2. Make a forward pass with the input data on the fc model;
    3. Compute the loss between the output and the label data;
    4. Optimize the model (backbone and FC) parameters with respect to the loss.

    This reduces the code duplication for autoencoder models, and makes it
    easier to implement new models by only changing the backbone model. More
    complex models, that does not follow this pipeline, should not inherit from
    this class.
    Note that, for this class the input data is a tuple of tensors, where the
    first tensor is the input data and the second tensor is the mask or label.
    """

    def __init__(
        self,
        backbone: Union[torch.nn.Module, FromPretrained],
        fc: Union[torch.nn.Module, FromPretrained],
        loss_fn: torch.nn.Module,
        adapter: Optional[callable] = None,
        learning_rate: float = 1e-3,
        flatten: bool = True,
        train_metrics: Optional[Dict[str, Metric]] = None,
        val_metrics: Optional[Dict[str, Metric]] = None,
        test_metrics: Optional[Dict[str, Metric]] = None,
    ):
        """Initialize the model with the backbone, fc, loss function and
        metrics. Metrics are used to evaluate the model during training,
        validation, testing or prediction. It will be logged using
        lightning logger at the end of each epoch. Metrics should implement
        the `torchmetrics.Metric` interface.

        Parameters
        ----------
        backbone : torch.nn.Module
            The backbone model. Usually the encoder/decoder part of the model.
        fc : torch.nn.Module
            The fully connected model, usually used to classification tasks.
            Use `torch.nn.Identity()` if no FC model is needed.
        loss_fn : torch.nn.Module
            The function used to compute the loss.
        learning_rate : float, optional
            The learning rate to Adam optimizer, by default 1e-3
        flatten : bool, optional
            If `True` the input data will be flattened before passing through
            the fc model, by default True

        train_metrics : Dict[str, Metric], optional
            The metrics to be used during training, by default None
        val_metrics : Dict[str, Metric], optional
            The metrics to be used during validation, by default None
        test_metrics : Dict[str, Metric], optional
            The metrics to be used during testing, by default None
        predict_metrics : Dict[str, Metric], optional
            The metrics to be used during prediction, by default None
        """
        super().__init__()
        self.backbone = backbone
        self.fc = fc
        self.loss_fn = loss_fn
        self.adapter = adapter
        self.learning_rate = learning_rate
        self.flatten = flatten

        self.metrics = {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        }

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
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )
        return optimizer
