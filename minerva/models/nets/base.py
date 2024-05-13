from typing import Dict, Optional

import lightning as L
import torch


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
        backbone: torch.nn.Module,
        fc: torch.nn.Module,
        loss_fn: torch.nn.Module,
        learning_rate: float = 1e-3,
        flatten: bool = True,
    ):
        """Initialize the model.

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
        """
        super().__init__()
        self.backbone = backbone
        self.fc = fc
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.flatten = flatten

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
            x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

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


class LoggedSupervisedModel(L.LightningModule):
    """Simple pipeline for supervised models with logging."""

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        learning_rate: float = 1e-3,
        flatten: bool = True,
        train_metric: Optional[torch.nn.Module] = None,
        val_metric: Optional[torch.nn.Module] = None,
        test_metric: Optional[torch.nn.Module] = None,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.flatten = flatten

        if train_metric is not None:
            self.train_metric = train_metric
            self.log_train_metric = True

        if val_metric is not None:
            self.val_metric = val_metric
            self.log_val_metric = True

        if test_metric is not None:
            self.test_metric = test_metric
            self.log_test_metric = True

        self.train_step_outputs = []
        self.train_step_labels = []

        self.val_step_outputs = []
        self.val_step_labels = []

        self.test_step_outputs = []
        self.test_step_labels = []

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
        return self.model(x)

    def _single_step(
        self, batch: torch.Tensor, batch_idx: int, step_name: str
    ) -> torch.Tensor:

        x, y = batch
        y_hat = self.forward(x)
        loss = self._loss_func(y_hat, y)

        if step_name == "train":
            self.train_step_outputs.append(y_hat)
            self.train_step_labels.append(y)

        elif step_name == "val":
            self.val_step_outputs.append(y_hat)
            self.val_step_labels.append(y)

        elif step_name == "test":
            self.test_step_outputs.append(y_hat)
            self.test_step_labels.append(y)

        self.log(
            f"{step_name}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        return self._single_step(batch, batch_idx, step_name="train")

    def on_train_batch_end(self):
        if self.log_train_metric:
            y_hat = torch.cat(self.train_step_outputs)
            y = torch.cat(self.train_step_labels)
            metric = self.train_metric(y_hat, y)
            self.log_dict(
                {"train_metric": metric},
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        self.train_step_outputs.clear()
        self.train_step_labels.clear()

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        return self._single_step(batch, batch_idx, step_name="val")

    def on_validation_batch_end(self):
        if self.log_val_metric:
            y_hat = torch.cat(self.val_step_outputs)
            y = torch.cat(self.val_step_labels)
            metric = self.val_metric(y_hat, y)
            self.log_dict(
                {"val_metric": metric},
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        self.val_step_outputs.clear()
        self.val_step_labels.clear()

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        return self._single_step(batch, batch_idx, step_name="test")

    def on_test_batch_end(self):
        if self.log_test_metric:
            y_hat = torch.cat(self.test_step_outputs)
            y = torch.cat(self.test_step_labels)
            metric = self.test_metric(y_hat, y)
            self.log_dict(
                {"test_metric": metric},
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        self.test_step_outputs.clear()
        self.test_step_labels.clear()

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
