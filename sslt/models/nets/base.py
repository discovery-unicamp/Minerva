from typing import Dict
import torch
import lightning as L


class SimpleReconstructionNet(L.LightningModule):
    """Simple autoencoder pipeline for reconstruction tasks

    This class implements a very common pipeline for autoencoder models, which
    are used to reconstruct the input data. It consists in:

    1. Make a forward pass with the input data on the backbone model;
    2. Compute the loss between the output and the input data;
    3. Optimize the model parameters with respect to the loss.

    This reduces the code duplication for autoencoder models, and makes it
    easier to implement new models by only changing the backbone model. More
    complex models, that does not follow this pipeline, should not inherit from
    this class.

    Note that this class assumes that input data is a single tensor and not a
    tuple of tensors (e.g., data and label).
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        learning_rate: float = 1e-3,
        loss_fn: torch.nn.Module = None,
    ):
        """Simple autoencoder pipeline for reconstruction tasks.

        Parameters
        ----------
        backbone : torch.nn.Module
            The backbone model that will be used to make the forward pass and
            will be optimized with respect to the loss.
        learning_rate : float, optional
            The learning rate to Adam optimizer, by default 1e-3
        loss_fn : torch.nn.Module, optional
            The function used to compute the loss. If `None`, it will be used
            the MSELoss, by default None.
        """
        super().__init__()
        self.backbone = backbone
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn or torch.nn.MSELoss()

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
        return self.backbone(x)

    def _single_step(
        self, batch: torch.Tensor, batch_idx: int, step_name: str
    ) -> torch.Tensor:
        """Perform a single train/validation/test step. It consists in making a
        forward pass with the input data on the backbone model, computing the
        loss between the output and the input data, and logging the loss.

        Parameters
        ----------
        batch : torch.Tensor
            The input data. It must be a single tensor and not a tuple of
            tensors (e.g., data and label).
        batch_idx : int
            The index of the batch.
        step_name : str
            The name of the step. It will be used to log the loss.

        Returns
        -------
        torch.Tensor
            A tensor with the loss value.
        """
        x = batch
        y_hat = self.forward(x)
        loss = self._loss_func(y_hat, x)
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
        x, y = batch
        y_hat = self.forward(x)
        return y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )
        return optimizer
