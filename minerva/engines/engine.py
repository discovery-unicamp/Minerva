from typing import Any, Union

import lightning.pytorch as L
import numpy as np
import torch


class _Engine:
    """Main interface for Engine classes. Engines are used to alter the behavior of a model's prediction.
    An engine should be able to take a `model` and input data `x` and return a prediction.
    An use case for Engines is patched inference, where the model's default input size is smaller them the desired input size.
    The engine can be used to make predictions in patches and combine this predictions in to a single output.
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(
        self,
        model: Union[L.LightningModule, torch.nn.Module],
        x: Union[torch.Tensor, np.ndarray],
    ):
        raise NotImplementedError


class _Inferencer(L.LightningModule):
    """This class acts as a normal `L.LightningModule` that wraps a
    `SimpleSupervisedModel` model allowing it to perform inference with a
    custom engine (e.g., PatchInferencerEngine, SlidingWindowInferencerEngine).
    This is useful when the model's default input size is smaller than the
    desired input size (sample size). In this case, the engine split the input
    tensor into patches, perform inference in each patch, and combine them into
    a single output of the desired size. The combination of patches can be
    parametrized by a `weight_function` allowing a customizable combination of
    patches (e.g, combining using weighted average). It is important to note
    that only model's forward are wrapped, and, thus, any method that requires
    the forward method (e.g., training_step, predict_step) will be performed in
    patches, transparently to the user.
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform inference using the inference engine.

        Parameters
        ----------
        x : torch.Tensor
            Batch of input data.
        """
        return self.inferencer(self.model, x)

    def _single_step(
        self, batch: torch.Tensor, batch_idx: int, step_name: str
    ) -> torch.Tensor:
        """Perform a single step of the training/validation loop.

        Parameters
        ----------
        batch : torch.Tensor
            The input data.
        batch_idx : int
            The index of the batch.
        step_name : str
            The name of the step, either "train" or "val".

        Returns
        -------
        torch.Tensor
            The loss value.
        """
        x, y = batch
        y_hat = self.forward(x.float())
        loss = self.model._loss_func(y_hat, y.squeeze(1))

        metrics = self.model._compute_metrics(y_hat, y, step_name)
        for metric_name, metric_value in metrics.items():
            self.log(
                metric_name,
                metric_value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

        self.log(
            f"{step_name}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return loss

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        return self._single_step(batch, batch_idx, "train")

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        return self._single_step(batch, batch_idx, "val")

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        return self._single_step(batch, batch_idx, "test")
