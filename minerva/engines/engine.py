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
