from typing import Any, Union

import lightning.pytorch as L
import numpy as np
import torch


class _Engine:
    def __init__(self) -> None:
        super().__init__()

    def __call__(
        self,
        model: Union[L.LightningModule, torch.nn.Module],
        x: Union[torch.Tensor, np.ndarray],
    ):
        raise NotImplementedError
