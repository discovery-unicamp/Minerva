from typing import Any, Dict, List, Union

import numpy as np
import torch

from .transform import _as_keys, _Transform


class SplitTransform(_Transform):
    """Split every listed key into equal parts along a dimension (1 key to many).

    Output keys are ``{key}_0``, ``{key}_1``, ... ``{key}_{num_splits - 1}``.
    The source keys are removed; unlisted keys pass through unchanged.
    """

    def __init__(
        self,
        keys_to_transform: List[str],
        num_splits: int = 2,
        split_dimension: int = 0,
    ):
        """
        Parameters
        ----------
        keys_to_transform : List[str]
            Keys of the entries to split.
        num_splits : int, optional
            Number of equal parts, by default 2.
        split_dimension : int, optional
            Dimension to split along, by default 0.
        """
        super().__init__()
        if num_splits <= 0:
            raise ValueError(
                f"Expected input 'num_splits' to be a positive integer greater than 0, but received {num_splits}."
            )
        if split_dimension < 0:
            raise ValueError(
                f"Expected input 'split_dimension' to be a positive integer greater than or equal to 0, but received {split_dimension}."
            )
        self.keys_to_transform = _as_keys(keys_to_transform)
        self.num_splits = num_splits
        self.split_dimension = split_dimension

    def _split(self, x: Union[np.ndarray, torch.Tensor]):
        if not isinstance(x, (np.ndarray, torch.Tensor)):
            raise TypeError(
                f"Expected input 'x' to be either a numpy array or a Pytorch tensor, but received an object of type {type(x)}."
            )
        if self.split_dimension >= len(x.shape):
            raise ValueError(
                f"Invalid split dimension: expected the split dimension to be less than {len(x.shape)}, but received {self.split_dimension}."
            )
        if x.shape[self.split_dimension] % self.num_splits != 0:
            raise ValueError(
                f"Invalid split: expected {self.num_splits} to divide equally the dimension {x.shape[self.split_dimension]}."
            )
        if isinstance(x, np.ndarray):
            return np.split(x, self.num_splits, axis=self.split_dimension)
        return torch.split(
            x,
            x.shape[self.split_dimension] // self.num_splits,
            dim=self.split_dimension,
        )

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        outputs = {}
        for key in self.keys_to_transform:
            for i, part in enumerate(self._split(data[key])):
                outputs[f"{key}_{i}"] = part
        for key in self.keys_to_transform:
            del data[key]
        data.update(outputs)
        return data
