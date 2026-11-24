from typing import Any, Dict, List, Union

import numpy as np
import torch

from .transform import _as_keys, _Transform


class SwapAxesTransform(_Transform):
    """Exchange two dimensions of every listed NumPy array or PyTorch tensor.

    Formerly ``transpose_transform.TransposeTransform``. That name now belongs
    to :class:`minerva.transforms.transform.TransposeTransform`, which
    permutes all axes.
    """

    def __init__(self, keys_to_transform: List[str], dim0: int, dim1: int):
        """
        Parameters
        ----------
        keys_to_transform : List[str]
            Keys of the entries to transform.
        dim0 : int
            The first dimension to exchange.
        dim1 : int
            The second dimension to exchange.
        """
        super().__init__()
        if dim0 < 0:
            raise ValueError(
                f"Dimension {dim0} must be a positive integer greater than or equal to 0."
            )
        if dim1 < 0:
            raise ValueError(
                f"Dimension {dim1} must be a positive integer greater than or equal to 0."
            )
        self.keys_to_transform = _as_keys(keys_to_transform)
        self.dim0 = dim0
        self.dim1 = dim1

    def _swap(self, x: Union[np.ndarray, torch.Tensor]):
        if not isinstance(x, (np.ndarray, torch.Tensor)):
            raise TypeError(
                f"Input type {type(x)} must be a numpy array or a Pytorch tensor."
            )
        if self.dim0 >= len(x.shape) or self.dim1 >= len(x.shape):
            raise ValueError(
                f"Dimensions {self.dim0} and {self.dim1} must be less than {len(x.shape)}."
            )
        if isinstance(x, np.ndarray):
            dims = list(range(x.ndim))
            dims[self.dim0], dims[self.dim1] = dims[self.dim1], dims[self.dim0]
            return np.transpose(x, dims)
        return x.transpose(self.dim0, self.dim1)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return self._map_keys(data, self._swap)
