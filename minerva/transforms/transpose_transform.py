import numpy as np
import torch
from .transform import _Transform
from typing import Union, Tuple


class TransposeTransform(_Transform):
    """Exchange two dimensions of a NumPy array or PyTorch tensor."""

    def __init__(self, dim0: int, dim1: int):
        """
        A transform that transposes the input data along two dimensions. When applied to a
        dataset, this transform will swap the specified dimensions.

        Parameters
        ----------
        dim0 : int
            The first dimension to transpose.
        dim1 : int
            The second dimension to transpose.
        """
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

        if dim0 < 0:
            raise ValueError(
                f"Dimension {dim0} must be a positive integer greater than or equal to 0."
            )
        if dim1 < 0:
            raise ValueError(
                f"Dimension {dim1} must be a positive integer greater than or equal to 0."
            )

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Tuple:
        """
        Transpose the input data along the specified dimensions.

        Parameters
        ----------
        x : Union[np.ndarray, torch.Tensor]
            The input data to transpose.

        Returns
        -------
        Tuple
            The transposed data.
        """
        if not isinstance(x, (np.ndarray, torch.Tensor)):
            raise TypeError(
                f"Input type {type(x)} must be a numpy array or a Pytorch tensor."
            )
        if self.dim0 >= len(x.shape) or self.dim1 >= len(x.shape):
            raise ValueError(
                f"Dimensions {self.dim0} and {self.dim1} must be less than {len(x.shape)}."
            )
        result = None
        if isinstance(x, np.ndarray):
            dimensions = list(range(x.ndim))
            dimensions[self.dim0], dimensions[self.dim1] = (
                dimensions[self.dim1],
                dimensions[self.dim0],
            )
            result = np.transpose(x, dimensions)

        elif isinstance(x, torch.Tensor):
            result = x.transpose(self.dim0, self.dim1)
        return result
