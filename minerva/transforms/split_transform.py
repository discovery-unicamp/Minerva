import numpy as np
import torch
from .transform import _Transform
from typing import Union, Tuple


class SplitTransform(_Transform):

    def __init__(self, num_splits: int = 2, split_dimension: int = 0):
        """A transform that splits the input data along some dimension.
        When applied to a dataset, this transform will split the input data into
        the specified number of splits.

        Parameters
        ----------
        num_splits : int
            The number of splits to divide the input into.
        split_dimension : int
            The dimension along which to split the input data.
        """
        super().__init__()
        self.num_splits = num_splits
        self.split_dimension = split_dimension

        if num_splits <= 0:
            raise ValueError(
                f"Expected input 'num_splits' to be a positive integer greater than 0, but received {num_splits}."
            )
        if split_dimension < 0:
            raise ValueError(
                f"Expected input 'split_dimension' to be a positive integer greater than or equal to 0, but received {split_dimension}."
            )

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Tuple:
        """Split the input data into the specified number of splits.

        Parameters
        ----------
        x : Union[np.ndarray, torch.Tensor]
            The input data to split.

        Returns
        -------
        Tuple
            The split data.
        """
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
            splits = np.split(x, self.num_splits, axis=self.split_dimension)
        elif isinstance(x, torch.Tensor):
            splits = torch.split(
                x,
                x.shape[self.split_dimension] // self.num_splits,
                dim=self.split_dimension,
            )
        return splits
