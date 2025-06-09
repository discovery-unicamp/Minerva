from torch.utils.data import Dataset, Subset
from math import ceil, floor


def build_indices(size: int, start: int, end: int):
    """
    Recursively builds a list of `size` indices that are approximately
    evenly distributed across the interval [start, end) using a
    divide-and-conquer midpoint strategy.

    Parameters
    ----------
    size : int
        The number of indices to generate.
    start : int
        The start of the interval (inclusive).
    end : int
        The end of the interval (exclusive).

    Returns
    -------
    List[int]
        A list of indices of length `size`, approximately evenly
        spaced within the given interval.
    """
    if (end <= start) or (size <= 0):
        return []

    midpoint = (end + start) // 2

    remainder = size - 1
    left_apportion = ceil(remainder / 2)
    right_apportion = floor(remainder / 2)

    right_indices = build_indices(left_apportion, start, midpoint)
    left_indices = build_indices(right_apportion, midpoint + 1, end)

    return right_indices + [midpoint] + left_indices


class BinaryTreeSubset(Subset):
    def __init__(self, dataset: Dataset, size: int):
        """
        A subset of a PyTorch Dataset whose elements are selected using a
        binary tree-style midpoint sampling strategy for approximate even
        distribution.

        This is useful for tasks such as hierarchical sampling or balanced
        data reduction, where a representative subset of a dataset is
        desired while preserving some notion of coverage across the index
        space.

        Parameters
        ----------
        dataset : Dataset
            The base dataset from which to create the subset.
        size : int
            The number of samples to include in the subset. Must be
            positive and no greater than the length of the base dataset.

        Raises
        ------
        ValueError
            If `size` is non-positive or exceeds the size of the dataset.
        """
        if size <= 0:
            raise ValueError(f"`size` must be a positive integer, but got {size=}")
        len_base = len(dataset)  # type: ignore
        if size > len_base:
            raise ValueError(
                f"Cannot create a subset of size {size} "
                f"because the base dataset has a size of {len_base}"
            )

        super().__init__(dataset, build_indices(size, 0, len_base))

    def __str__(self):
        return f"{self.dataset} Binary Tree Subset with {len(self.indices)} samples"
