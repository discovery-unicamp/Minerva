from typing import Union
import numpy as np


class _Reader:
    """
    A base class for reading data from a directory.

    Args:
        path (Union[str, Path]): The path to the file or directory.

    Raises:
        NotImplementedError: This class is meant to be subclassed and the methods should be implemented in the derived classes.

    """

    def __init__(self):
        raise NotImplementedError()

    def __getitem__(self, index: Union[int, slice]) -> np.ndarray:
        """
        Retrieve an item or a slice from the reader.

        Args:
            index (int or slice): The index or slice to retrieve.

        Returns:
            np.ndarray: The retrieved item or slice.

        Raises:
            NotImplementedError: This method is not implemented and should be overridden in a subclass.
        """
        raise NotImplementedError()

    def __len__(self) -> int:
        """
        Returns the length of the Reader object.

        :return: The length of the Reader object.
        :rtype: int
        """
        raise NotImplementedError()
