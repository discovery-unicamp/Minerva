from typing import Any

import numpy as np


class _Reader:
    """
    Base class for readers. Readers define an ordered collection of data and
    provide methods to access it. This class primarily handles:

    1. Definition of data structure and storage.
    2. Reading data from the source.

    The access is handled by the __getitem__ and __len__ methods, which should be
    implemented by a subclass. Readers usually returns a single item at a time,
    that can be a single image, a single label, etc.
    """

    def __getitem__(self, index: int) -> Any:
        """Retrieve an item from the reader at the specified index.

        Parameters
        ----------
        index : int
            Index of the item to retrieve.

        Returns
        -------
        Any
            An item from the reader.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """
        Get the length of the reader.

        Returns
        -------
        int
            The length of the reader.
        """
        raise NotImplementedError
