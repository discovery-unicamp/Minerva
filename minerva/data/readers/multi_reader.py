from typing import Any, Callable, Optional, Sequence
import numpy as np

from minerva.data.readers import _Reader


class MultiReader(_Reader):
    """Reader that composes items from other readers.

    Its i-th item is the i-th item of each of the child-readers merged
    together according to a collate_fn function."""

    def __init__(
        self,
        readers: Sequence[_Reader],
        preprocess: Optional[Callable] = None,
        collate_fn: Optional[Callable] = np.stack,
    ):
        """Collects data from multiple readers and collates them

        Parameters
        ----------
        readers: Sequence[_Reader]
            The readers from which the data will be collected. At least one must be
            provided. If the readers have different lengths, data will only be
            collected up until the length of the smallest child-reader.
        preprocess: Optional[Callable]
            A function to be applied individually to each item read from the child-readers.
            Defaults to an identity function (i.e. no changes to the data).
        collate_fn: Optional[Callable]
            A function that recieves a list of items read from the child-readers after
            preprocessing and returns a single item for this reader.
            Defaults to numpy.stack, which means it must be provided if the preprocessing
            function does not always return same-shape numpy arrays.
        """
        assert len(readers) > 0, "MultiReader expects at least one reader as argument."

        self._readers = readers
        self.preprocess = preprocess or (lambda x: x)
        self.collate_fn = collate_fn

    def __len__(self) -> int:
        """Returns the length the reader, defined as the length of the smallest
        child-reader

        Returns
        -------
        int
            The length of the reader."""
        return min(len(reader) for reader in self._readers)

    def __getitem__(self, index: int) -> Any:
        """Retrieves the items from each reader at the specified index and collates them
        accordingly.

        Parameters
        ----------
        index : int
            Index of the item to retrieve.

        Returns
        -------
        Any
            An item from the reader.
        """

        return self.collate_fn(
            [self.preprocess(reader[index]) for reader in self._readers]
        )
