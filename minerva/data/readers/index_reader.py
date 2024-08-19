from typing import Optional

from minerva.data.readers.reader import _Reader


class IndexReader(_Reader):
    """
    A class that returns the asked index as the item. Useful for some ssl methods and techniques.
    If you previously have the length of your dataset you can set it on the initialization, otherwise when calling len it will return None.
    This class does not support slicing, negative indexes or out of range indexes.
    """

    def __init__(self, len: Optional[int] = None) -> None:
        super().__init__()
        assert (
            len is None or len > 0
        ), "The length of the dataset must be a positive number."
        self.len = len

    def __getitem__(self, index: int) -> int:
        if index >= 0:
            return index
        raise IndexError("Negative indexes are not supported.")

    def __len__(self) -> int:
        if self.len:
            return self.len
        raise TypeError("The length of the dataset is not defined.")
