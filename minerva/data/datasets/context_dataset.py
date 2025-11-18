from typing import Optional, Tuple

import numpy as np
from torch.utils.data import Dataset

from minerva.data.readers.reader import _Reader
from minerva.transforms.transform import _Transform


class ContextDataset(Dataset):

    def __init__(
        self,
        readers: Tuple[_Reader, _Reader],
        transform: Optional[_Transform] = None,
    ) -> None:
        """
        A PyTorch Dataset class for handling paired image and mask data with optional context transformations.

        Parameters
        ----------
        readers : Tuple[_Reader, _Reader]
            A tuple containing two reader objects. The first reader should provide images,
            and the second reader should provide corresponding masks. Both readers must
            support indexing and have the same length.
        transform : Optional[_Transform], default=None
            An optional transformation function or callable that takes a tuple of (image, mask)
            and returns a transformed tuple of (image, mask). If None, no transformations
            are applied.
        """
        self.readers = readers
        self.transform = transform

    def __len__(self) -> int:
        return len(self.readers[0])

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        img = self.readers[0][idx]
        mask = self.readers[1][idx]

        if self.transform:
            img, mask = self.transform((img, mask))

        return (img, mask)
