from typing import List, Tuple

import numpy as np
from base import SimpleDataset

from sslt.data.readers.reader import _Reader
from sslt.transforms.transform import TransformPipeline


class SupervisedSemanticSegmentationDataset(SimpleDataset):
    """
    A dataset class for supervised semantic segmentation tasks.
    """

    def __init__(
        self, readers: List[_Reader], transforms: TransformPipeline | None = None
    ):
        """
        Initializes a SupervisedSemanticSegmentationDataset object.

        Parameters
        ----------
            readers: List[_Reader]
                List of data readers. It must contain exactly 2 readers.
                The first reader for the input data and the second reader for the target data.
            transforms: TransformPipeline | None
                Optional data transformation pipeline.

        Raises:
            AssertionError: If the number of readers is not exactly 2.
        """
        super().__init__(readers, transforms)

        self.readers = readers
        self.transforms = transforms

        assert (
            len(self.readers) == 2
        ), "SupervisedSemanticSegmentationDataset requires exactly 2 readers"

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data from sources and apply specified transforms.

        Parameters
        ----------
        index : int
            The index of the sample to load.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing two numpy arrays representing the data.

        """
        data = [reader[index] for reader in self.readers]

        if self.transforms is not None:
            data = [self.transforms(data[i]) for i in range(2)]

        return (data[0], data[1])
