from typing import Any, List, Optional, Tuple

import numpy as np

from minerva.data.datasets.base import SimpleDataset
from minerva.data.readers.reader import _Reader
from minerva.transforms.transform import _Transform


class SupervisedReconstructionDataset(SimpleDataset):
    """A simple dataset class for supervised reconstruction tasks.

    In summary, each element of the dataset is a pair of data, where the first
    element is the input data and the second element is the target data.
    Usually, both input and target data have the same shape.

    This dataset is useful for supervised tasks such as image reconstruction,
    semantic segmentation, and object detection, where the input data is the
    original data and the target is a mask or a segmentation map.

    Examples
    --------

    1. Semantic Segmentation Dataset:

        ```python
        from minerva.data.readers import ImageReader
        from minerva.transforms import ImageTransform
        from minerva.data.datasets import SupervisedReconstructionDataset

        # Create the readers
        image_reader = ImageReader("path/to/images")
        mask_reader = ImageReader("path/to/masks")

        # Create the transforms
        image_transform = ImageTransform()

        # Create the dataset
        dataset = SupervisedReconstructionDataset(
            readers=[image_reader, mask_reader],
            transforms=image_transform
        )
        # Load the first sample
        dataset[0]  # Returns a tuple: (image, mask)
        ```
    """

    def __init__(
        self,
        readers: List[_Reader],
        transforms: Optional[_Transform] = None,
        support_context_transforms: bool = False,
    ):
        """A simple dataset class for supervised reconstruction tasks.

        Parameters
        ----------
        readers: List[_Reader]
            List of data readers. It must contain exactly 2 readers.
            The first reader for the input data and the second reader for the
            target data.
        transforms: Optional[_Transform]
            Optional data transformation pipeline.

        Raises
        -------
            AssertionError: If the number of readers is not exactly 2.
        """
        super().__init__(readers, transforms)
        self.support_context_transforms = support_context_transforms

        assert (
            len(self.readers) == 2
        ), "SupervisedReconstructionDataset requires exactly 2 readers"

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Load data from sources and apply specified transforms. The same
        transform is applied to both input and target data.

        Parameters
        ----------
        index : int
            The index of the sample to load.

        Returns
        -------
        Tuple[Any, Any]
            A tuple containing two elements: the input data and the target data.

        """
        if not self.support_context_transforms:
            data = super().__getitem__(index)

            return (data[0], data[1])
        else:

            data = []

            # For each reader and transform, read the data and apply the transform.
            # Then, append the transformed data to the list of data.
            for reader, transform in zip(reversed(self.readers), self.transforms):
                sample = reader[index]
                # Apply the transform if it is not None
                if transform is not None:
                    sample = transform(sample)
                data.append(sample)
            # Return the list of transformed data or a single sample if return_single
            # is True and there is only one reader.
            if self.return_single:
                return data[1]
            else:
                return tuple(reversed(data))
