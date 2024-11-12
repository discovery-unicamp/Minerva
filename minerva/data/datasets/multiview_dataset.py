from typing import Any, Sequence, Tuple
from torch.utils.data import Dataset
from minerva.transforms.transform import _Transform
from minerva.data.readers import _Reader
import numpy as np

class MultiViewDataset(Dataset):
    """Dataset for generating stacked multi-view samples by applying specified
    transformation pipelines.

    This dataset wraps a reader, applying a series of transformation pipelines to
    each sample, then stacking the results into a single array. For each sample in
    the dataset, multiple views are generated through different transformations
    and stacked into a single output. This can be useful for contrastive learning,
    multi-view representations, or other tasks requiring varied perspectives on the
    data.
    """

    def __init__(
        self,
        reader: _Reader,
        transform_pipelines: Sequence[_Transform]
    ):
        """Initialize the multi-view dataset with a reader and transformation pipelines.

        Parameters
        ----------
        reader : \_Reader
            The reader instance used to load data samples.
        transform_pipelines : Sequence[\_Transform]
            A sequence of transformations to apply to each sample. Each pipeline
            generates a distinct view of the sample, and the resulting views are stacked
            into an array.
        
        Examples
        --------
        ```python
        from minerva.data.datasets import MultiViewDataset
        from minerva.transforms import Flip, PerlinMasker
        from minerva.data.readers import ImageReader

        # Load a reader
        image_reader = ImageReader("path/to/images")

        # Define transformation pipelines
        transforms = [Flip(), PerlinMasker(3)]

        # Create the multi-view dataset
        multi_view_dataset = MultiViewDataset(image_reader, transforms)
        
        # Each __getitem__ call returns a stacked array of transformed images
        stacked_views = multi_view_dataset[0]  # Returns np.stack([view1, view2])
        ```
        """
        self.reader = reader
        self.transform_pipelines = transform_pipelines

    def __len__(self) -> int:
        """The length of the dataset is defined by the length of the reader.

        Returns
        -------
        int
            The number of samples in the dataset.
        """
        return len(self.reader)

    def __getitem__(self, idx: int) -> np.ndarray:
        """Retrieve a sample from the reader and apply each transform pipeline.

        Parameters
        ----------
        idx : int
            The index of the sample to load from the reader.

        Returns
        -------
        np.ndarray
            An array containing stacked transformed versions of the sample along axis 0
        """
        data = [P(self.reader[idx]) for P in self.transform_pipelines]
        return np.stack(data)
