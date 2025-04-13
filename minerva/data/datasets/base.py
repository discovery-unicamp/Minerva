from typing import Any, Iterable, List, Optional, Tuple, Union

from torch.utils.data import Dataset

from minerva.data.readers.reader import _Reader
from minerva.transforms.transform import _Transform


class SimpleDataset(Dataset):
    """Dataset is responsible for loading data from multiple readers and
    responsible for loading data from multiple readers and

    apply specified transforms. It is a generic implementation that can be
    used to create differents dataset, from supervised to unsupervised ones.

    This class implements the common pipeline for reading and transforming data.
    The __getitem__ pipeline is as follows:

    For each reader R and transform list T:
        1. Read the data from the reader R at the index idx.
        2. Apply the transforms T to the data.
        3. Append the transformed data to the list of data.
    Return the tuple of transformed data.
    """

    def __init__(
        self,
        readers: Union[_Reader, List[_Reader]],
        transforms: Optional[Union[_Transform, List[_Transform]]] = None,
        return_single: bool = False,
    ):
        """Load data from multiple sources and apply specified transforms.

        Parameters
        ----------
        readers : Union[_Reader, List[_Reader]]
            The list of readers to load data from. It can be a single reader or
            a list of readers.
        transforms : Optional[Union[_Transform, List[_Transform]]], optional
            The list of transforms to apply to each sample. This can be:
            -   None, in which case no transform is applied.
            -   A single transform, in which case the same transform is applied
                to data from all readers.
            -   A list of transforms, in which case each transform is applied
                to the corresponding reader. That is, the first transform is
                applied to the first reader, the second transform is applied to
                the second reader, and so on.
        return_single : bool, optional
            If True, the __getitem__ method will return a single sample  when
            a single reader is used. This is useful for unsupervised datasets,
            where we usually have a single reader. If False, the __getitem__
            method will return a tuple of samples, where each sample is from a
            different reader, from same index. This is useful for supervised
            datasets, where the data from different readers are related and
            should be returned together. The default is False.

        Examples
        --------
        1. Supervised Dataset:
        ```python
        from minerva.data.readers import ImageReader, LabelReader
        from minerva.transforms import ImageTransform, LabelTransform
        from minerva.data.datasets import SimpleDataset

        # Create the readers
        image_reader = ImageReader("path/to/images")
        label_reader = LabelReader("path/to/labels")

        # Create the transforms
        image_transform = ImageTransform()
        label_transform = None          # No transform for the labels
        # Create the dataset
        dataset = SimpleDataset(
            readers=[image_reader, label_reader],
            transforms=[image_transform, label_transform]
        )

        dataset[0]  # Returns [image, label]
        ```

        2. Unsupervised Dataset:
        ```python
        from minerva.data.readers import ImageReader
        from minerva.transforms import ImageTransform
        from minerva.data.datasets import SimpleDataset

        # Create the reader
        image_reader = ImageReader("path/to/images")

        # Create the transform
        image_transform = ImageTransform()
        # Create the dataset
        dataset = SimpleDataset(
            readers=[image_reader],
            transforms=image_transform,
            return_single=True
        )
        dataset[0]  # Returns image
        ```

        """
        self.readers = readers
        self.transforms = transforms
        self.return_single = return_single

        # ---------------- Parsing readers ----------------
        if not isinstance(self.readers, Iterable):
            self.readers = [self.readers]
        # ---------------- Parsing transforms ----------------
        # If no transform is provided, use the identity transform.
        # It will generate a list of None transforms with the same length
        # as the number of readers.
        if self.transforms is None:
            self.transforms = [None] * len(self.readers)

        # If a single transform is provided, use the same transform for all
        # readers, that is, generate a list of the same transform with the same
        # length as the number of readers.
        if not isinstance(self.transforms, Iterable):
            self.transforms = [self.transforms] * len(self.readers)

        # ---------------- Validating objects ----------------
        assert len(self.readers) == len(
            self.transforms
        ), "The number of readers and transforms must be the same."

        # If return_single is True, there must be only one reader.
        assert (
            not self.return_single or len(self.readers) == 1
        ), "If return_single is True, there must be only one reader."

    def __len__(self) -> int:
        """The length of the dataset is the length of the first reader.

        Returns
        -------
        int
            The number of samples in the dataset.
        """
        return len(self.readers[0])

    def __getitem__(self, idx: int) -> Union[Any, Tuple[Any, ...]]:
        """Load data from multiple sources and apply specified transforms.

        Parameters
        ----------
        idx : int
            The index of the sample to load.

        Returns
        -------
        List[Any]
            A list of transformed data from each reader.
        """
        data = []

        # For each reader and transform, read the data and apply the transform.
        # Then, append the transformed data to the list of data.
        for reader, transform in zip(self.readers, self.transforms):
            sample = reader[idx]
            # Apply the transform if it is not None
            if transform is not None:
                sample = transform(sample)
            data.append(sample)
        # Return the list of transformed data or a single sample if return_single
        # is True and there is only one reader.
        if self.return_single:
            return data[0]
        else:
            return tuple(data)

    def __str__(self) -> str:
        readers = self.readers if isinstance(self.readers, list) else [self.readers]
        transforms = (
            self.transforms if isinstance(self.transforms, list) else [self.transforms]
        )

        readers_info = "\n".join(
            [
                f"   └── Reader {i}: {reader}\n   │     └── Transform: {transform}"
                for i, (reader, transform) in enumerate(zip(readers, transforms))
            ]
        )

        return (
            f"{'=' * 50}\n"
            f"{'📂 SimpleDataset Information':^50}\n"
            f"{'=' * 50}\n"
            f"📌 Dataset Type: {self.__class__.__name__}\n"
            f"{readers_info}\n"
            f"   │\n"
            f"   └── Total Readers: {len(self.readers)}\n"
            f"{'=' * 50}"
        )

    def __repr__(self) -> str:
        return self.__str__()
