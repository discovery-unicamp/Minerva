from typing import Any, Iterable, List, Optional, Tuple, Union

from torch.utils.data import Dataset, Subset as _Subset, ConcatDataset as _ConcatDataset

from minerva.data.readers.reader import _Reader
from minerva.transforms.transform import _Transform

import random


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
        transforms: Optional[
            Union[Optional[_Transform], List[Optional[_Transform]]]
        ] = None,
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
                the second reader, and so on. The transform can also be None for
                some readers, in which case no transform is applied to that
                reader's data.
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


class Subset(_Subset):
    def __init__(self, dataset: Dataset, indices: List[int]):
        """Create a subset of a dataset with specified indices.
        Parameters
        ----------
        dataset : Dataset
            The dataset to create a subset from.
        indices : List[int]
            The indices of the samples to include in the subset.
        """
        super().__init__(dataset, indices)

    def __str__(self) -> str:
        return f"{self.dataset}\nSubset with {len(self.indices)} samples"


class FractionalSubset(_Subset):
    def __init__(self, dataset: Dataset, fraction: Union[float, int]):
        """Create a subset of a dataset with a specified fraction or fixed number of samples.
        Parameters
        ----------
        dataset : Dataset
            The dataset to create a subset from.
        fraction : float or int
            The fraction can be:
            - A float (from 0 to 1) representing the percentage of the dataset to include; or
            - An integer from 1 to the size of the dataset representing the number of samples to include.
        Raises
        ------
        ValueError
            If the fraction is not in the valid range.
        TypeError
            If the fraction is not a float or an int.
        """
        dataset_size = len(dataset)

        if isinstance(fraction, float):
            if not (0 < fraction <= 1):
                raise ValueError("Fraction as float must be between 0 and 1.")
            num_samples = int(dataset_size * fraction)
        elif isinstance(fraction, int):
            if not (0 < fraction <= dataset_size):
                raise ValueError(
                    "Integer fraction must be between 1 and the size of the dataset."
                )
            num_samples = fraction
        else:
            raise TypeError("Fraction must be a float or an int.")

        self.fraction = fraction
        indices = list(range(dataset_size))
        selected_indices = indices[:num_samples]
        super().__init__(dataset, selected_indices)

    def __str__(self) -> str:
        desc = f"{self.dataset}\nFractional Subset with {len(self.indices)} samples"
        if isinstance(self.fraction, float):
            desc += f" ({self.fraction * 100:.2f}%)"
        return desc


class FractionalRandomSubset(_Subset):
    def __init__(
        self, dataset: Dataset, fraction: Union[float, int], seed: Optional[int] = None
    ):
        """Create a random subset of a dataset with a specified fraction or fixed number of samples.
        Parameters
        ----------
        dataset : Dataset
            The dataset to create a subset from.
        fraction : float or int
            The fraction can be:
            - A float (from 0 to 1) representing the percentage of the dataset to include; or
            - An integer from 1 to the size of the dataset representing the number of samples to include.
        seed : Optional[int], optional
            The random seed for reproducibility, by default None.
        Raises
        ------
        ValueError
            If the fraction is not in the valid range.
        TypeError
            If the fraction is not a float or an int.

        """
        dataset_size = len(dataset)

        if isinstance(fraction, float):
            if not (0 < fraction <= 1):
                raise ValueError("Fraction as float must be between 0 and 1.")
            num_samples = int(dataset_size * fraction)
        elif isinstance(fraction, int):
            if not (0 < fraction <= dataset_size):
                raise ValueError(
                    "Integer fraction must be between 1 and the size of the dataset."
                )
            num_samples = fraction
        else:
            raise TypeError("Fraction must be a float or an int.")

        self.seed = seed
        self.rng = random.Random(seed)
        self.fraction = fraction
        indices = list(range(dataset_size))
        selected_indices = self.rng.sample(indices, num_samples)
        super().__init__(dataset, selected_indices)

    def __str__(self) -> str:
        desc = (
            f"{self.dataset}\nRandom Fractional Subset with {len(self.indices)} samples"
        )
        if isinstance(self.fraction, float):
            desc += f" ({self.fraction * 100:.2f}%)"
        desc += f". Using seed: {self.seed}"
        return desc


class ConcatDataset(_ConcatDataset):
    def __str__(self) -> str:
        return f"{self.datasets}\nConcatenated {len(self.datasets)}. Totaling {len(self)} samples"
