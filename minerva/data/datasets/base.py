from collections import OrderedDict
from typing import Any, List, Mapping, Optional, Union

from torch.utils.data import Dataset, Subset as _Subset, ConcatDataset as _ConcatDataset

from minerva.data.readers.reader import _Reader
from minerva.transforms.transform import _Transform

import random


class MinervaDataset(Dataset):
    """Combine keyed readers with one transform pipeline.

    Every sample is a flat dict (an ``OrderedDict``). Its keys are the keys of
    ``readers``; its values are what each reader returns at the index. The
    transform pipeline then takes that dict and returns the final sample:

        data = OrderedDict((key, reader[idx]) for key, reader in readers.items())
        return transforms(data)

    A new task is a new set of readers and transforms. Do not subclass unless
    it is a true edge case (e.g. filtering to a class percentage).
    """

    def __init__(
        self,
        readers: "OrderedDict[str, _Reader]",
        transforms: Optional[_Transform] = None,
    ):
        """
        Parameters
        ----------
        readers : OrderedDict[str, _Reader]
            Readers keyed by the name of their output in the sample dict.
            Readers do not need to know their key. The length of the dataset
            is the length of the first reader.
        transforms : _Transform, optional
            One dict-in, dict-out transform (usually a ``_TransformsPipeline``)
            applied to the whole sample. If None, the sample is returned as
            read.

        Examples
        --------
        Supervised segmentation:

        ```python
        from collections import OrderedDict
        from minerva.data.datasets import MinervaDataset
        from minerva.transforms import (
            CropTransform, NormalizeTransform, _TransformsPipeline,
        )

        dataset = MinervaDataset(
            readers=OrderedDict(image=image_reader, label=label_reader),
            transforms=_TransformsPipeline([
                CropTransform(["image", "label"], (256, 256)),
                NormalizeTransform(["image"], mean=[0.5], std=[0.5]),
            ]),
        )
        dataset[0]  # {"image": ..., "label": ...}
        ```

        Unsupervised: use a single reader.

        ```python
        dataset = MinervaDataset(OrderedDict(image=image_reader), transforms)
        dataset[0]  # {"image": ...}
        ```
        """
        if not isinstance(readers, Mapping):
            raise TypeError(
                "`readers` must be a mapping (e.g. OrderedDict) from key to reader, "
                f"got {type(readers).__name__}."
            )
        if len(readers) == 0:
            raise ValueError("`readers` must contain at least one reader.")
        self.readers = OrderedDict(readers)
        self.transforms = transforms

    def __len__(self) -> int:
        """The length of the dataset is the length of the first reader."""
        return len(next(iter(self.readers.values())))

    def __getitem__(self, idx: int) -> "OrderedDict[str, Any]":
        data = OrderedDict((key, reader[idx]) for key, reader in self.readers.items())
        if self.transforms is not None:
            data = self.transforms(data)
        return data

    def __str__(self) -> str:
        readers_info = "\n".join(
            f"   └── Reader '{key}': {reader}" for key, reader in self.readers.items()
        )
        return (
            f"{'=' * 50}\n"
            f"{'📂 MinervaDataset Information':^50}\n"
            f"{'=' * 50}\n"
            f"📌 Dataset Type: {self.__class__.__name__}\n"
            f"{readers_info}\n"
            f"   │\n"
            f"   └── Transforms: {self.transforms}\n"
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
