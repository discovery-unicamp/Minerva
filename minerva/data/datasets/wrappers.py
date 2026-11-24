import wrapt
from typing import Any, Callable, Optional, Sequence, Tuple

from torch.utils.data import Dataset


class SSLDatasetWrapper(wrapt.ObjectProxy):
    """
    Wrapper around a dataset to apply self-supervised learning technique transforms.

    Parameters
    ----------
    dataset : Dataset
        The underlying PyTorch or Minerva dataset to wrap.
    technique_transforms : callable, optional
        A dict-in, dict-out transform (such as ``DINOTransform``) applied to
        every sample.
    input_key : str, optional
        Key given to the first element of a tuple sample, by default "image".
    target_key : str, optional
        Key given to the second element of a tuple sample, by default "label".

    Notes
    -----
    Samples that are already dicts are transformed as they are. Tuple samples
    ``(input, target, ...)`` from tuple-based datasets are first converted to
    ``{input_key: input, target_key: target}``, bridging them into the dict
    pipeline; a bare (non-tuple) sample becomes ``{input_key: sample}``.
    """

    def __init__(
        self,
        dataset: Dataset,
        technique_transforms: Optional[Callable],
        input_key: str = "image",
        target_key: str = "label",
    ) -> None:
        super().__init__(dataset)
        self.__wrapped__ = dataset
        self.__transforms__ = technique_transforms
        self.__input_key__ = input_key
        self.__target_key__ = target_key

    def __getitem__(self, idx: int) -> Any:
        """
        Get sample by index with SSL technique transforms applied.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        Any
            The sample, as a dict if technique transforms are set.
        """
        item = self.__wrapped__[idx]

        if self.__transforms__ is None:
            return item

        if isinstance(item, dict):
            data = item
        elif isinstance(item, (tuple, list)):
            data = {self.__input_key__: item[0]}
            if len(item) > 1:
                data[self.__target_key__] = item[1]
        else:
            data = {self.__input_key__: item}
        return self.__transforms__(data)


class ToTupleDataset(Dataset):
    """Outward compat wrapper: a dict dataset as a ``(data, label)`` tuple dataset.

    Lets external PyTorch training loops that expect tuples consume a
    :class:`~minerva.data.datasets.MinervaDataset`.

    Parameters
    ----------
    dataset : Dataset
        A dataset whose samples are dicts.
    data_key : str, optional
        Key of the data element, by default "image".
    target_key : str, optional
        Key of the target element, by default "label". Pass None to return
        only the data.
    """

    def __init__(
        self,
        dataset: Dataset,
        data_key: str = "image",
        target_key: Optional[str] = "label",
    ) -> None:
        self.dataset = dataset
        self.data_key = data_key
        self.target_key = target_key

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore

    def __getitem__(self, idx: int) -> Any:
        sample = self.dataset[idx]
        if self.target_key is None:
            return sample[self.data_key]
        return sample[self.data_key], sample[self.target_key]


class FromTupleDataset(Dataset):
    """Inward compat wrapper: a tuple dataset (e.g. torchvision ``CIFAR10``) as a dict dataset.

    Parameters
    ----------
    dataset : Dataset
        A dataset whose samples are tuples (or a bare element).
    keys : Sequence[str], optional
        Key of each tuple element, in order, by default ``("image", "label")``.
    transforms : callable, optional
        A dict-in, dict-out transform applied to each sample.
    """

    def __init__(
        self,
        dataset: Dataset,
        keys: Sequence[str] = ("image", "label"),
        transforms: Optional[Callable] = None,
    ) -> None:
        self.dataset = dataset
        self.keys = tuple(keys)
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore

    def __getitem__(self, idx: int) -> Any:
        item = self.dataset[idx]
        if not isinstance(item, (tuple, list)):
            item = (item,)
        if len(item) != len(self.keys):
            raise ValueError(
                f"Sample has {len(item)} elements but {len(self.keys)} keys were given."
            )
        data = dict(zip(self.keys, item))
        return self.transforms(data) if self.transforms is not None else data
