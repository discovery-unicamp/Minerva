# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.
#
# References:
#   https://github.com/facebookresearch/dinov3/blob/main/dinov3/data/datasets/image_net.py

import os
from enum import Enum
from io import BytesIO
from collections import OrderedDict
from typing import Callable, Optional, Tuple, Union, Literal

import numpy as np
from PIL import Image

from minerva.data.datasets.base import MinervaDataset
from minerva.data.readers.reader import _Reader
from minerva.transforms import ExternalTransform
from minerva.transforms.transform import _Transform, _TransformsPipeline


class _Split(Enum):
    """ImageNet dataset split enumeration."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"  # NOTE: torchvision does not support the test split

    @property
    def length(self) -> int:
        """Return the standard number of samples for this split."""
        split_lengths = {
            _Split.TRAIN: 1_281_167,
            _Split.VAL: 50_000,
            _Split.TEST: 100_000,
        }
        return split_lengths[self]

    def get_dirname(self, class_id: Optional[str] = None) -> str:
        """
        Get the directory name for a given class within the split.

        Parameters
        ----------
        class_id : str, optional
            The WordNet synset/class identifier.

        Returns
        -------
        str
            Directory path relative to the dataset root.
        """
        return self.value if class_id is None else os.path.join(self.value, class_id)

    def get_image_relpath(
        self, actual_index: int, class_id: Optional[str] = None
    ) -> str:
        """
        Get relative file path for an image.

        Parameters
        ----------
        actual_index : int
            Index of the image within the class or split.
        class_id : str, optional
            The WordNet synset/class identifier.

        Returns
        -------
        str
            Relative path to the image JPEG file.
        """
        if self == _Split.TRAIN:
            dirname = self.get_dirname(class_id)
            basename = f"{class_id}_{actual_index}"
        else:  # self in (_Split.VAL, _Split.TEST):
            dirname = self.value
            basename = f"ILSVRC2012_{self.value}_{actual_index:08d}"
        return os.path.join(dirname, basename + ".JPEG")

    def parse_image_relpath(self, image_relpath: str) -> Tuple[str, int]:
        """
        Parse relative image file path into class ID and index.

        Parameters
        ----------
        image_relpath : str
            Relative path to parse.

        Returns
        -------
        tuple of (str, int)
            A tuple of (class_id, actual_index).
        """
        assert self != _Split.TEST
        dirname, filename = os.path.split(image_relpath)
        class_id = os.path.split(dirname)[-1]
        basename, _ = os.path.splitext(filename)
        actual_index = int(basename.split("_")[-1])
        return class_id, actual_index


class ImageNet(MinervaDataset):
    """
    ImageNet (ILSVRC 2012) dataset implemented as a Minerva `MinervaDataset`.

    Samples are dicts with the keys ``"image"`` (a PIL image) and ``"label"``
    (an int, or None for the test split).

    Parameters
    ----------
    split : {"train", "test", "val"}
        Dataset split to load.
    root : str
        Path to the root directory containing the ImageNet images.
    extra : str
        Path to the directory containing extra metadata `.npy` entries files.
    transform : callable, optional
        Torchvision-style transform applied to the ``"image"`` entry (PIL
        image in, anything out). Wrapped with ``ExternalTransform``.
    target_transform : callable, optional
        Torchvision-style transform applied to the ``"label"`` entry.
    transforms : _Transform, optional
        A dict-in, dict-out transform applied to the whole sample, after
        ``transform`` and ``target_transform``.
    """

    Split = Union[_Split]

    def __init__(
        self,
        split: Literal["train", "test", "val"],
        root: str,
        extra: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[_Transform] = None,
    ) -> None:
        split = ImageNet.Split(split)
        steps = []
        if transform is not None:
            steps.append(ExternalTransform(["image"], transform))
        if target_transform is not None:
            steps.append(ExternalTransform(["label"], target_transform))
        if transforms is not None:
            steps.append(transforms)
        super().__init__(
            readers=OrderedDict(
                image=_ImageNetDataReader(root, extra, split),
                label=_ImageNetTargetReader(root, extra, split),
            ),
            transforms=_TransformsPipeline(steps) if steps else None,
        )
        self.root = root
        self.extra = extra
        self.split = split


class _ImageNetBaseReader(_Reader):
    """
    Base reader for ImageNet metadata and file indexing.

    Parameters
    ----------
    root : str
        Path to the root directory of the ImageNet images.
    extra : str
        Path to the directory containing extra `.npy` metadata entries.
    split : ImageNet.Split
        Dataset split.
    """

    def __init__(self, root: str, extra: str, split: "ImageNet.Split"):
        self.root = root
        self._extra_root = extra
        self._split = split

        self._entries = None
        self._class_ids = None
        self._class_names = None

    @property
    def split(self) -> "ImageNet.Split":
        """Return the dataset split."""
        return self._split

    def _get_extra_full_path(self, extra_path: str) -> str:
        return os.path.join(self._extra_root, extra_path)

    def _load_extra(self, extra_path: str) -> np.ndarray:
        extra_full_path = self._get_extra_full_path(extra_path)
        return np.load(extra_full_path, mmap_mode="r")

    @property
    def _entries_path(self) -> str:
        return f"entries-{self._split.value.upper()}.npy"

    def _get_entries(self) -> np.ndarray:
        if self._entries is None:
            self._entries = self._load_extra(self._entries_path)
        assert self._entries is not None
        return self._entries

    def get_class_id(self, index: int) -> Optional[str]:
        """
        Get WordNet class ID for a sample by index.

        Parameters
        ----------
        index : int
            Sample index.

        Returns
        -------
        str or None
            WordNet class ID, or None for the test split.
        """
        entries = self._get_entries()
        class_id = entries[index]["class_id"]
        return None if self.split == _Split.TEST else str(class_id)

    def __len__(self) -> int:
        entries = self._get_entries()
        assert len(entries) == self.split.length
        return len(entries)


class _ImageNetDataReader(_ImageNetBaseReader):
    """Reader for loading RGB PIL images from ImageNet disk storage."""

    def __getitem__(self, index: int) -> Image.Image:
        """
        Load and return the image at the specified index.

        Parameters
        ----------
        index : int
            Index of the sample to read.

        Returns
        -------
        PIL.Image.Image
            RGB PIL Image.
        """
        try:
            entries = self._get_entries()
            actual_index = entries[index]["actual_index"]

            class_id = self.get_class_id(index)

            image_relpath = self.split.get_image_relpath(actual_index, class_id)
            image_full_path = os.path.join(self.root, image_relpath)
            with open(image_full_path, mode="rb") as f:
                image_data = f.read()
            f = BytesIO(image_data)
            image = Image.open(f).convert(mode="RGB")
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e
        return image


class _ImageNetTargetReader(_ImageNetBaseReader):
    """Reader for loading target class integer indices from ImageNet entries."""

    def __getitem__(self, index: int) -> Optional[int]:
        """
        Load and return the integer class target at the specified index.

        Parameters
        ----------
        index : int
            Index of the sample.

        Returns
        -------
        int or None
            Integer class target, or None for the test split.
        """
        entries = self._get_entries()
        class_index = entries[index]["class_index"]
        return None if self.split == _Split.TEST else int(class_index)
