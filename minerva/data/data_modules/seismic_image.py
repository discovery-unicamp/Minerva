from lightning import LightningDataModule
from torch.utils.data import DataLoader, ConcatDataset, Subset
from minerva.data.datasets.seismic_image import SeismicImageDataset
from minerva.utils.typing import PathLike
from typing import Sequence, Union, Optional, Tuple
import torch


class SeismicImageDataModule(LightningDataModule):

    def __init__(
        self,
        root_dirs: Union[PathLike, Sequence[PathLike]],
        batch_size: int,
        resize: Optional[Tuple[int, int]] = None,
        labels: bool = True,
        drop_last: bool = False,
        cap: Optional[float] = None,
        seed: int = 0,
    ):
        """
        Seismic segmentation dataset in the form of tiff files, optionally annotated
        with single-channel pngs.

        Parameters
        ----------
        root_dirs: Union[PathLike, Sequence[PathLike]]
            Root directory or list of root directories where the dataset files are
            located. Each root directory's structure must be
            ```
            root_dir
            ├── images
            │   ├── train
            │   │   └── file_0.tiff
            │   ├── val
            │   │   └── file_1.tiff
            │   └── test
            │       └── file_2.tiff
            └── annotations
                ├── train
                │   └── file_0.png
                ├── val
                │   └── file_1.png
                └── test
                    └── file_2.png
            ```
            where the annotation directory is optional, but must be consistent across
            directories.

        batch_size: int
            The batch size for the dataloaders

        resize: Tuple[int, int], optional
            A shape to which to resize the images after reading them. If the dataset
            contains images of different shapes (e.g. inlines and crosslines) this is
            mandatory. If left as `None`, no resizing takes place. Defaults to `None`

        labels: bool
            Whether to return the segmentation annotation along with the seismic image.
            Must be `False` if the dataset does not contain annotations. Defaults to
            `True`

        drop_last: bool
            Whether to drop the last incomplete batch. Defaults to `False`

        cap: float, optional
            A value between 0 and 1 signifying the percentage of the training dataset to
            use. If set, the apropriate percentage will be randomly selected according to
            the seed, rounded down. If not set, the whole training dataset is used.
            Defaults to `None`

        seed: int
            The seed for the random selection as determined by the parameter `cap`. If
            cap is not set, this parameter does nothing. Defaults to `0`
        """
        super().__init__()

        self.batch_size = batch_size
        self.drop_last = drop_last

        assert (
            (cap is None) or (0 <= cap <= 1)
        ), f"`cap` must be None or a value in the interval [0, 1], but received {cap=}"

        if not isinstance(root_dirs, (tuple, list)):
            root_dirs = [root_dirs]

        self.train_set = ConcatDataset(
            [SeismicImageDataset(dir, "train", resize, labels) for dir in root_dirs]
        )
        
        if cap is not None:
            random_indices = torch.randperm(
                len(self.train_set), generator=torch.Generator().manual_seed(seed)
            )
            random_indices = random_indices[:int(cap * len(self.train_set))]          
            self.train_set = Subset(self.train_set, random_indices)

        self.val_set = ConcatDataset(
            [SeismicImageDataset(dir, "val", resize, labels) for dir in root_dirs]
        )

        self.test_set = ConcatDataset(
            [SeismicImageDataset(dir, "test", resize, labels) for dir in root_dirs]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            self.batch_size,
            True,
            drop_last=self.drop_last,
            num_workers=11,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            self.batch_size,
            False,
            drop_last=self.drop_last,
            num_workers=11,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            self.batch_size,
            False,
            num_workers=11,
        )
