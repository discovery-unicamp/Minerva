from lightning import LightningDataModule
from torch.utils.data import DataLoader, ConcatDataset
from minerva.data.datasets import MultiViewDataset
from minerva.data.readers import TiffReader
from minerva.utils.typing import PathLike
from minerva.transforms import _Transform
from typing import Sequence, Union


class SeismicImageMultiview(LightningDataModule):

    def __init__(
        self,
        root_dirs: Union[PathLike, Sequence[PathLike]],
        transform_pipelines: Sequence[_Transform],
        batch_size: int,
    ):
        """
        LightningDataModule for loading multi-view seismic image datasets with specified
        transformation pipelines.

        Parameters
        ----------
        root_dirs : Union[PathLike, Sequence[PathLike]]
            Path or list of paths to the root directories containing the seismic images.
            Each root directory should have an `images/train` and `images/val`
            subdirectory with the respective dataset split.
            
        transform_pipelines : Sequence[\_Transform]
            A list of transformations to apply to each sample. Each entry in this
            list generates a distinct view of the sample, and the resulting views are
            stacked into an array, allowing for contrastive or other multi-view learning
            applications.
            
        batch_size : int
            The number of samples per batch for training and validation.
        """
        super().__init__()

        self.batch_size = batch_size

        if not isinstance(root_dirs, (tuple, list)):
            root_dirs = [root_dirs]

        self.train_set = [
            MultiViewDataset(TiffReader(dir / "images" / "train"), transform_pipelines)
            for dir in root_dirs
        ]
        self.train_set = ConcatDataset(self.train_set)

        self.val_set = [
            MultiViewDataset(TiffReader(dir / "images" / "val"), transform_pipelines)
            for dir in root_dirs
        ]
        self.val_set = ConcatDataset(self.val_set)


    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            self.batch_size,
            True,
            drop_last=True,
            num_workers=11,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            self.batch_size,
            True,
            drop_last=True,
            num_workers=11,
        )