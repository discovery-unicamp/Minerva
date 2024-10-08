from lightning import LightningDataModule
from torch.utils.data import DataLoader, ConcatDataset
from minerva.data.datasets.seismic_image import SeismicImageDataset
from minerva.utils.typing import PathLike
from typing import Sequence, Union, Optional, Tuple


class SeismicImageDataModule(LightningDataModule):

    def __init__(
        self,
        root_dirs: Union[PathLike, Sequence[PathLike]],
        batch_size: int,
        resize: Optional[Tuple[int, int]] = None,
        labels: bool = True,
        drop_last: bool = False,
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
        """
        super().__init__()

        self.batch_size = batch_size
        self.drop_last = drop_last

        if not isinstance(root_dirs, (tuple, list)):
            root_dirs = [root_dirs]

        self.train_set = ConcatDataset(
            [SeismicImageDataset(dir, "train", resize, labels) for dir in root_dirs]
        )

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
