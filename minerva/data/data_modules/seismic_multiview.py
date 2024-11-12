from lightning import LightningDataModule
from torch.utils.data import DataLoader, ConcatDataset
from minerva.data.datasets.seismic_image import SeismicImageDataset
from minerva.data.datasets import MultiViewsDataset
from minerva.utils.typing import PathLike
from minerva.transforms import _Transform
from typing import Sequence, Union


class SeismicMultiviewDataModule(LightningDataModule):

    def __init__(
        self,
        root_dirs: Union[PathLike, Sequence[PathLike]],
        transform_pipeline: _Transform,
        batch_size: int,
        drop_last: bool = False,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.drop_last = drop_last

        if not isinstance(root_dirs, (tuple, list)):
            root_dirs = [root_dirs]

        train_set = ConcatDataset(
            [SeismicImageDataset(dir, "train", labels=False) for dir in root_dirs]
        )

        val_set = ConcatDataset(
            [SeismicImageDataset(dir, "val", labels=False) for dir in root_dirs]
        )
        
        self.train_set = MultiViewsDataset(train_set, transform_pipeline)
        self.val_set = MultiViewsDataset(val_set, transform_pipeline)

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
