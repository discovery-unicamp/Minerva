from typing import Callable, List, Tuple
import lightning as L
import numpy as np
from sslt.data.datasets.supervised_dataset import (
    SupervisedReconstructionDataset,
)
from sslt.data.readers.zarr_reader import PatchedZarrSubArrayReader
from torch.utils.data import DataLoader
from sslt.utils.typing import PathLike


def splitter(
    start_percent: float, end_percent: float
) -> Callable[[List[int]], List[int]]:
    return lambda x: x[int(len(x) * start_percent) : int(len(x) * end_percent)]


class F3AttributeDataModule(L.LightningDataModule):
    def __init__(
        self,
        original_path: PathLike,
        attribute_path: str,
        data_shape: Tuple[int, int, int],
        stride: Tuple[int, int, int] = None,
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        super().__init__()
        self.original_path = original_path
        self.attribute_path = attribute_path
        self.data_shape = data_shape
        self.stride = stride
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_split = splitter(0.0, 0.8)
        self.val_split = splitter(0.8, 0.9)
        self.test_split = splitter(0.9, 1.0)

    def _get_split(self, split: Callable[[List[int]], List[int]]):
        input_reader = PatchedZarrSubArrayReader(
            path=self.original_path,
            data_shape=self.data_shape,
            stride=self.stride,
            per_axis_indices={0: split, 1: None, 2: None},
        )
        target_reader = PatchedZarrSubArrayReader(
            path=self.attribute_path,
            data_shape=self.data_shape,
            stride=self.stride,
            per_axis_indices={0: split, 1: None, 2: None},
        )
        return SupervisedReconstructionDataset(
            [input_reader, target_reader],
            transforms=lambda x: x.astype(np.float32),
        )

    def train_dataloader(self):
        dataset = self._get_split(self.train_split)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        dataset = self._get_split(self.val_split)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        dataset = self._get_split(self.test_split)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
