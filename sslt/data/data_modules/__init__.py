import os
from typing import Tuple
import lightning as L
from sslt.data.readers.zarr_reader import PatchedZarrReader
from sslt.data.datasets.supervised_dataset import (
    SupervisedReconstructionDataset,
)
from sslt.utils.typing import PathLike
from sslt.transforms.transform import _Transform


class F3AttributeDataset(L.LightningDataModule):
    def __init__(
        self,
        # Paths
        
        data_path: PathLike,
        attribute_path: PathLike,
        # Zarr reader parameters
        data_shape: Tuple[int, ...],
        stride: Tuple[int, ...] = None,
        pad_width: Tuple[Tuple[int, int], ...] = None,
        pad_mode: str = "constant",
        pad_kwargs: dict = None,
        # Dataset parameters
        transforms: _Transform = None,
        validation_split: float = 0.2,
        # Dataloader parameters
        batch_size: int = 32,
        num_workers: int = None,
    ):
        super().__init__()
        self.data_path = data_path
        self.attribute_path = attribute_path
        self.data_shape = data_shape
        self.stride = stride
        self.pad_width = pad_width
        self.pad_mode = pad_mode
        self.pad_kwargs = pad_kwargs        
        self.transforms = transforms
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.num_workers = num_workers or os.cpu_count()
        
    