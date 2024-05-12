import os
from pathlib import Path
from typing import List, Tuple
from minerva.data.readers.zarr_reader import PatchedZarrReader
from minerva.data.datasets.supervised_dataset import (
    SupervisedReconstructionDataset,
)
import lightning as L
from minerva.utils.typing import PathLike
from minerva.transforms.transform import _Transform
from torch.utils.data import DataLoader


class F3ReconstructionDataModule(L.LightningDataModule):
    def __init__(
        self,
        # SupervisedReconstructionDataset args
        input_path: PathLike,
        target_path: PathLike,
        data_shape: Tuple[int, ...] = (1, 500, 500),
        stride: Tuple[int, ...] = None,
        input_transform: _Transform = None,
        target_transform: _Transform = None,
        # DataLoader args
        batch_size: int = 1,
        num_workers: int = None,
    ):
        """Create a data module for the F3 reconstruction task, such as seismic
        attribute regression. This  data module assumes that the input data and
        the target data are stored in Zarr arrays. These arrays are volumetric
        data, and thisdata module will generate patches from the input and
        target data and provide them to the model during training.

        Parameters
        ----------
        input_path : PathLike
            Location of the input data Zarr array (e.g., seismic data)
        target_path : PathLike
            Location of the target data Zarr array (e.g., seismic attribute)
        data_shape : Tuple[int, ...]
            Shape of the patches to be extracted from the data arrays. Usually,
            this is a 3D shape (channels, height, width), by default
            (1, 500, 500)
        stride : Tuple[int, ...], optional
            The stride between consecutive patches. If `None`, the stide will
            be the same as `data_shape`. By default None
        input_transform : _Transform, optional
            Transform to be applied to the input data, by default None
        target_transforms : _Transform, optional
            Transforms to be applied to the target data, by default None
        batch_size : int, optional
            The batch size to be used in the DataLoader, by default 1
        num_workers : int, optional
            The number of workers to be used in the DataLoader, by default None
        """
        super().__init__()

        # SupervisedReconstructionDataset args
        self.input_path = input_path
        self.target_path = target_path
        self.data_shape = data_shape
        self.stride = stride
        self.input_transform = input_transform
        self.target_transform = target_transform

        # DataLoader args
        self.batch_size = batch_size
        self.num_workers = (
            num_workers if num_workers is not None else os.cpu_count()
        )

        # Private attributes
        self._dataset = None

    def _get_dataset(
        self,
        input_path,
        target_path,
        data_shape,
        stride,
        input_transform,
        target_transform,
    ):
        data_reader = PatchedZarrReader(
            path=input_path,
            data_shape=data_shape,
            stride=stride,
        )
        target_reader = PatchedZarrReader(
            path=target_path,
            data_shape=data_shape,
            stride=stride,
        )
        return SupervisedReconstructionDataset(
            readers=[data_reader, target_reader],
            transforms=[input_transform, target_transform],
        )

    def setup(self, stage: str):
        # TODO Here we should add balancing methods and to split the data

        # For now, we are using the same dataset for train, test, and predict
        if stage == "fit" or stage == "test" or stage == "predict":
            self._dataset = self._get_dataset(
                input_path=self.input_path,
                target_path=self.target_path,
                data_shape=self.data_shape,
                stride=self.stride,
                input_transform=self.input_transform,
                target_transform=self.target_transform,
            )

        else:
            raise ValueError(f"Stage {stage} is not valid")

    def train_dataloader(self):
        return DataLoader(
            self._dataset, batch_size=self.batch_size, shuffle=True
        )

    def test_data_loader(self):
        return DataLoader(
            self._dataset, batch_size=self.batch_size, shuffle=False
        )

    def predict_dataloader(self):
        return DataLoader(
            self._dataset, batch_size=self.batch_size, shuffle=False
        )
