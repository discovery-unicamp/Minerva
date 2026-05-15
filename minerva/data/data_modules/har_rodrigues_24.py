from typing import List, Optional, Union
from torch.utils.data import DataLoader
from lightning import LightningDataModule
from minerva.data.datasets.har_rodrigues_24 import HARDatasetCPC
from minerva.utils.typing import PathLike


# Defining the data loader for the implementation
class HARDataModuleCPC(LightningDataModule):
    def __init__(
        self,
        data_path: Union[PathLike, List[PathLike]],
        input_size: int = 6,
        window: int = 60,
        overlap: int = 30,
        batch_size: int = 64,
        use_train_as_val: bool = False,
        use_val_with_train: bool = True,
        columns: Optional[List[str]] = None,
        num_workers: int = 8,
        drop_last: bool = True,
        label: Optional[str] = "standard activity code",
        transpose_data: bool = True,
    ):
        """Data module for Human Activity Recognition (HAR) using CPC.

        This class handles the creation of training, validation, and test
        dataloaders for the HAR dataset. It uses the HARDatasetCPC class to
        load the data.

        Parameters
        ----------
        data_path : Union[PathLike, List[PathLike]]
            The root directory where the dataset is stored. If a list is
            the datasets will be concatenated, in their respective order, to
            each partition (train, val, test).
        input_size : int, optional
            The number of input features (default is 6).
        window : int, optional
            The size of the sliding window (default is 60).
        overlap : int, optional
            The overlap size for the sliding window (default is 30).
        batch_size : int, optional
            The batch size for the dataloaders (default is 64).
        use_val_with_train : bool
            Whether to use the validation set as the training set.
        label : Optional[str]
            The column to be used as the label. If None, no labels will be
            used. If 'return_index_as_label', the index of the data will be
            used as the label.
        transpose_data : bool
            If True, the data will be returned as a vector of shape (C, T),
            else the data will be returned as a vector of shape  (T, C).
        """
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.label = label
        self.transpose_data = transpose_data

        self.train_dataset = HARDatasetCPC(
            data_path,
            input_size,
            window,
            overlap,
            phase="train",
            use_train_as_val=use_train_as_val,
            use_val_with_train=use_val_with_train,
            columns=columns,
            label=label,
            transpose_data=transpose_data,
        )
        self.val_dataset = HARDatasetCPC(
            data_path,
            input_size,
            window,
            overlap,
            phase="val",
            use_train_as_val=use_train_as_val,
            use_val_with_train=use_val_with_train,
            columns=columns,
            label=label,
            transpose_data=transpose_data,
        )
        self.test_dataset = HARDatasetCPC(
            data_path,
            input_size,
            window,
            overlap,
            phase="test",
            use_train_as_val=use_train_as_val,
            use_val_with_train=use_val_with_train,
            columns=columns,
            label=label,
            transpose_data=transpose_data,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
        )

    def __repr__(self):
        return (
            f"HARDataModuleCPC(batch_size={self.batch_size}, datasets={self.data_path})"
        )
