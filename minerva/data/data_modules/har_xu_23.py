import os
from pathlib import Path
from typing import List

import lightning as L
import numpy as np
from torch.utils.data import DataLoader

from minerva.data.datasets.har_xu_23 import HarDataset, TNCDataset
from minerva.utils.typing import PathLike


class HarDataModule(L.LightningDataModule):
    def __init__(
        self,
        processed_data_dir: PathLike,
        batch_size: int = 16,
        mc_sample_size: int = 5,
        epsilon: int = 3,
        adf: bool = True,
        window_size: int = 128,
        use_train_as_val: bool = False,
        num_workers: int = 8,
        use_val_with_train: bool = False,
    ):
        """
        This DataModule handles the loading and preparation of data for
        training, validation, and testing. The data is expected to be stored
        in 3 numpy (.npy) files named `train_data.npy`, `val_data.npy`, and
        `test_data.npy`. They are NumPy arrays storing the concatenated
        accelerometer and gyroscope data.

        This numpy arrays (files) must have the following shape (n_samples,
        n_timesteps, n_channels) and are produced at specific window size by
        another data processing script available in
        https://github.com/maxxu05/rebar/blob/main/data/process/har_processdata.py

        The original files have exact shape of:
        - `train_data.npy`: `(41, 15038, 6)`
        - `val_data.npy`: `(9, 15038, 6)`
        - `test_data.npy`: `(9, 15038, 6)`

        The Python script performs a series of tasks to facilitate the
        preprocessing and organization of dataset, processing
        The raw accelerometer and gyroscope data for each participant are,
        filtering out sequences shorter than a set threshold.
        The data is then split into training, validation, and test sets, which
        are saved as NumPy arrays along with corresponding participant names.

        For the dataloader, the .npy files are transposed into the shape
        (n_samples, n_channels, n_timesteps) and passed to the TNCDataset

        Parameters
        ----------
        processed_data_dir: PathLike
            Path to the directory where the processed .npy files are stored.
            Inside this path must have 3 files, named train_data.npy,
            val_data.npy, and test_data.npy.
        batch_size : int, optional
            The batch size to use for the DataLoader. Defaults to 16.
        mc_sample_size : int, optional
            This value determines how many neighboring and non-neighboring
            windows are used per data sample. Defaults to 5.
        epsilon : int, optional
            This parameter controls the "spread" of neighboring windows.
        adf : bool, optional
            Flag indicating whether to use ADF (Augmented Dickey-Fuller)
            testing for finding neighbors. Defaults to True.
        window_size : int, optional
            The size of the windows to be used for each sample in the TNC
            dataset. Defaults to 128.
        use_val_with_train : bool, optional
            If True, the validation and train sets will be concatenated in
            order to create a large train set. By default, this is True.
        """
        super().__init__()
        self.processed_data_dir = Path(processed_data_dir)
        self.batch_size = batch_size
        self.mc_sample_size = mc_sample_size
        self.epsilon = epsilon
        self.adf = adf
        self.window_size = window_size
        self.num_workers = num_workers
        self.use_val_with_train = use_val_with_train

        self.har_train = np.load(self.processed_data_dir / "train_data.npy")
        self.har_val = np.load(self.processed_data_dir / "val_data.npy")
        self.har_test = np.load(self.processed_data_dir / "test_data.npy")

        # Handle use_val_with_train and use_train_as_val
        if use_train_as_val:
            self.har_val = self.har_train
        elif use_val_with_train:
            self.har_train = np.concatenate([self.har_train, self.har_val], axis=0)

        # Print dataset sizes after concatenation
        # print(f"\nFinal Training Data Size: {self.har_train.shape}")
        # print(f"Final Validation Data Size: {self.har_val.shape}")
        # print(f"Final Test Data Size: {self.har_test.shape}")

    def train_dataloader(self):
        """
        Returns the DataLoader for the training dataset.

        Returns
        -------
        DataLoader
            DataLoader for the training dataset.
        """
        return DataLoader(
            TNCDataset(
                np.transpose(self.har_train, (0, 2, 1)),
                self.mc_sample_size,
                self.window_size,
                self.epsilon,
                self.adf,
            ),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """
        Returns the DataLoader for the validation dataset.

        Returns
        -------
        DataLoader
            DataLoader for the validation dataset.
        """
        return DataLoader(
            TNCDataset(
                np.transpose(self.har_val, (0, 2, 1)),
                self.mc_sample_size,
                self.window_size,
                self.epsilon,
                self.adf,
            ),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        """
        Returns the DataLoader for the test dataset.

        Returns
        -------
        DataLoader
            DataLoader for the test dataset.
        """
        return DataLoader(
            TNCDataset(
                np.transpose(self.har_test, (0, 2, 1)),
                self.mc_sample_size,
                self.window_size,
                self.epsilon,
                self.adf,
            ),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

