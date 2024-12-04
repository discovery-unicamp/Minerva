import os
from pathlib import Path
import lightning as L
import numpy as np
from torch.utils.data import DataLoader
from minerva.utils.typing import PathLike
from minerva.data.datasets.har_xu_23 import TNCDataset, HarDataset
from typing import List


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
        """
        super().__init__()
        self.processed_data_dir = Path(processed_data_dir)
        self.batch_size = batch_size
        self.mc_sample_size = mc_sample_size
        self.epsilon = epsilon
        self.adf = adf
        self.window_size = window_size
        self.num_workers = num_workers

        self.har_train = np.load(self.processed_data_dir / "train_data.npy")
        if use_train_as_val:
            self.har_val = self.har_train
        else:
            self.har_val = np.load(self.processed_data_dir / "val_data.npy")
        self.har_test = np.load(self.processed_data_dir / "test_data.npy")

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


class HarDataModule_Downstream(L.LightningDataModule):
    def __init__(
        self,
        root_data_dir: PathLike,
        feature_column_prefixes: List[str] = [
            "accel-x",
            "accel-y",
            "accel-z",
            "gyro-x",
            "gyro-y",
            "gyro-z",
        ],
        target_column: str = "standard activity code",
        flatten: bool = False,
        batch_size: int = 16,
        num_workers: int = 8,
    ):
        """
        DataModule for downstream tasks in human activity recognition (HAR) using the UCI dataset.

        This module handles loading and batching of data for training, validation, and testing.
        It relies on the `HarDataset` class to load the dataset from `.npy` files.

        Parameters
        ----------
        root_data_dir : PathLike
            Directory containing the dataset files. The directory should have the following files:
            - train_data_subseq.npy
            - train_labels_subseq.npy
            - val_data.npy
            - val_labels_subseq.npy
            - test_data.npy
            - test_labels_subseq.npy

            These files should contain subsequences of data (e.g., 128 samples per subsequence) and their corresponding labels.
        feature_column_prefixes : List[str], optional
            List of prefixes for feature columns. Defaults to accelerometer and gyroscope data prefixes:
            ["accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"].
        target_column : str, optional
            Name of the column for the target variable. Defaults to 'standard activity code'.
        flatten : bool, optional
            If True, flattens the input data. Defaults to False.
        batch_size : int, optional
            Number of samples per batch. Defaults to 16.

        Example method
        -------
        train_dataloader() -> DataLoader
            Returns the DataLoader for the training dataset.
            The shape of each batch is:
            - Features: [batch_size, num_timesteps, num_features]
            - Labels: [batch_size]
        """
        super().__init__()
        self.root_data_dir = root_data_dir
        self.feature_column_prefixes = feature_column_prefixes
        self.target_column = target_column
        self.flatten = flatten
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _get_dataset_dataloader(
        self, annotate: str, shuffle: bool
    ) -> DataLoader[HarDataset]:
        """
        Get DataLoader for a specific annotation type.

        Parameters
        ----------
        annotate : str
            Annotation type for the dataset (e.g., 'train', 'val', 'test').
        shuffle : bool
            Whether to shuffle the data.

        Returns
        -------
        DataLoader[HarDataset]
            DataLoader for the specified dataset.
        """
        dataset = HarDataset(
            self.root_data_dir,
            annotate,
            feature_column_prefixes=self.feature_column_prefixes,
            target_column=self.target_column,
            flatten=self.flatten,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader[HarDataset]:
        """
        Returns the DataLoader for the training dataset.

        Returns
        -------
        DataLoader[HarDataset]
            DataLoader for the training dataset.
        """
        return self._get_dataset_dataloader("train", shuffle=False)

    def val_dataloader(self) -> DataLoader[HarDataset]:
        """
        Returns the DataLoader for the validation dataset.

        Returns
        -------
        DataLoader[HarDataset]
            DataLoader for the validation dataset.
        """
        return self._get_dataset_dataloader("val", shuffle=False)

    def test_dataloader(self) -> DataLoader[HarDataset]:
        """
        Returns the DataLoader for the test dataset.

        Returns
        -------
        DataLoader[HarDataset]
            DataLoader for the test dataset.
        """
        return self._get_dataset_dataloader("test", shuffle=False)

    def predict_dataloader(self) -> DataLoader[HarDataset]:
        """
        Returns the DataLoader for the test dataset.

        Returns
        -------
        DataLoader[HarDataset]
            DataLoader for the test dataset.
        """
        # Reuse the test_dataloader logic for predict
        return self.test_dataloader()
