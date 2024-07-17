import os
from pathlib import Path
import lightning as L
import numpy as np
from torch.utils.data import DataLoader
from minerva.utils.typing import PathLike
from minerva.data.datasets.har_xu_23 import TNCDataset

from typing import List, Tuple, Union
from torch.utils.data import  Dataset
import torch

class HarDataModule(L.LightningDataModule):
    def __init__(
        self,
        processed_data_dir: PathLike,
        batch_size: int = 16,
        mc_sample_size: int = 5,
        epsilon: int = 3,
        adf: bool = True,
        window_size: int = 128,
    ):
        """
        This DataModule handles the loading and preparation of data for training, validation,
        and testing. The data is expected to be stored in 3 .npy files named train_data.npy, val_data.npy, and test_data.npy.

        This .npy files are of shape (n_samples, n_timesteps, n_channels) and are produced at specific window size by 
        another data processing script available in https://github.com/maxxu05/rebar/blob/main/data/process/har_processdata.py

        The Python script performs a series of tasks to facilitate the preprocessing and organization of dataset, processing
        The raw accelerometer and gyroscope data for each participant are, filtering out sequences shorter than a set threshold. 
        The data is then split into training, validation, and test sets, which are saved as NumPy arrays along with corresponding participant names.

        For the dataloader, the .npy files are transposed into the shape (n_samples, n_channels, n_timesteps) and passed to the TNCDataset

        Parameters
        ----------
        processed_data_dir: PathLike
            Path to the directory where the processed .npy files are stored. 
            It must have 3 files, named train_data.npy, val_data.npy, and test_data.npy. 
        batch_size : int, optional
            The batch size to use for the DataLoader. Defaults to 16.
        mc_sample_size : int, optional
            This value determines how many neighboring and non-neighboring windows are used per data sample. Defaults to 5.
        epsilon : int, optional
            This parameter controls the "spread" of neighboring windows. 
        adf : bool, optional
            Flag indicating whether to use ADF (Augmented Dickey-Fuller) testing for finding neighbors. Defaults to True.
        window_size : int, optional
            The size of the windows to be used for each sample in the TNC dataset. Defaults to 128.

        """        
        super().__init__()
        self.processed_data_dir = processed_data_dir
        self.batch_size = batch_size
        self.mc_sample_size = mc_sample_size
        self.epsilon = epsilon
        self.adf = adf
        self.window_size = window_size

        self.setup()

    def setup(self):
        """
        Loads the training, validation, and test datasets from the processed .npy files.
        """
        processedharpath = self.processed_data_dir

        self.har_train = np.load(os.path.join(processedharpath, "train_data.npy"))
        self.har_val = np.load(os.path.join(processedharpath, "val_data.npy"))
        self.har_test = np.load(os.path.join(processedharpath, "test_data.npy"))

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
        )

#TODO move to dataset, 
#TODO replace path for Pathlike and review docs
class HarDataset(Dataset):
    def __init__(
        self,
        data_path: Union[Path, str],
        annotate: str,
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
    ):
        """
        Dataset class for loading and preparing human activity recognition data.

        Parameters
        ----------
        data_path : Union[Path, str]
            Path to the directory containing the dataset files.
        annotate : str
            Annotation type for the dataset (e.g., 'train', 'val', 'test').
        feature_column_prefixes : List[str], optional
            Prefixes for the feature columns in the dataset. Defaults to accelerometer and gyroscope data.
        target_column : str, optional
            Column name for the target variable. Defaults to 'standard activity code'.
        flatten : bool, optional
            Whether to flatten the input data. Defaults to False.
        """
        super().__init__()
        self.data_path = Path(data_path)
        self.annotate = annotate
        self.feature_column_prefixes = feature_column_prefixes
        self.target_column = target_column
        self.flatten = flatten

        # Load data
        self.data = np.load(self.data_path / f"{self.annotate}_data_subseq.npy")
        self.labels = np.load(self.data_path / f"{self.annotate}_labels_subseq.npy")
        assert len(self.data) == len(self.labels), "Data and labels must have the same length"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        Tuple[torch.Tensor, int]
            Tuple containing the features and the target label.
        """
        data = self.data[idx]
        if self.flatten:
            data = data.flatten()

        features = data
        target = self.labels[idx]

        # Convert to torch.FloatTensor and torch.LongTensor
        features = torch.FloatTensor(features)
        target = torch.tensor(target, dtype=torch.long)

        return features, target

class HarDataModule_Downstream(L.LightningDataModule):
    def __init__(
        self,
        root_data_dir: Union[Path, str],
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
    ):
        """
        DataModule for downstream tasks in human activity recognition.

        Parameters
        ----------
        root_data_dir : Union[Path, str]
            Root directory containing the dataset files.
        feature_column_prefixes : List[str], optional
            Prefixes for the feature columns in the dataset. Defaults to accelerometer and gyroscope data.
        target_column : str, optional
            Column name for the target variable. Defaults to 'standard activity code'.
        flatten : bool, optional
            Whether to flatten the input data. Defaults to False.
        batch_size : int, optional
            Batch size for the DataLoader. Defaults to 16.
        """
        super().__init__()
        self.root_data_dir = Path(root_data_dir)
        self.feature_column_prefixes = feature_column_prefixes
        self.target_column = target_column
        self.flatten = flatten
        self.batch_size = batch_size

    def setup(self) -> None:
        """
        Setup method for the DataModule. No setup needed for loading .npy files.
        """
        pass

    def _get_dataset_dataloader(self, annotate: str, shuffle: bool) -> DataLoader[HarDataset]:
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
            num_workers=1,
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
        return self._get_dataset_dataloader("train", shuffle=True)

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



