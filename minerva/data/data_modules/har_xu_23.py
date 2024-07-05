import os
from pathlib import Path
import lightning as L
import numpy as np
from torch.utils.data import DataLoader
import sys
sys.path.append('../')
from datasets import har_xu_23

class HarDataModule(L.LightningDataModule):
    """
    This DataModule handles the loading and preparation of data for training, validation,
    and testing. The data is expected to be stored in .npy files.

    This .npy files are of shape (n_samples, n_timesteps, n_channels) and are produced at specific window size by 
    another data processing script available in https://github.com/maxxu05/rebar/blob/main/data/process/har_processdata.py

    The Python script performs a series of tasks to facilitate the preprocessing and organization of dataset, processing
    The raw accelerometer and gyroscope data for each participant are, filtering out sequences shorter than a set threshold. 
    The data is then split into training, validation, and test sets, which are saved as NumPy arrays along with corresponding participant names.

    For the dataloader, the .npy files are transposed into the shape (n_samples, n_channels, n_timesteps) and passed to the TNCDataset

    Parameters
    ----------
    processed_data_dir : str, optional
        The directory where the processed .npy files are stored. Defaults to "data/har/processed".
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

    def __init__(
        self,
        processed_data_dir: str = "data/har/processed",
        batch_size: int = 16,
        mc_sample_size: int = 5,
        epsilon: int = 3,
        adf: bool = True,
        window_size: int = 128,
    ):
        super().__init__()
        self.processed_data_dir = Path(processed_data_dir)
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
            har_xu_23.TNCDataset(
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
            har_xu_23.TNCDataset(
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
            har_xu_23.TNCDataset(
                np.transpose(self.har_test, (0, 2, 1)),
                self.mc_sample_size,
                self.window_size,
                self.epsilon,
                self.adf,
            ),
            batch_size=self.batch_size,
            shuffle=False,
        )
