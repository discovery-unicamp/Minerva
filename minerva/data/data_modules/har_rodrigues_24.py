from torch.utils.data import DataLoader
from lightning import LightningDataModule
from minerva.data.datasets.har_rodrigues_24 import HARDatasetCPC
from minerva.utils.typing import PathLike

# Defining the data loader for the implementation
class HARDataModuleCPC(LightningDataModule):
    def __init__(
        self,
        root_dir: PathLike,
        data_file="RealWorld_raw",
        input_size=6,
        window=60,
        overlap=30,
        batch_size=64,
        drop_last=False,
    ):
        """Data module for Human Activity Recognition (HAR) using CPC.

        This class handles the creation of training, validation, and test 
        dataloaders for the HAR dataset. It uses the HARDatasetCPC class to 
        load the data.

        Parameters
        ----------
        root_dir : str
            The root directory where the dataset is stored.
        data_file : str, optional
            The name of the data file (default is "RealWorld_raw").
        input_size : int, optional
            The number of input features (default is 6).
        window : int, optional
            The size of the sliding window (default is 60).
        overlap : int, optional
            The overlap size for the sliding window (default is 30).
        batch_size : int, optional
            The batch size for the dataloaders (default is 64).
        drop_last : bool, optional
            Whether to drop the last incomplete batch (default is False).
        """
        super().__init__()
        self.batch_size = batch_size
        self.train_dataset = HARDatasetCPC(
            root_dir, data_file, input_size, window, overlap, phase="train"
        )
        self.val_dataset = HARDatasetCPC(
            root_dir, data_file, input_size, window, overlap, phase="val"
        )
        self.test_dataset = HARDatasetCPC(
            root_dir, data_file, input_size, window, overlap, phase="test"
        )
        self.drop_last = drop_last

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last = self.drop_last, num_workers=11
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, drop_last = self.drop_last, num_workers=11
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last = self.drop_last, num_workers=11
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=11
        )