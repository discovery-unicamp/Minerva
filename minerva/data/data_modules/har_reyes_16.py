import os
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Tuple, Union
import lightning as L
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch


class ReyesDataset(Dataset):
    def __init__(self, path):
        dataset = pd.read_csv(path, header=None)
        self.X, self.Y = self.convert(dataset)
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.len
    
    def convert(self, dataset, ncanais = 9, tamanho = 128):
        dataset = np.asarray(dataset)
        X = torch.tensor(dataset[:, :tamanho*ncanais],dtype=torch.float64)
        Y = torch.tensor(dataset[:,tamanho*ncanais],dtype=torch.int32)
        X = X.reshape(X.shape[0], ncanais, -1)
        return X,Y


class ReyesModule(L.LightningDataModule):
    def __init__(
        self,
        # General DataModule parameters
        root_data_dir: Union[Path, str],
        # DataLoader parameters
        batch_size: int = 42,
        percentage: float = 1,
    ):
        super().__init__()
        self.root_data_dir = Path(root_data_dir)
        self.batch_size = batch_size
        self.csv_files = {
            "train": os.path.join(self.root_data_dir, "train.csv"),
            "validation": os.path.join(self.root_data_dir, "train.csv"),
            "test": os.path.join(self.root_data_dir, "test.csv"),
        }
        self.percentage = percentage
        self.setup()

    def setup(self, stage: str = None) -> None:
        # Verify that the data is available. If not, fectch and unzip dataset
        for k, v in self.csv_files.items():
            if not os.path.exists(v):
                print(v, "file is missing")
                raise FileNotFoundError


    def _get_dataset_dataloader(
        self, path: Path, shuffle: bool
    ) -> DataLoader[ReyesDataset]:
        dataset = ReyesDataset(path)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=2,
            drop_last=True,
        )
        return dataloader

    def train_dataloader(self):
        dataloader = self._get_dataset_dataloader(
            self.root_data_dir / "train.csv", shuffle=True
        )
        return dataloader

    def val_dataloader(self):
        dataloader = self._get_dataset_dataloader(
            self.root_data_dir / "train.csv", shuffle=False
        )
        return dataloader

    def test_dataloader(self):
        dataloader = self._get_dataset_dataloader(
            self.root_data_dir / "test.csv", shuffle=False
        )
        return dataloader