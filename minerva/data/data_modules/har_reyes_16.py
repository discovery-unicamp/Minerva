import os
from pathlib import Path
from typing import Union
import lightning as L
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset, Subset
import torch
from minerva.utils.typing import PathLike
import random


class ReyesDataset(Dataset):
    """
    A dataset loader for data of UCI-HAR (https://doi.org/10.24432/C54S4K) used in the paper:
    "An Analysis of Time-Frequency Consistency in Human Activity Recognition" by Hecker et al.
    the dataset file is consisted by 9 channels, 3 for x y z of total acceleration, body acceleration
    and body gyroscope; 128 samples for each channel; one label for each sample, that could be 
    walking, walking upstairs, walking downstairs, sitting, standing, and lying. This dataset
    was sampled at 50Hz, so each sample has a duration of 2.56 seconds.
    This dataset class loads a csv file with no header, where each row is a sample. The first
    128 columns are the time_steps of the first channel, the next 128 columns are the time_steps of the
    second channel, and so on. The last column is the label of the sample, totalizing 1153 columns.
    The label is a float number from 0.0 to 5.0, representing the activity, by the order mentioned.
    The ReyesDataset class inherits from torch.utils.data.Dataset.
    
    """
    def __init__(self, path: PathLike):
        """
        Builder of the ReyesDataset class.
        
        Parameters
        ----------
        path : PathLike
            The path to the csv file with the desired dataset
        
        """
        dataset = pd.read_csv(path, header=None)
        self.X, self.Y = self.convert(dataset)
        self.len = self.X.shape[0]

    def __getitem__(self, index: int):
        """
        Get a sample from the dataset by its index.

        Parameters
        ----------
        index : int
            The index of the desired sample

        Returns
        -------
        tuple
            A tuple with the sample and its label. The sample is a torch tensor with
            shape (9, 128) or (channels, time_steps). The label is a integer from 0 to 5.
        
        """
        return self.X[index], self.Y[index]

    def __len__(self):
        """
        Get the length of the dataset.

        Returns
        -------
        int
            The number of samples in the dataset.
        
        """
        return self.len
    
    def convert(self, dataset: pd.DataFrame, ncanais: int = 9, tamanho: int = 128): # dataset is a pandas dataframe
        """
        Convert the dataset from a pandas dataframe to a torch tensor.
        
        Parameters
        ----------
        dataset : pd.DataFrame
            The dataset to be converted
        ncanais : int
            The number of channels in the dataset
        tamanho : int
            The number of time_steps in each channel
        
        Returns
        -------
        tuple
            A tuple with the converted dataset. The first element is a torch tensor with
            shape (n_samples, n_channels, n_time_steps) with type float64. The second element is a torch tensor
            with shape (n_samples,) with type integer.
        
        """
        dataset = np.asarray(dataset)
        X = torch.tensor(dataset[:, :tamanho*ncanais],dtype=torch.float64)
        Y = torch.tensor(dataset[:,tamanho*ncanais],dtype=torch.long)
        
        X = X.reshape(X.shape[0], ncanais, -1)
        return X,Y


class ReyesModule(L.LightningDataModule):
    """
    A datamodule for the UCI-HAR dataset (https://doi.org/10.24432/C54S4K) used in the paper:
    "An Analysis of Time-Frequency Consistency in Human Activity Recognition" by Hecker et al.
    the datamodule is consisted by a train, validation and test dataloaders, each one with a default batch size
    of 42. The dataset files are consisted by 9 channels, 3 for x y z of total acceleration, body acceleration and
    body gyroscope; 128 samples for each channel; one label for each sample, that could be walking, walking upstairs,
    walking downstairs, sitting, standing, and lying.
    The train dataset is loaded from the file train.csv, the validation dataset is loaded from the file train.csv,
    once there is no validation dataset on the original work repository, and the test dataset is loaded from the
    file test.csv. This datamodule class inherits from lightning.LightningDataModule. It is possible to set the
    percentage of the train and validation datasets to be used.
    
    """

    def __init__(
        self,
        # General DataModule parameters
        root_data_dir: PathLike,
        # DataLoader parameters
        batch_size: int = 42,
        percentage: float = 1.0,
        num_workers: int = 2,
        seed: int = 42,
        balanced_division: bool = True,
    ):
        """
        Builder of the ReyesModule class.

        Parameters
        ----------
        root_data_dir : PathLike
            The root directory of the dataset files
        batch_size : int
            The batch size of the dataloaders, default is 42
        percentage : float
            The percentage of the dataset to be used, default is 1.0
        num_workers : int
            The number of workers to be used in the dataloaders, default is 2
        seed : int
            The seed to be used in the random functions, default is 42
        balanced_division : bool
            If True and percentage is smaller than 1.0, the dataloader will have all classes with same number of samples (or differ by 1). If False the subset is chosen randomly. Default is True
            There is still a possibliity of some class not being present by the loader, if all samples have been selected as not being part of any batch on the shuffle of data loader.
            Example: perfectly balanced dataset: 0 1 2 3 4 0 1 2 3 4 with batch size 4 and batches selected by dataloader: [0 1 2 3] [0 1 2 3] (all 2 samples of class 4 are not present), or a batch size of 6:
            [0 1 2 3 0 1] (all 2 sample of class 4 are not present). This is a limitation of the dataloader, and not of the datamodule.

        """
        super().__init__()
        self.root_data_dir = Path(root_data_dir)
        self.batch_size = batch_size
        self.csv_files = {
            "train": os.path.join(self.root_data_dir, "train.csv"),
            "validation": os.path.join(self.root_data_dir, "train.csv"),
            "test": os.path.join(self.root_data_dir, "test.csv"),
        }
        self.percentage = percentage
        self.num_workers = num_workers
        self.seed = seed
        self.balanced_division = balanced_division

        # Verify that the data is available. If not, raise an error.
        for k, v in self.csv_files.items():
            if not os.path.exists(v):
                print(v, "file is missing")
                raise FileNotFoundError        


    def _get_dataset_dataloader(
        self, path: Path, shuffle: bool, percentage: float = 1.0
    ) -> DataLoader[ReyesDataset]:
        """
        Get a dataloader from a dataset file, shuffling the samples if shuffle is True
        and setting the percentage of the datasets to be used.
        This function differ from the solution implemented in article to the percentage,
        because this way is more accurate.
        
        Parameters
        ----------
        path : Path
            The path to the dataset file
        shuffle : bool
            If True, the samples will be shuffled
        percentage : float
            The percentage of the dataset to be used

        Returns
        -------
        DataLoader
            A DataLoader with the desired dataset
        
        """
        dataset = ReyesDataset(path)

        # if percentage is set, chose random len*percentage samples and build a subset
        if percentage < 1.0:
            if self.balanced_division:
                indices = []
                amostras = {}
                for i in range(len(dataset)):
                    amostra = dataset[i]
                    if amostra[1].item() not in amostras:
                        amostras[amostra[1].item()] = []
                    amostras[amostra[1].item()].append(i)
                
                random.seed(self.seed)
                if shuffle:
                    for key in amostras:
                        random.shuffle(amostras[key])
                maximo = int(len(dataset) * percentage)
                for i in range(maximo):
                    for key in amostras:
                        if len(amostras[key]) > 0:
                            indices.append(amostras[key].pop())
                        if len(indices) >= maximo:
                            break
                    if len(indices) >= maximo:
                        break
            else:
                indices = list(range(len(dataset)))
                random.seed(self.seed)
                indices = random.sample(
                        indices, int(len(indices) * percentage)
                    )
            dataset = Subset(dataset, indices)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=True,
        )
        return dataloader

    def train_dataloader(self):
        """
        Get the train dataloader by location defined by root_data_dir/train.csv.

        Returns
        -------
        DataLoader
            A DataLoader with the train dataset
        """
        dataloader = self._get_dataset_dataloader(
            self.root_data_dir / "train.csv", shuffle=True, percentage=self.percentage
        )
        return dataloader

    def val_dataloader(self):
        """
        Get the validaton dataloader by location defined by root_data_dir/train.csv. This dataloader uses the same
        dataset as the train dataloader, because there is no validation dataset in the original work repository.

        Returns
        -------
        DataLoader
            A DataLoader with the train dataset
        """
        dataloader = self._get_dataset_dataloader(
            self.root_data_dir / "train.csv", shuffle=False, percentage=self.percentage
        )
        return dataloader

    def test_dataloader(self):
        """

        Get the test dataloader by location defined by root_data_dir/test.csv.
        
        Returns
        -------
        DataLoader
            A DataLoader with the test dataset
        """
        dataloader = self._get_dataset_dataloader(
            self.root_data_dir / "test.csv", shuffle=False
        )
        return dataloader
    
    def predict_dataloader(self):
        """

        Get the predict dataloader by location defined by root_data_dir/test.csv, equivalent to the test dataloader.
        
        Returns
        -------
        DataLoader
            A DataLoader with the test dataset
        """
        dataloader = self._get_dataset_dataloader(
            self.root_data_dir / "test.csv", shuffle=False
        )
        return dataloader