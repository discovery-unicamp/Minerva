from minerva.data.datasets.series_dataset import (
    MultiModalSeriesCSVDataset,
    SeriesFolderCSVDataset,
)

from torch.utils.data import DataLoader, ConcatDataset, Subset
from typing import Callable, Dict, Optional, Tuple, Union, List
import random
import lightning as L
from pathlib import Path

import os

from minerva.utils.typing import PathLike
from minerva.samplers.domain_sampler import RandomDomainSampler
from collections import defaultdict


def parse_transforms(
    transforms: Union[List[Callable], Dict[str, List[Callable]]],
) -> Dict[str, List[Callable]]:
    """Parse the transforms parameter to a dictionary with the split name as
    key and a list of transforms as value.

    Parameters
    ----------
    transforms : Union[List[Callable], Dict[str, List[Callable]]]
        This could be:
        - None: No transforms will be applied
        - List[Callable]: A list of transforms that will be applied to the
            data. The same transforms will be applied to all splits.
        - Dict[str, List[Callable]]: A dictionary with the split name as
            key and a list of transforms as value. The split name must be
            one of: "train", "validation", "test" or "predict".

    Returns
    -------
    Dict[str, List[Callable]]
        A dictionary with the split name as key and a list of transforms as
        value.
    """
    if isinstance(transforms, list) or transforms is None:
        return {
            "train": transforms,
            "validation": transforms,
            "test": transforms,
            "predict": transforms,
        }
    elif isinstance(transforms, dict):
        # Check if the keys are valid
        valid_keys = ["train", "validation", "test", "predict"]
        assert all(
            key in valid_keys for key in transforms.keys()
        ), f"Invalid transform key. Must be one of: {valid_keys}"
        new_transforms = {
            "train": None,
            "validation": None,
            "test": None,
            "predict": None,
        }
        new_transforms.update(transforms)
        return new_transforms


def parse_num_workers(num_workers: Optional[int] = None) -> int:
    """Parse the num_workers parameter. If None, use all cores.

    Parameters
    ----------
    num_workers : int
        Number of workers to load data. If None, then use all cores

    Returns
    -------
    int
        Number of workers to load data.
    """
    n = num_workers if num_workers is not None else os.cpu_count()
    return n or 1


class UserActivityFolderDataModule(L.LightningDataModule):
    def __init__(
        self,
        # Dataset Params
        data_path: PathLike,
        features: List[str] = (
            "accel-x",
            "accel-y",
            "accel-z",
            "gyro-x",
            "gyro-y",
            "gyro-z",
        ),  # type: ignore
        label: str = "standard activity code",
        pad: bool = False,
        transforms: Optional[Union[List[Callable], Dict[str, List[Callable]]]] = None,
        cast_to: str = "float32",
        # Loader params
        batch_size: int = 1,
        num_workers: Optional[int] = None,
    ):
        """Define the dataloaders for train, validation and test splits for
        HAR datasets. The data must be in the following folder structure:
        It is a wrapper around ``SeriesFolderCSVDataset`` dataset class.
        The ``SeriesFolderCSVDataset`` class assumes that the data is in a
        folder with multiple CSV files. Each CSV file is a single sample that
        can be composed of multiple time steps (rows). Each column is a feature
        of the sample.

        For instance, if we have two samples, user-1.csv and user-2.csv,
        the directory structure will look something like:

        data_path
        ├── user-1.csv
        └── user-2.csv

        And the data will look something like:
        - user-1.csv:
            +---------+---------+--------+
            | accel-x | accel-y | class  |
            +---------+---------+--------+
            | 0.502123| 0.02123 | 1      |
            | 0.682012| 0.02123 | 1      |
            | 0.498217| 0.00001 | 1      |
            +---------+---------+--------+
        - user-2.csv:
            +---------+---------+--------+
            | accel-x | accel-y | class  |
            +---------+---------+--------+
            | 0.502123| 0.02123 | 0      |
            | 0.682012| 0.02123 | 0      |
            | 0.498217| 0.00001 | 0      |
            | 3.141592| 1.414141| 0      |
            +---------+---------+--------+

        The ``features`` parameter is used to select the columns that will be
        used as features. For instance, if we want to use only the accel-x
        column, we can set ``features=["accel-x"]``. If we want to use both
        accel-x and accel-y, we can set ``features=["accel-x", "accel-y"]``.

        The label column is specified by the ``label`` parameter. Note that we
        have one label per time-step and not a single label per sample.

        The dataset will return a 2-element tuple with the data and the label,
        if the ``label`` parameter is specified, otherwise return only the data.


        Parameters
        ----------
        data_path : PathLike
            The location of the directory with CSV files.
        features: List[str]
            A list with column names that will be used as features. If None,
            all columns except the label will be used as features.
        pad: bool, optional
            If True, the data will be padded to the length of the longest
            sample. Note that padding will be applyied after the transforms,
            and also to the labels if specified.
        label: str, optional
            Specify the name of the column with the label of the data
        transforms : Union[List[Callable], Dict[str, List[Callable]]], optional
            This could be:
            - None: No transforms will be applied
            - List[Callable]: A list of transforms that will be applied to the
                data. The same transforms will be applied to all splits.
            - Dict[str, List[Callable]]: A dictionary with the split name as
                key and a list of transforms as value. The split name must be
                one of: "train", "validation", "test" or "predict".
        cast_to: str, optional
            Cast the numpy data to the specified type
        batch_size : int, optional
            The size of the batch
        num_workers : int, optional
            Number of workers to load data. If None, then use all cores
        """
        super().__init__()

        # ---- Dataset Parameters ----
        # Allowing multiple datasets
        self.data_path = Path(data_path)
        self.features = features
        self.label = label
        self.pad = pad
        self.transforms = parse_transforms(transforms)

        # ---- Loader Parameters ----
        self.batch_size = batch_size
        self.num_workers = parse_num_workers(num_workers)
        self.cast_to = cast_to

        # ---- Class specific ----
        self.datasets = {}

    def _load_dataset(self, split_name: str) -> SeriesFolderCSVDataset:
        """Create a ``SeriesFolderCSVDataset`` dataset with the given split.

        Parameters
        ----------
        split_name : str
            Name of the split (train, validation or test). This will be used to
            load the corresponding CSV file.

        Returns
        -------
        SeriesFolderCSVDataset
            The dataset with the given split.
        """
        assert split_name in [
            "train",
            "validation",
            "test",
            "predict",
        ], f"Invalid split_name: {split_name}"

        if split_name == "predict":
            split_name = "test"

        return SeriesFolderCSVDataset(
            self.data_path / split_name,
            features=self.features,
            label=self.label,
            pad=self.pad,
            transforms=self.transforms[split_name],
            cast_to=self.cast_to,
        )

    def setup(self, stage: str):
        """Assign the datasets to the corresponding split. ``self.datasets``
        will be a dictionary with the split name as key and the dataset as
        value.

        Parameters
        ----------
        stage : str
            The stage of the setup. This could be:
            - "fit": Load the train and validation datasets
            - "test": Load the test dataset
            - "predict": Load the predict dataset

        Raises
        ------
        ValueError
            If the stage is not one of: "fit", "test" or "predict"
        """
        if stage == "fit":
            self.datasets["train"] = self._load_dataset("train")
            self.datasets["validation"] = self._load_dataset("validation")
        elif stage == "test":
            self.datasets["test"] = self._load_dataset("test")
        elif stage == "predict":
            self.datasets["predict"] = self._load_dataset("test")
        else:
            raise ValueError(f"Invalid setup stage: {stage}")

    def _get_loader(self, split_name: str, shuffle: bool) -> DataLoader:
        """Get a dataloader for the given split.

        Parameters
        ----------
        split_name : str
            The name of the split. This must be one of: "train", "validation",
            "test" or "predict".
        shuffle : bool
            Shuffle the data or not.

        Returns
        -------
        DataLoader
            A dataloader for the given split.
        """
        return DataLoader(
            self.datasets[split_name],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=True,
        )

    def train_dataloader(self) -> DataLoader:
        return self._get_loader("train", shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._get_loader("validation", shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._get_loader("test", shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self._get_loader("predict", shuffle=False)

    def __str__(self):
        return f"UserActivityFolderDataModule(data_path={self.data_path}, batch_size={self.batch_size})"

    def __repr__(self) -> str:
        return str(self)


class MultiModalHARSeriesDataModule(L.LightningDataModule):
    def __init__(
        self,
        # Dataset params
        data_path: PathLike | List[PathLike],
        feature_prefixes: List[str] = (
            "accel-x",
            "accel-y",
            "accel-z",
            "gyro-x",
            "gyro-y",
            "gyro-z",
        ),  # type: ignore
        label: str = "standard activity code",
        features_as_channels: bool = True,
        transforms: Optional[Union[List[Callable], Dict[str, List[Callable]]]] = None,
        cast_to: str = "float32",
        # Loader params
        batch_size: int = 1,
        num_workers: Optional[int] = None,
        data_percentage: float = 1.0,
        use_train_as_validation: bool = False,
        use_val_with_train: bool = False,
        map_labels: Optional[Dict[int, int]] = None,
        drop_last: bool = True,
        n_domains_per_sample: Optional[int] = None,
        samples_per_class: Optional[int] = None,
        seed: Optional[int] = None,
        predict_split: str = "test",
        shuffle_train: bool = True,
    ):
        """Define the dataloaders for train, validation and test splits for
        HAR datasets. This datasets assumes that the data is in a single CSV
        file with series of data. Each row is a single sample that can be
        composed of multiple modalities (series). Each column is a feature of
        some series with the prefix indicating the series. The suffix may
        indicates the time step. For instance, if we have two series, accel-x
        and accel-y, the data will look something like:

        +-----------+-----------+-----------+-----------+--------+
        | accel-x-0 | accel-x-1 | accel-y-0 | accel-y-1 |  class |
        +-----------+-----------+-----------+-----------+--------+
        | 0.502123  | 0.02123   | 0.502123  | 0.502123  |  0     |
        | 0.6820123 | 0.02123   | 0.502123  | 0.502123  |  1     |
        | 0.498217  | 0.00001   | 1.414141  | 3.141592  |  2     |
        +-----------+-----------+-----------+-----------+--------+

        The ``feature_prefixes`` parameter is used to select the columns that
        will be used as features. For instance, if we want to use only the
        accel-x series, we can set ``feature_prefixes=["accel-x"]``. If we want
        to use both accel-x and accel-y, we can set
        ``feature_prefixes=["accel-x", "accel-y"]``. If None is passed, all
        columns will be used as features, except the label column.
        The label column is specified by the ``label`` parameter.

        The dataset will return a 2-element tuple with the data and the label,
        if the ``label`` parameter is specified, otherwise return only the data.

        If ``features_as_channels`` is ``True``, the data will be returned as a
        vector of shape `(C, T)`, where C is the number of channels (features)
        and `T` is the number of time steps. Else, the data will be returned as
        a vector of shape  T*C (a single vector with all the features).

        Parameters
        ----------
        data_path : PathLike
            The path to the folder with "train.csv", "validation.csv" and
            "test.csv" files inside it.
        feature_prefixes : Union[str, List[str]], optional
            The prefix of the column names in the dataframe that will be used
            to become features. If None, all columns except the label will be
            used as features.
        label : str, optional
            The name of the column that will be used as label
        features_as_channels : bool, optional
            If True, the data will be returned as a vector of shape (C, T),
            else the data will be returned as a vector of shape  T*C.
        cast_to: str, optional
            Cast the numpy data to the specified type
        transforms : Union[List[Callable], Dict[str, List[Callable]]], optional
            This could be:
            - None: No transforms will be applied
            - List[Callable]: A list of transforms that will be applied to the
                data. The same transforms will be applied to all splits.
            - Dict[str, List[Callable]]: A dictionary with the split name as
                key and a list of transforms as value. The split name must be
                one of: "train", "validation", "test" or "predict".
        batch_size : int, optional
            The size of the batch
        num_workers : int, optional
            Number of workers to load data. If None, then use all cores
        data_percentage : float, optional
            The percentage of the data that will be used. This is useful to
            create a small datasets.
        use_train_as_validation : bool, optional
            If True, the train dataset will be used as validation dataset.
        use_val_with_train: bool, optional
            If True, the validation and train sets will be concatenated in
            order to create a large train set. By default, this is False.
        map_labels : Dict[int, int], optional
            A dictionary to map the labels to a new label. The key is the
            original label and the value is the new label.
        drop_last : bool, optional
            Drop the last batch if it is not complete.
        n_domains_per_sample : int, optional
            This is inly useful when using multiple domains (`data_path`). It
            will allow creating batches with same number of samples from
            multiple domains. If None, it will just use concatenate all
            datasets and sample in a non-stratified way. By default, None-
        samples_per_class : int, optional
            If not None, use this number of samples per class for the train
            split. This will override the data_percentage parameter.
        seed: Optional[int] = None
            Seed for sampling the dataset. If None, no seed is set.
        predict_split: str
            The name of the split to use for prediction. This will be used to
            load the dataset for prediction. By default, this is "test".
        shuffle_train: str
            If True, the train dataset will be shuffled.

        Notes
        -----
        - If `data_percentage` is set to a value less than 1.0, a random subset
            of the dataset will be used, containing approximately the specified
            percentage of the total data. This sampling is not stratified.
        - If `samples_per_class` is specified, the train split will contain an
            equal number of samples for each class, as defined by this
            parameter. This option is mutually exclusive with data_percentage;
            both cannot be used at the same time.
        - The `seed` parameter controls the randomness of sampling: If `seed` is
            set (i.e., an integer), sampling becomes deterministic, ensuring
            the same subset is selected on each run. This improves
            reproducibility and supports cumulative sampling—for example,
            progressively increasing `samples_per_class` will retain consistency
            across runs by sampling the same initial elements. If seed is
            `None`,  sampling is non-deterministic, and different subsets may
            be chosen each time.

        Raises
        ------
        ValueError
            If `samples_per_class` and `data_percentage` are both set.
        """
        super().__init__()
        self.data_path = data_path if isinstance(data_path, list) else [data_path]
        self.data_path = [Path(data) for data in self.data_path]
        self.feature_prefixes = feature_prefixes
        if isinstance(self.feature_prefixes, tuple):
            self.feature_prefixes = list(self.feature_prefixes)
        self.label = label
        self.features_as_channels = features_as_channels
        self.transforms = parse_transforms(transforms)
        self.cast_to = cast_to
        self.batch_size = batch_size
        self.num_workers = parse_num_workers(num_workers)
        self.data_percentage = data_percentage
        self.datasets = {}
        self.use_train_as_validation = use_train_as_validation
        self.use_val_with_train = use_val_with_train
        self.map_labels = map_labels
        self.drop_last = drop_last
        self.n_domains_per_sample = n_domains_per_sample
        self.samples_per_class = samples_per_class
        if self.data_percentage < 1.0 and self.samples_per_class is not None:
            raise ValueError("Cannot use both data_percentage and samples_per_class")

        if self.data_percentage > 1.0 or self.data_percentage < 0.0:
            raise ValueError("data_percentage must be between 0 and 1.0, inclusive.")
        self.seed = seed
        self.rng = random.Random(seed)
        self.predict_split = predict_split
        self.shuffle_train = shuffle_train

    def _sample_dataset(self, dataset):
        """Sample the dataset based on the specified parameters.

        If `samples_per_class` is specified, a subset will be created
        containing the specified number of samples for each class. If
        `data_percentage` is specified, a random subset of the dataset
        will be created containing approximately the specified percentage
        of the total data. If neither is specified, the entire dataset
        will be returned.

        Note
        -----
        The `seed` parameter controls the randomness of sampling:
        If `seed` is set (i.e., an integer), sampling becomes deterministic,
        ensuring the same subset is selected on each run and allowing for
        cumulative sampling (e.g., progressively increasing `samples_per_class`
        will retain consistency across runs by sampling the same initial
        elements). If `seed` is `None`, sampling is non-deterministic, and
        different subsets may be chosen each time.

        Parameters
        ----------
        dataset: Dataset
            A map-like dataset to sample from. This should be a-

        Returns
        -------
        Dataset
            A sampled dataset.

        Raises
        ------
        ValueError
            If `samples_per_class` is specified and a class has fewer
            samples than the specified number.
        """
        if self.samples_per_class is not None:
            if self.samples_per_class <= 0:
                raise ValueError("samples_per_class must be a positive integer.")
            class_indices = defaultdict(list)
            dset_indices = list(range(len(dataset)))
            self.rng.shuffle(dset_indices)  # Shuffle indices to ensure randomness

            # Group indices by class (note that are shuffled before!)
            # Thus, each list in class_indices will contain indices of samples (shuffled)
            for idx in dset_indices:
                label = dataset[idx][1]
                class_indices[label].append(idx)

            sampled_indices = []
            for label, indices in class_indices.items():
                if len(indices) < self.samples_per_class:
                    raise ValueError(
                        f"Class {label} has only {len(indices)} samples, "
                        f"but {self.samples_per_class} were requested."
                    )
                sampled_indices += indices[: self.samples_per_class]
            subset_ = Subset(dataset, sampled_indices)
            x_list=[]
            y_list=[]
            for i in range(len(subset_)):
                x,y = subset_[i]
                x_list.append(x)
                y_list.append(y)
            print(f'\n\ny list is {y_list} \n\n\n')

            return subset_

        elif self.data_percentage < 1.0:
            # Existing percentage-based sampling
            indices = list(range(len(dataset)))
            self.rng.shuffle(indices)
            indices = indices[: int(self.data_percentage * len(dataset))]
            subset_ = Subset(dataset, indices)


            return subset_
        else:
            return dataset

    def _load_dataset(
        self, split_name: str
    ) -> Tuple[Union[MultiModalSeriesCSVDataset, ConcatDataset], List[int]]:
        """Create a ``MultiModalSeriesCSVDataset`` dataset with the given split.

        Parameters
        ----------
        split_name : str
            The name of the split. This must be one of: "train", "validation",
            "test" or "predict".

        Returns
        -------
        MultiModalSeriesCSVDataset
            A MultiModalSeriesCSVDataset dataset with the given split.
        """
        if split_name == "predict":
            split_name = self.predict_split

        assert split_name in [
            "train",
            "validation",
            "test",
            "predict",
        ], f"Invalid split_name: {split_name}"

        datasets = []
        domain_labels = []
        for i, data in enumerate(self.data_path):
            data = Path(data)
            dataset = MultiModalSeriesCSVDataset(
                data / f"{split_name}.csv",
                feature_prefixes=self.feature_prefixes,
                label=self.label,
                features_as_channels=self.features_as_channels,
                cast_to=self.cast_to,
                transforms=self.transforms[split_name],
                map_labels=self.map_labels,
            )

            # Apply sampling only to the train split
            if split_name == "train":
                dataset = self._sample_dataset(dataset)

            domain_labels += [i] * len(dataset)
            datasets.append(dataset)

        if len(datasets) == 1:
            return datasets[0], domain_labels
        else:
            return ConcatDataset(datasets), domain_labels

    def setup(self, stage: str):
        """Assign the datasets to the corresponding split. ``self.datasets``
        will be a dictionary with the split name as key and the dataset as
        value.

        Parameters
        ----------
        stage : str
            The stage of the setup. This could be:
            - "fit": Load the train and validation datasets
            - "test": Load the test dataset
            - "predict": Load the predict dataset

        Raises
        ------
        ValueError
            If the stage is not one of: "fit", "test" or "predict"
        """
        if stage == "fit":
            self.datasets["train"] = self._load_dataset("train")

            if self.use_val_with_train:
                train_dataset, train_domains = self.datasets["train"]
                val_dataset, val_domains = self._load_dataset("validation")
                train_dataset = ConcatDataset([train_dataset, val_dataset])
                train_domains += val_domains
                self.datasets["train"] = (train_dataset, train_domains)

            if self.use_train_as_validation:
                self.datasets["validation"] = self.datasets["train"]
            else:
                self.datasets["validation"] = self._load_dataset("validation")

        elif stage == "test":
            self.datasets["test"] = self._load_dataset("test")
        elif stage == "predict":
            self.datasets["predict"] = self._load_dataset("predict")
        else:
            raise ValueError(f"Invalid setup stage: {stage}")

    def _get_loader(self, split_name: str, shuffle: bool) -> DataLoader:
        """Get a dataloader for the given split.

        Parameters
        ----------
        split_name : str
            The name of the split. This must be one of: "train", "validation",
            "test" or "predict".
        shuffle : bool
            Shuffle the data or not.

        Returns
        -------
        DataLoader
            A dataloader for the given split.
        """
        dataset, domain_labels = self.datasets[split_name]
        if self.n_domains_per_sample is not None:
            print(
                f"Using DataLoader with RandomDomainSampler with n_domains_per_sample={self.n_domains_per_sample}"
            )
            sampler = RandomDomainSampler(
                dataset,
                domain_labels,
                batch_size=self.batch_size,
                consistent_iterating=False,
                n_domains_per_sample=self.n_domains_per_sample,
            )
            return DataLoader(
                dataset,
                batch_sampler=sampler,
                num_workers=self.num_workers,
            )
        else:
            print(f"Using DataLoader with shuffle={shuffle}")
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=shuffle,
                pin_memory=True,
                drop_last=self.drop_last,
            )

    def train_dataloader(self) -> DataLoader:
        return self._get_loader("train", shuffle=self.shuffle_train)

    def val_dataloader(self) -> DataLoader:
        return self._get_loader("validation", shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._get_loader("test", shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self._get_loader("predict", shuffle=False)

    def __str__(self):
        return f"MultiModalHARSeriesDataModule(data_path={', '.join([str(d) for d in self.data_path])}, batch_size={self.batch_size})"

    def __repr__(self) -> str:
        return str(self)