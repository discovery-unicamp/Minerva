from typing import Callable, List, Optional, Tuple, Union
from pathlib import Path
from collections.abc import Iterable

import numpy as np
import pandas as pd
import contextlib


class MultiModalSeriesCSVDataset:
    def __init__(
        self,
        data_path: Union[Path, str],
        feature_prefixes: Union[str, List[str]] = None,
        label: str = None,
        features_as_channels: bool = True,
        cast_to: str = "float32",
        transforms: Optional[Union[Callable, List[Callable]]] = None,
    ):
        """This datasets assumes that the data is in a single CSV file with
        series of data. Each row is a single sample that can be composed of
        multiple modalities (series). Each column is a feature of some series
        with the prefix indicating the series. The suffix may indicates the
        time step. For instance, if we have two series, accel-x and accel-y,
        the data will look something like:

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
        data_path : Union[Path, str]
            The location of the CSV file
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
        transforms: Optional[List[Callable]], optional
            A list of transforms that will be applied to each sample
            individually. Each transform must be a callable that receives a
            numpy array and returns a numpy array. The transforms will be
            applied in the order they are specified.

        Examples
        --------
        # Using the data from the example above, and features_as_channels=False
        >>> data_path = "data.csv"
        >>> dataset = MultiModalSeriesCSVDataset(
                data_path,
                feature_prefixes=["accel-x", "accel-y"],
                label="class"
            )
        >>> data, label = dataset[0]
        >>> data.shape
        (4, )

        # Using the data from the example above, and features_as_channels=True
        >>> dataset = MultiModalSeriesCSVDataset(
                data_path,
                feature_prefixes=["accel-x", "accel-y"],
                label="class",
                features_as_channels=True
            )
        >>> data, label = dataset[0]
        >>> data.shape
        (2, 2)

        # And the dataset length
        >>> len(dataset)
        3

        """
        self.data_path = Path(data_path)

        if feature_prefixes is not None:
            if not isinstance(feature_prefixes, Iterable):
                feature_prefixes = [feature_prefixes]
        self.feature_prefixes = feature_prefixes
        self.label = label
        self.cast_to = cast_to
        self.features_as_channels = features_as_channels
        if transforms is not None:
            if not isinstance(transforms, list):
                transforms = [transforms]
        else:
            transforms = []
        self.transforms = transforms
        self.data, self.labels = self._load_data()

    def _load_data(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Load data from the CSV file

        Returns
        -------
        Tuple[np.ndarray, Optional[np.ndarray]]
            A 2-element tuple with the data and the labels. The second element
            is None if the label is not specified.
        """
        df = pd.read_csv(self.data_path)

        # Select columns with the given prefixes
        # If None, select all columns except the label (if specified)
        if self.feature_prefixes is None:
            selected_columns = [col for col in df.columns if col != self.label]
        else:
            selected_columns = [
                col
                for col in df.columns
                if any(prefix in col for prefix in self.feature_prefixes)
            ]

        selected_columns = list(selected_columns)

        # Select the data from the selected columns
        data = df[selected_columns].to_numpy()

        # If features_as_channels is True, reshape the data to (N, C, T) where
        # N is the number of samples, C is the number of channels and T is the
        # number of time steps
        if self.features_as_channels:
            data = data.reshape(
                -1,
                len(self.feature_prefixes),
                data.shape[1] // len(self.feature_prefixes),
            )

        # Cast the data to the specified type
        if self.cast_to:
            data = data.astype(self.cast_to)

        # If label is specified, return the data and the labels
        if self.label:
            labels = df[self.label].to_numpy()
            return data, labels
        # If label is not specified, return only the data
        else:
            return data, None

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
        self, index: int
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        # Get data and apply transforms
        data = self.data[index]
        for transform in self.transforms:
            data = transform(data)

        # Return data and label if specified, else return only the data
        if self.label:
            return data, self.labels[index]
        else:
            return data

    def __str__(self) -> str:
        return f"MultiModalSeriesCSVDataset at {self.data_path} ({len(self)} samples)"

    def __repr__(self) -> str:
        return str(self)


class SeriesFolderCSVDataset:
    def __init__(
        self,
        data_path: Union[Path, str],
        features: Union[str, List[str]] = None,
        label: str = None,
        pad: bool = False,
        cast_to: str = "float32",
        transforms: Optional[List[Callable]] = None,
        lazy: bool = False,
    ):
        """This dataset assumes that the data is in a folder with multiple CSV
        files. Each CSV file is a single sample that can be composed of
        multiple time steps (rows). Each column is a feature of the sample.

        For instance, if we have two samples, sample-1.csv and sample-2.csv,
        the directory structure will look something like:

        data_path
        ├── sample-1.csv
        └── sample-2.csv

        And the data will look something like:
        - sample-1.csv:
            +---------+---------+--------+
            | accel-x | accel-y | class  |
            +---------+---------+--------+
            | 0.502123| 0.02123 | 1      |
            | 0.682012| 0.02123 | 1      |
            | 0.498217| 0.00001 | 1      |
            +---------+---------+--------+
        - sample-2.csv:
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

        Notes
        -----
        - Samples may have different number of time steps. Use ``pad`` to pad
            the data to the length of the longest sample.

        Examples
        --------
        # Using the data from the example above
        >>> data_dir = "train_folder"
        >>> dataset = SeriesFolderCSVDataset(
                data_dir,
                features=["accel-x", "accel-y"],
                label="class"
            )
        >>> data, label = dataset[0]
        >>> data.shape
        (2, 3)
        >>> label.shape
        (3,)
        >>> data, label = dataset[1]
        >>> data.shape
        (2, 4)
        >>> label.shape
        (4,)

        Parameters
        ----------
        data_path : str
            The location of the directory with CSV files
        features: List[str]
            A list with column names that will be used as features. If None,
            all columns except the label will be used as features.
        pad: bool, optional
            If True, the data will be padded to the length of the longest
            sample. Note that padding will be applyied after the transforms,
            and also to the labels if specified.
        label: str, optional
            Specify the name of the column with the label of the data
        cast_to: str, optional
            Cast the numpy data to the specified type
        transforms: Optional[List[Callable]], optional
            A list of transforms that will be applied to each sample
            individually. Each transform must be a callable that receives a
            numpy array and returns a numpy array. The transforms will be
            applied in the order they are specified.
        lazy: bool, optional
            If True, the data will be loaded lazily (i.e. the CSV files will be
            read only when needed)
        """
        self.data_path = Path(data_path)
        if features is not None:
            if not isinstance(features, Iterable):
                features = [features]
        self.features = features
        self.label = label
        self.pad = pad
        self.cast_to = cast_to
        if transforms is not None:
            if not isinstance(transforms, list):
                transforms = [transforms]
        else:
            transforms = []
        self.transforms = transforms

        self._files = self._scan_data()
        # Data contains all the data if lazy is False else None
        self._cache = self._read_all_csv() if not lazy else None
        self._longest_sample_size = self._get_longest_sample_size()

    @contextlib.contextmanager
    def _disable_fix_length(self):
        """Decorator to disable fix_length when calling a function"""
        old_fix_length = self.pad
        self.pad = False
        yield
        self.pad = old_fix_length

    def _scan_data(self) -> List[Path]:
        """List the CSV files in the data directory

        Returns
        -------
        List[Path]
            List of CSV files
        """
        return list(sorted(self.data_path.glob("*.csv")))

    def _get_longest_sample_size(self) -> int:
        """Return the size of the longest sample in the dataset

        Returns
        -------
        int
            The size of the longest sample in the dataset
        """
        if not self.pad:
            return 0

        # Iterate
        with self._disable_fix_length():
            longest_sample_size = max(
                self[i][0].shape[-1] for i in range(len(self))
            )
        return longest_sample_size

    def _read_csv(self, path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Read a single CSV file (a single sample)

        Parameters
        ----------
        path : Path
            The path to the CSV file

        Returns
        -------
        Tuple[np.ndarray, Optional[np.ndarray]]
            A 2-element tuple with the data and the label. If the label is not
            specified, the second element is None.
        """
        # Read the data
        original_data = pd.read_csv(path)

        # Collect the features
        if self.features is None:
            selected_columns = [
                col for col in original_data.columns if col != self.label
            ]
        else:
            selected_columns = self.features
        # Transform it to a list if it is not
        selected_columns = list(selected_columns)

        data = original_data[selected_columns].values
        data = data.swapaxes(0, 1)

        # Cast the data to the specified type
        if self.cast_to:
            data = data.astype(self.cast_to)

        # Read the label if specified and return the data and the label
        if self.label is not None:
            return data, original_data[[self.label]].values
        # If label is not specified, return only the data
        else:
            return data, None

    def _read_all_csv(
        self,
    ) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """Read all the CSV files in the data directory

        Returns
        -------
        Union[Tuple[np.ndarray, np.ndarray], np.ndarray]
            A list of 2-element tuple with the data and the label. If the label is not specified, the second element of the tuples are None.
        """
        return [self._read_csv(f) for f in self._files]

    def __len__(self) -> int:
        return len(self._files)

    def _pad_data(self, data: np.ndarray) -> np.ndarray:
        """Pad the data to the length of the longest sample. In summary, this
        function makes the data cyclic.

        Parameters
        ----------
        data : np.ndarray
            The data to pad

        Returns
        -------
        np.ndarray
            The padded data
        """
        time_len = data.shape[-1]

        if time_len == self._longest_sample_size:
            return data

        # Repeat the data along the time axis to match the longest sample size
        repetitions = self._longest_sample_size // time_len + 1
        data = np.tile(data, (1, repetitions))[:, : self._longest_sample_size]
        return data

    def __getitem__(
        self, idx: int
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Get a single sample from the dataset

        Parameters
        ----------
        idx : int
            The index of the sample

        Returns
        -------
        Union[Tuple[np.ndarray, np.ndarray], np.ndarray]
            A 2-element tuple with the data and the label if the label is
            specified, otherwise only the data.
        """
        # If the data is not loaded, load it lazily (read the CSV file)
        if self._cache is None:
            data, label = self._read_csv(self._files[idx])
        # Else, read from the loaded data
        else:
            data, label = self._cache[idx]

        # Pad the data if fix_length is True
        if self.pad:
            data = self._pad_data(data)
            if label is not None:
                label = self._pad_data(label)

        # Apply transforms
        for transform in self.transforms:
            data = transform(data)

        # If label is specified, return the data and the label
        if label is not None:
            return data, label
        # Else, return only the data
        else:
            return data

    def __str__(self) -> str:
        return (
            f"SeriesFolderCSVDataset at {self.data_path} ({len(self)} samples)"
        )

    def __repr__(self) -> str:
        return str(self)

