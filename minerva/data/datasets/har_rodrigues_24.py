from typing import Iterable, List, Optional, Tuple, Union
import numpy as np
import os
from minerva.utils.typing import PathLike
import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import as_strided as ast


def norm_shape(
    shape: Union[int, Tuple[int, ...], np.ndarray]
) -> Tuple[int, ...]:
    """Normalize numpy array shapes so they're always expressed as a tuple,
    even for one-dimensional shapes.

    Parameters
    ----------
    shape : int, tuple, or numpy.ndarray
        The shape to be normalized.

    Returns
    -------
    Tuple[int, ...]
        The normalized shape.
    """
    if isinstance(shape, int):
        return (shape,)
    elif isinstance(shape, tuple):
        return shape
    elif isinstance(shape, np.ndarray):
        return tuple(shape.tolist())
    else:
        raise TypeError(
            "shape must be an int, a tuple of ints, or a numpy array"
        )


def sliding_window(a, ws, ss, flatten=True):
    """Return a sliding window over a in any number of dimensions

    Parameters:
        a  - an n-dimensional numpy array
        ws - an int (a is 1D) or tuple (a is 2D or greater) representing the size
             of each dimension of the window
        ss - an int (a is 1D) or tuple (a is 2D or greater) representing the
             amount to slide the window in each dimension. If not specified, it
             defaults to ws.
        flatten - if True, all slices are flattened, otherwise, there is an
                  extra dimension for each dimension of the input.

    Returns
        an array containing each n-dimensional window from a
    """

    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)

    # convert ws, ss, and a.shape to numpy arrays so that we can do math in every
    # dimension at once.
    ws = np.array(ws)
    ss = np.array(ss)
    shape = np.array(a.shape)

    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shape), len(ws), len(ss)]
    if 1 != len(set(ls)):
        raise ValueError(
            "a.shape, ws and ss must all have the same length. They were %s"
            % str(ls)
        )

    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shape):
        raise ValueError(
            "ws cannot be larger than a in any dimension.\
 a.shape was %s and ws was %s"
            % (str(a.shape), str(ws))
        )

    # how many slices will there be in each dimension?
    newshape = norm_shape(((shape - ws) // ss) + 1)
    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)
    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    strided = ast(a, shape=newshape, strides=newstrides)
    if not flatten:
        return strided

    # Collapse strided so that it has one more dimension than the window.  I.e.,
    # the new array is a flat list of slices.
    meat = len(ws) if ws.shape else 0
    firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    # remove any dimensions with size 1
    # dim = filter(lambda i : i != 1,dim)
    return strided.reshape(dim)


def opp_sliding_window(data_x, data_y, ws, ss):

    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))

    data_y = np.reshape(data_y, (len(data_y),))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, ws, ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(
        np.uint8
    )


class HARDatasetCPC(Dataset):
    def __init__(
        self,
        data_path: Union[PathLike, List[PathLike]],
        input_size: int,
        window: int,
        overlap: int,
        phase: str = "train",
        use_train_as_val: bool = False,
        columns: Optional[List[str]] = None,
    ):
        """
        Initializes the dataset by loading the dataset from CSV files,
        segmenting the data into windows, and preparing it for training
        or evaluation.

        Parameters
        ----------
        data_path : Union[PathLike, List[PathLike]]
            The path to the directory containing the dataset files. If a list of
            paths is provided, the datasets will be concatenated, in the order
            provided, into a single dataset.
        input_size : int
            The expected size of input features.
        window : int
            The size of the sliding window used to segment the data.
        overlap : int
            The overlap between consecutive windows.
        phase : str
            The phase of the dataset ('train', 'val', or 'test').
        use_train_as_val : bool
            Whether to use the training set as the validation set.
        columns : Optional[List[str]]
            The columns to be used as input features. If None, the default
            columns ['accel-x', 'accel-y', 'accel-z', 'gyro-x', 'gyro-y',
            'gyro-z'] will be used.
        """
        # Create a list of paths if only one path is provided
        self.paths = data_path if isinstance(data_path, list) else [data_path]
        self.use_train_as_val = use_train_as_val
        self.input_size = input_size
        self.columns = (
            columns
            if columns is not None
            else ["accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"]
        )

        self.data_raw = self.load_dataset()
        assert input_size == self.data_raw[phase]["data"].shape[1]

        # Obtaining the segmented data
        self.data, self.labels = opp_sliding_window(
            self.data_raw[phase]["data"],
            self.data_raw[phase]["labels"],
            window,
            overlap,
        )

    # Load .csv file

    def load_dataset(self):
        """
        Loads the dataset from CSV files, concatenates them into numpy arrays,
        and converts them to the appropriate data types.

        Returns
        -------
        dict
            A dictionary containing 'data' and 'labels' for 'train', 'val', and 'test'
            phases, where 'data' is a numpy array of concatenated data and 'labels'
            is a numpy array of concatenated labels.
        """
        datasets = {}

        for phase in ["train", "val", "test"]:
            if phase == "val":
                if self.use_train_as_val:
                    datasets[phase] = datasets["train"]
                    continue

            data_x = []
            data_y = []

            for path in self.paths:
                # Transform it to a path and add phase
                path = Path(path)
                phase_path = path / phase
                
                for f in phase_path.glob("*.csv"):
                    data = pd.read_csv(f)
                    x = data[self.columns].values
                    y = data["activity code"].values
                    data_x.append(x)
                    data_y.append(y)

            datasets[phase] = {
                "data": np.concatenate(data_x),
                "labels": np.concatenate(data_y),
            }
            datasets[phase]["data"] = datasets[phase]["data"].astype(np.float32)
            datasets[phase]["labels"] = datasets[phase]["labels"].astype(
                np.uint8
            )

        return datasets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index, :, :]

        # Aplicar permute para ter a forma [64, 6, 60]
        data = torch.from_numpy(data).float().permute(1, 0)

        label = torch.from_numpy(np.asarray(self.labels[index])).long()
        return data, label
