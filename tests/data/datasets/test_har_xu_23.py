import os
from typing import List, Tuple

import numpy as np
import pytest
import torch
from sklearn.metrics.pairwise import cosine_similarity
from statsmodels.tsa.stattools import adfuller
from torch.utils.data import Dataset

from minerva.data.datasets.har_xu_23 import HarDataset, TNCDataset
from minerva.utils.typing import PathLike

#################### TNC DATASET ####################


@pytest.fixture
def tnc_dataset_params():
    n_samples = 100
    n_channels = 6
    n_timesteps = 1000
    mc_sample_size = 5
    window_size = 128
    epsilon = 3
    rng = np.random.RandomState(42)
    data = rng.randn(n_samples, n_channels, n_timesteps)

    return (
        data,
        mc_sample_size,
        window_size,
        epsilon,
    )


@pytest.fixture(params=[True, False])
def tnc_dataset(request, tnc_dataset_params):
    data, mc_sample_size, window_size, epsilon = tnc_dataset_params
    adf = (
        request.param
    )  # Parametrize ADF test for determining neighbors and non-neighbors
    return TNCDataset(
        x=data,
        mc_sample_size=mc_sample_size,
        window_size=window_size,
        epsilon=epsilon,
        adf=adf,
    )


def test_len_tnc_dataset(tnc_dataset):
    assert len(tnc_dataset) == 100


def test_getitem_tnc_dataset(tnc_dataset):
    central_window, close_neighbors, non_neighbors = tnc_dataset[0]
    assert central_window.shape == (128, 6)
    assert close_neighbors.shape == (5, 128, 6)
    assert non_neighbors.shape == (5, 128, 6)


def test_getitem_tnc_dataset_error_correlation(tnc_dataset):
    # this should trick adf test to return a correlation error
    # The data is only zeros, so the correlation is not defined
    tnc_dataset.time_series = np.zeros_like(tnc_dataset.time_series)
    central_window, close_neighbors, non_neighbors = tnc_dataset[0]
    assert central_window.shape == (128, 6)
    assert close_neighbors.shape == (5, 128, 6)
    assert non_neighbors.shape == (5, 128, 6)

    tnc_dataset.time_series = np.ones_like(tnc_dataset.time_series)
    central_window, close_neighbors, non_neighbors = tnc_dataset[0]
    assert central_window.shape == (128, 6)
    assert close_neighbors.shape == (5, 128, 6)
    assert non_neighbors.shape == (5, 128, 6)


def test_tnc_dataset_small_time_series():
    """
    Tests whether TNCDataset can handle a very short time series.
    Should gracefully handle cases where `n_timesteps < 2 * window_size`.
    """
    n_samples = 10
    n_channels = 6
    n_timesteps = 10  # Smaller than 2 * window_size (128 * 2 = 256)
    mc_sample_size = 3
    window_size = 128
    epsilon = 2
    data = np.random.randn(n_samples, n_channels, n_timesteps)

    dataset = TNCDataset(
        x=data,
        mc_sample_size=mc_sample_size,
        window_size=window_size,
        epsilon=epsilon,
        adf=True,
    )

    with pytest.raises(ValueError):
        sample = dataset[0]


def test_tnc_dataset_identical_samples(tnc_dataset):
    """
    Tests if TNCDataset can handle cases where all samples are identical.
    The cosine similarity should be perfect, but it should still return
    a diverse set of neighbors.
    """
    tnc_dataset.time_series = np.ones_like(tnc_dataset.time_series)
    _, close_neighbors, _ = tnc_dataset[0]

    assert close_neighbors.shape == (5, 128, 6)
    assert not np.allclose(
        close_neighbors, np.zeros_like(close_neighbors)
    ), "Close neighbors should not be all zeros."


#################### HAR DATASET ####################
@pytest.fixture
def har_dataset_params(tmp_path):
    n_samples = 100
    n_timesteps = 128
    n_features = 6
    rng = np.random.RandomState(42)
    data = rng.randn(n_samples, n_timesteps, n_features)
    labels = rng.randint(0, 10, size=(n_samples,))

    data_path = tmp_path / "data"
    data_path.mkdir()
    np.save(data_path / "train_data_subseq.npy", data)
    np.save(data_path / "train_labels_subseq.npy", labels)
    np.save(data_path / "val_data_subseq.npy", data)
    np.save(data_path / "val_labels_subseq.npy", labels)
    np.save(data_path / "test_data_subseq.npy", data)
    np.save(data_path / "test_labels_subseq.npy", labels)

    return data_path, "train"


@pytest.fixture
def har_dataset(har_dataset_params):
    data_path, annotate = har_dataset_params
    return HarDataset(data_path=data_path, annotate=annotate)


def test_len_har_dataset(har_dataset):
    assert len(har_dataset) == 100


def test_getitem_har_dataset(har_dataset):
    features, label = har_dataset[0]
    assert features.shape == (128, 6)
    assert isinstance(label, torch.Tensor)
    assert label.dtype == torch.long


def test_getitem_har_dataset_flatten(har_dataset_params):
    data_path, annotate = har_dataset_params
    dataset = HarDataset(data_path=data_path, annotate=annotate, flatten=True)
    features, label = dataset[0]
    assert features.shape == (128 * 6,)
    assert isinstance(label, torch.Tensor)
    assert label.dtype == torch.long


def test_har_dataset_invalid_path():
    with pytest.raises(FileNotFoundError):
        HarDataset(data_path="/invalid/path", annotate="train")


def test_har_dataset_mismatched_data_labels(tmp_path):
    n_samples = 100
    n_timesteps = 128
    n_features = 6
    rng = np.random.RandomState(42)
    data = rng.randn(n_samples, n_timesteps, n_features)
    labels = rng.randint(0, 10, size=(n_samples - 1,))  # Mismatched length

    data_path = tmp_path / "data"
    data_path.mkdir()
    np.save(data_path / "train_data_subseq.npy", data)
    np.save(data_path / "train_labels_subseq.npy", labels)

    with pytest.raises(AssertionError):
        HarDataset(data_path=data_path, annotate="train")


def test_har_dataset_different_annotate(har_dataset_params):
    data_path, _ = har_dataset_params
    dataset = HarDataset(data_path=data_path, annotate="val")
    assert len(dataset) == 100
    features, label = dataset[0]
    assert features.shape == (128, 6)
    assert isinstance(label, torch.Tensor)
    assert label.dtype == torch.long

    dataset = HarDataset(data_path=data_path, annotate="test")
    assert len(dataset) == 100
    features, label = dataset[0]
    assert features.shape == (128, 6)
    assert isinstance(label, torch.Tensor)
    assert label.dtype == torch.long
