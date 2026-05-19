import contextlib
from collections.abc import Iterable
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytest
from torch.utils.data import Dataset

from minerva.data.datasets.series_dataset import MultiModalSeriesCSVDataset
from minerva.transforms.transform import _Transform


@pytest.fixture
def sample_csv(tmp_path):
    data = {
        "accel-x-0": [0.502123, 0.6820123, 0.498217],
        "accel-x-1": [0.02123, 0.02123, 0.00001],
        "accel-y-0": [0.502123, 0.502123, 1.414141],
        "accel-y-1": [0.502123, 0.502123, 3.141592],
        "class": [0, 1, 2],
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_dataset_length(sample_csv):
    dataset = MultiModalSeriesCSVDataset(data_path=sample_csv, label="class")
    assert len(dataset) == 3


def test_dataset_shape_features_as_channels(sample_csv):
    dataset = MultiModalSeriesCSVDataset(
        data_path=sample_csv,
        feature_prefixes=["accel-x", "accel-y"],
        label="class",
        features_as_channels=True,
    )
    data, label = dataset[0]
    assert data.shape == (2, 2)
    assert label == 0


def test_dataset_shape_features_as_vector(sample_csv):
    dataset = MultiModalSeriesCSVDataset(
        data_path=sample_csv,
        feature_prefixes=["accel-x", "accel-y"],
        label="class",
        features_as_channels=False,
    )
    data, label = dataset[0]
    assert data.shape == (4,)
    assert label == 0


def test_dataset_without_label(sample_csv):
    dataset = MultiModalSeriesCSVDataset(
        data_path=sample_csv,
        feature_prefixes=["accel-x", "accel-y"],
        label=None,
        features_as_channels=True,
    )
    data = dataset[0]
    assert data.shape == (2, 2)


def test_dataset_without_label_as_vector(sample_csv):
    dataset = MultiModalSeriesCSVDataset(
        data_path=sample_csv,
        feature_prefixes=["accel-x", "accel-y"],
        label=None,
        features_as_channels=False,
    )
    data = dataset[0]
    assert data.shape == (4,)


def test_dataset_without_any_feature(sample_csv):
    with pytest.raises(ValueError):
        dataset = MultiModalSeriesCSVDataset(
            data_path=sample_csv,
            feature_prefixes=[],
            label="class",
            features_as_channels=True,
        )


def test_dataset_with_single_feature(sample_csv):
    # ------ Feature as channel = True -----
    dataset = MultiModalSeriesCSVDataset(
        data_path=sample_csv,
        feature_prefixes=["accel-x-0"],
        label="class",
        features_as_channels=True,
    )
    data, label = dataset[0]
    assert data.shape == (1, 1)
    assert label == 0
    np.testing.assert_allclose(data, np.array([[0.502123]]))

    dataset = MultiModalSeriesCSVDataset(
        data_path=sample_csv,
        feature_prefixes=["accel"],
        label="class",
        features_as_channels=True,
    )
    data, label = dataset[0]
    assert data.shape == (1, 4)
    assert label == 0
    np.testing.assert_allclose(
        data, np.array([[0.502123, 0.02123, 0.502123, 0.502123]])
    )

    dataset = MultiModalSeriesCSVDataset(
        data_path=sample_csv,
        feature_prefixes="accel",
        label="class",
        features_as_channels=True,
    )
    data, label = dataset[0]
    assert data.shape == (1, 4)
    assert label == 0
    np.testing.assert_allclose(
        data, np.array([[0.502123, 0.02123, 0.502123, 0.502123]])
    )

    # Feature as channel = False
    dataset = MultiModalSeriesCSVDataset(
        data_path=sample_csv,
        feature_prefixes=["accel-x-0"],
        label="class",
        features_as_channels=False,
    )
    data, label = dataset[0]
    assert data.shape == (1,)
    assert label == 0
    np.testing.assert_allclose(data, np.array([0.502123]))

    dataset = MultiModalSeriesCSVDataset(
        data_path=sample_csv,
        feature_prefixes=["accel"],
        label="class",
        features_as_channels=False,
    )
    data, label = dataset[0]
    assert data.shape == (4,)
    assert label == 0
    np.testing.assert_allclose(data, np.array([0.502123, 0.02123, 0.502123, 0.502123]))

    dataset = MultiModalSeriesCSVDataset(
        data_path=sample_csv,
        feature_prefixes="accel",
        label="class",
        features_as_channels=False,
    )
    data, label = dataset[0]
    assert data.shape == (4,)
    assert label == 0
    np.testing.assert_allclose(data, np.array([0.502123, 0.02123, 0.502123, 0.502123]))


def test_dataset_without_label_and_single_feature(sample_csv):
    dataset = MultiModalSeriesCSVDataset(
        data_path=sample_csv,
        feature_prefixes=["accel-x-0"],
        label=None,
        features_as_channels=True,
    )
    data = dataset[0]
    assert data.shape == (1, 1)
    np.testing.assert_allclose(data, np.array([[0.502123]]))

    dataset = MultiModalSeriesCSVDataset(
        data_path=sample_csv,
        feature_prefixes=["accel"],
        label=None,
        features_as_channels=True,
    )
    data = dataset[0]
    assert data.shape == (1, 4)
    np.testing.assert_allclose(
        data, np.array([[0.502123, 0.02123, 0.502123, 0.502123]])
    )

    dataset = MultiModalSeriesCSVDataset(
        data_path=sample_csv,
        feature_prefixes="accel",
        label=None,
        features_as_channels=True,
    )
    data = dataset[0]
    assert data.shape == (1, 4)
    np.testing.assert_allclose(
        data, np.array([[0.502123, 0.02123, 0.502123, 0.502123]])
    )

    dataset = MultiModalSeriesCSVDataset(
        data_path=sample_csv,
        feature_prefixes=None,
        label=None,
        features_as_channels=True,
    )
    data = dataset[0]
    assert data.shape == (5, 1)
    np.testing.assert_allclose(
        data, np.array([[0.502123], [0.02123], [0.502123], [0.502123], [0]])
    )

    # Feature as channel = False
    dataset = MultiModalSeriesCSVDataset(
        data_path=sample_csv,
        feature_prefixes=["accel-x-0"],
        label=None,
        features_as_channels=False,
    )
    data = dataset[0]
    assert data.shape == (1,)
    np.testing.assert_allclose(data, np.array([0.502123]))

    dataset = MultiModalSeriesCSVDataset(
        data_path=sample_csv,
        feature_prefixes=["accel"],
        label=None,
        features_as_channels=False,
    )
    data = dataset[0]
    assert data.shape == (4,)
    np.testing.assert_allclose(data, np.array([0.502123, 0.02123, 0.502123, 0.502123]))

    dataset = MultiModalSeriesCSVDataset(
        data_path=sample_csv,
        feature_prefixes="accel",
        label=None,
        features_as_channels=False,
    )
    data = dataset[0]
    assert data.shape == (4,)
    np.testing.assert_allclose(data, np.array([0.502123, 0.02123, 0.502123, 0.502123]))

    dataset = MultiModalSeriesCSVDataset(
        data_path=sample_csv,
        feature_prefixes=None,
        label=None,
        features_as_channels=False,
    )
    data = dataset[0]
    assert data.shape == (5,)
    np.testing.assert_allclose(
        data, np.array([0.502123, 0.02123, 0.502123, 0.502123, 0])
    )


def test_dataset_with_transform(sample_csv):
    class Transform(_Transform):
        def __init__(self, multiplier: int):
            self.multiplier = multiplier

        def __call__(self, data):
            return data * self.multiplier

    dataset = MultiModalSeriesCSVDataset(
        data_path=sample_csv,
        feature_prefixes=["accel-x", "accel-y"],
        label="class",
        features_as_channels=True,
        transforms=[Transform(multiplier=2)],
    )
    data, label = dataset[0]
    np.testing.assert_allclose(
        data,
        np.array([[0.502123 * 2, 0.02123 * 2], [0.502123 * 2, 0.502123 * 2]]),
    )
    assert label == 0

    # Without a list of transforms
    dataset = MultiModalSeriesCSVDataset(
        data_path=sample_csv,
        feature_prefixes=["accel-x", "accel-y"],
        label="class",
        features_as_channels=True,
        transforms=Transform(multiplier=2),
    )
    data, label = dataset[0]
    np.testing.assert_allclose(
        data,
        np.array([[0.502123 * 2, 0.02123 * 2], [0.502123 * 2, 0.502123 * 2]]),
    )
    assert label == 0

    # With multiple transforms
    dataset = MultiModalSeriesCSVDataset(
        data_path=sample_csv,
        feature_prefixes=["accel-x", "accel-y"],
        label="class",
        features_as_channels=True,
        transforms=[Transform(multiplier=2), Transform(multiplier=3)],
    )
    data, label = dataset[0]
    np.testing.assert_allclose(
        data,
        np.array([[0.502123 * 6, 0.02123 * 6], [0.502123 * 6, 0.502123 * 6]]),
    )
    assert label == 0


def test_dataset_with_map_labels(sample_csv):
    map_labels = {0: 10, 1: 20, 2: 30}
    dataset = MultiModalSeriesCSVDataset(
        data_path=sample_csv,
        feature_prefixes=["accel-x", "accel-y"],
        label="class",
        features_as_channels=True,
        map_labels=map_labels,
    )
    data, label = dataset[0]
    assert label == 10

    data, label = dataset[1]
    assert label == 20

    data, label = dataset[2]
    assert label == 30


def test_dataset_with_invalid_map_labels(sample_csv):
    with pytest.raises(ValueError):
        map_labels = {0: 10, 1: 20}
        dataset = MultiModalSeriesCSVDataset(
            data_path=sample_csv,
            feature_prefixes=["accel-x", "accel-y"],
            label="class",
            features_as_channels=True,
            map_labels=map_labels,
        )


def test_dataset_str(sample_csv):
    dataset = MultiModalSeriesCSVDataset(
        data_path=sample_csv,
        feature_prefixes=["accel-x", "accel-y"],
        label="class",
        features_as_channels=True,
    )
    assert str(dataset) == f"MultiModalSeriesCSVDataset at {sample_csv} (3 samples)"
    assert repr(dataset) == f"MultiModalSeriesCSVDataset at {sample_csv} (3 samples)"


def test_dataset_with_invalid_label(sample_csv):
    with pytest.raises(ValueError):
        dataset = MultiModalSeriesCSVDataset(
            data_path=sample_csv,
            feature_prefixes=["accel-x", "accel-y"],
            label="invalid",
            features_as_channels=True,
        )


def test_dataset_with_return_index_as_label(sample_csv):
    dataset = MultiModalSeriesCSVDataset(
        data_path=sample_csv,
        feature_prefixes=["accel-x", "accel-y"],
        label="return_index_as_label",
        features_as_channels=True,
        map_labels={0: 10, 1: 20, 2: 30},
    )
    labels_from_dataset = dataset[:][1]
    ground_truth = np.arange(len(dataset))

    while sum(abs(labels_from_dataset - ground_truth)) == 0:
        np.random.shuffle(labels_from_dataset)

    assert len(labels_from_dataset) == len(ground_truth)
    assert sum(abs(labels_from_dataset - ground_truth)) > 0
    sorted_labels_from_dataset = np.sort(labels_from_dataset)
    assert sum(abs(sorted_labels_from_dataset - ground_truth)) == 0
