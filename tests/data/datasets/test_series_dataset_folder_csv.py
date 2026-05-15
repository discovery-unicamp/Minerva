import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
from tempfile import TemporaryDirectory, NamedTemporaryFile
from minerva.data.datasets.series_dataset import (
    SeriesFolderCSVDataset,
)

from minerva.transforms.transform import _Transform


@pytest.fixture
def sample_data(tmp_path):
    """Fixture to create sample CSV data in a temporary directory."""
    data_path = Path(tmp_path)

    # Sample 1
    df1 = pd.DataFrame(
        {
            "accel-x": [0.5, 0.68, 0.49],
            "accel-y": [0.02, 0.02, 0.00],
            "class": [1, 1, 1],
        }
    )
    df1.to_csv(data_path / "sample-1.csv", index=False)

    # Sample 2 (longer sequence)
    df2 = pd.DataFrame(
        {
            "accel-x": [0.5, 0.68, 0.49, 3.14],
            "accel-y": [0.02, 0.02, 0.00, 1.41],
            "class": [0, 0, 0, 0],
        }
    )
    df2.to_csv(data_path / "sample-2.csv", index=False)
    return data_path


def test_dataset_initialization(sample_data):
    """Test dataset initializes correctly with default and custom parameters."""
    dataset = SeriesFolderCSVDataset(
        sample_data, features=["accel-x", "accel-y"], label="class"
    )

    assert len(dataset) == 2  # Check number of samples
    assert isinstance(dataset.features, list)
    assert dataset.label == "class"
    assert dataset.cast_to == "float32"
    assert dataset.transforms == []


def test_data_loading(sample_data):
    """Test that data is correctly loaded and structured."""
    dataset = SeriesFolderCSVDataset(
        sample_data, features=["accel-x", "accel-y"], label="class"
    )

    data, label = dataset[0]

    assert isinstance(data, np.ndarray)
    assert isinstance(label, np.ndarray)
    assert data.shape == (2, 3)  # (features, time-steps)
    assert label.shape == (3, 1)  # (time-steps, 1)

    data, label = dataset[1]
    assert data.shape == (2, 4)  # (features, time-steps)
    assert label.shape == (4, 1)  # (time-steps, 1)


def test_data_loading_single_feature(sample_data):
    """Test that data is correctly loaded and structured."""
    dataset = SeriesFolderCSVDataset(sample_data, features="accel-x", label="class")

    data, label = dataset[0]

    assert isinstance(data, np.ndarray)
    assert isinstance(label, np.ndarray)
    assert data.shape == (1, 3)  # (features, time-steps)
    assert label.shape == (3, 1)  # (time-steps, 1)
    np.testing.assert_allclose(data, np.array([[0.5, 0.68, 0.49]]))

    data, label = dataset[1]
    assert data.shape == (1, 4)  # (features, time-steps)
    assert label.shape == (4, 1)  # (time-steps, 1)
    np.testing.assert_allclose(data, np.array([[0.5, 0.68, 0.49, 3.14]]))


def test_data_loading_without_label(sample_data):
    """Test that data is correctly loaded and structured when no label is specified."""
    dataset = SeriesFolderCSVDataset(
        sample_data, features=["accel-x", "accel-y"], label=None
    )

    data = dataset[0]

    assert isinstance(data, np.ndarray)
    assert data.shape == (2, 3)  # (features, time-steps)
    np.testing.assert_allclose(data, np.array([[0.5, 0.68, 0.49], [0.02, 0.02, 0.00]]))


def test_data_loading_without_features(sample_data):
    """Test that data is correctly loaded and structured when no features are specified."""
    dataset = SeriesFolderCSVDataset(sample_data, features=None, label="class")

    data, label = dataset[0]

    assert isinstance(data, np.ndarray)
    assert data.shape == (2, 3)  # (features, time-steps)
    np.testing.assert_allclose(data, np.array([[0.5, 0.68, 0.49], [0.02, 0.02, 0.00]]))


def test_data_loading_without_features_and_label(sample_data):
    """Test that data is correctly loaded and structured when no features and label are specified."""
    dataset = SeriesFolderCSVDataset(sample_data, features=None, label=None)

    data = dataset[0]

    assert isinstance(data, np.ndarray)
    assert data.shape == (3, 3)  # (features, time-steps)
    np.testing.assert_allclose(
        data, np.array([[0.5, 0.68, 0.49], [0.02, 0.02, 0.00], [1, 1, 1]])
    )

    data = dataset[1]
    assert isinstance(data, np.ndarray)
    assert data.shape == (3, 4)  # (features, time-steps)
    np.testing.assert_allclose(
        data,
        np.array([[0.5, 0.68, 0.49, 3.14], [0.02, 0.02, 0.00, 1.41], [0, 0, 0, 0]]),
    )


def test_no_feature(sample_data):
    """Test that an error is raised when an invalid feature is specified."""
    with pytest.raises(ValueError):
        dataset = SeriesFolderCSVDataset(sample_data, features=[], label="class")


def test_lazy_loading(sample_data):
    """Test that lazy loading defers file reading until accessed."""
    dataset = SeriesFolderCSVDataset(
        sample_data, features=["accel-x", "accel-y"], label="class", lazy=True
    )

    # Before accessing, _cache should be None
    assert dataset._cache is None

    # Accessing an item should load data from file
    data, label = dataset[0]
    assert data.shape == (2, 3)
    assert label.shape == (3, 1)


def test_padding_functionality(sample_data):
    """Test that dataset correctly pads sequences to the longest sample size."""
    dataset = SeriesFolderCSVDataset(
        sample_data, features=["accel-x", "accel-y"], label="class", pad=True
    )

    # Shorter sequence (padded to 4 time steps) 3->4 time steps (reflect)
    data, label = dataset[0]
    assert data.shape == (2, 4)  # Padded to 4 time steps
    assert label.shape == (4, 1)  # Labels should also be padded
    np.testing.assert_allclose(
        data, np.array([[0.5, 0.68, 0.49, 0.5], [0.02, 0.02, 0.00, 0.02]])
    )
    np.testing.assert_allclose(label, np.array([[1], [1], [1], [1]]))

    # Longer sequence (no padding is needed)
    data, label = dataset[1]
    assert data.shape == (2, 4)  # Padded to 4 time steps
    assert label.shape == (4, 1)  # Labels should also be padded
    np.testing.assert_allclose(
        data, np.array([[0.5, 0.68, 0.49, 3.14], [0.02, 0.02, 0.00, 1.41]])
    )
    np.testing.assert_allclose(label, np.array([[0], [0], [0], [0]]))


def test_transforms(sample_data):
    """Test that transforms are correctly applied to the dataset."""

    class Transform(_Transform):
        def __init__(self, multiplier):
            self.multiplier = multiplier

        def __call__(self, x):
            return x * self.multiplier

    dataset = SeriesFolderCSVDataset(
        sample_data,
        features=["accel-x", "accel-y"],
        label="class",
        transforms=[Transform(2)],
    )

    data, label = dataset[0]
    assert data.shape == (2, 3)  # (features, time-steps)
    np.testing.assert_allclose(
        data, np.array([[0.5, 0.68, 0.49], [0.02, 0.02, 0.00]]) * 2
    )

    data, label = dataset[1]
    assert data.shape == (2, 4)  # (features, time-steps)
    np.testing.assert_allclose(
        data, np.array([[0.5, 0.68, 0.49, 3.14], [0.02, 0.02, 0.00, 1.41]]) * 2
    )

    # Single transform
    dataset = SeriesFolderCSVDataset(
        sample_data,
        features=["accel-x", "accel-y"],
        label="class",
        transforms=Transform(2),
    )

    data, label = dataset[0]
    assert data.shape == (2, 3)  # (features, time-steps)
    np.testing.assert_allclose(
        data, np.array([[0.5, 0.68, 0.49], [0.02, 0.02, 0.00]]) * 2
    )

    data, label = dataset[1]
    assert data.shape == (2, 4)  # (features, time-steps)
    np.testing.assert_allclose(
        data, np.array([[0.5, 0.68, 0.49, 3.14], [0.02, 0.02, 0.00, 1.41]]) * 2
    )

    # Multiple transforms
    dataset = SeriesFolderCSVDataset(
        sample_data,
        features=["accel-x", "accel-y"],
        label="class",
        transforms=[Transform(2), Transform(3)],
    )

    data, label = dataset[0]
    assert data.shape == (2, 3)  # (features, time-steps)
    np.testing.assert_allclose(
        data, np.array([[0.5, 0.68, 0.49], [0.02, 0.02, 0.00]]) * 6
    )

    data, label = dataset[1]
    assert data.shape == (2, 4)  # (features, time-steps)
    np.testing.assert_allclose(
        data, np.array([[0.5, 0.68, 0.49, 3.14], [0.02, 0.02, 0.00, 1.41]]) * 6
    )


def test_string_representation(sample_data):
    """Test the __str__ and __repr__ methods."""
    dataset = SeriesFolderCSVDataset(sample_data)
    assert str(dataset) == repr(dataset)
    assert f"SeriesFolderCSVDataset at {sample_data}" in str(dataset)


def test_empty_directory():
    """Test dataset behavior with an empty directory."""
    with TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError):
            dataset = SeriesFolderCSVDataset(tmpdir)


def test_invalid_directory():
    """Test dataset behavior with an invalid directory."""
    with pytest.raises(ValueError):
        dataset = SeriesFolderCSVDataset("invalid-directory")


def test_not_directory():
    """Test dataset behavior with a non-directory path."""
    with NamedTemporaryFile() as tmpfile:
        with pytest.raises(ValueError):
            dataset = SeriesFolderCSVDataset(tmpfile.name)
