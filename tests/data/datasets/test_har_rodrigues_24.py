import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from minerva.data.datasets.har_rodrigues_24 import (
    HARDatasetCPC,
    norm_shape,
    sliding_window,
    opp_sliding_window,
)


@pytest.fixture
def sample_data(tmp_path):
    # Create sample data
    data = {
        "accel-x": np.random.rand(100),
        "accel-y": np.random.rand(100),
        "accel-z": np.random.rand(100),
        "gyro-x": np.random.rand(100),
        "gyro-y": np.random.rand(100),
        "gyro-z": np.random.rand(100),
        "activity code": np.random.randint(0, 5, 100),
    }
    df = pd.DataFrame(data)

    # Create train, val, test directories and save sample data
    for phase in ["train", "val", "test"]:
        phase_path = tmp_path / phase
        phase_path.mkdir()
        df.to_csv(phase_path / "sample.csv", index=False)

    return tmp_path


def test_norm_shape():
    x = norm_shape(shape=1)
    assert isinstance(x, tuple)
    assert x == (1,)

    x = norm_shape(shape=(1, 2, 3))
    assert isinstance(x, tuple)
    assert x == (1, 2, 3)

    x = norm_shape(np.array([1, 2, 3]))
    assert isinstance(x, tuple)
    assert x == (1, 2, 3)

    with pytest.raises(TypeError):
        norm_shape(shape="invalid_type")


def test_sliding_window():
    a = np.arange(10)
    ws = 3
    result = sliding_window(a, ws, ss=1)
    expected = np.array(
        [
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [4, 5, 6],
            [5, 6, 7],
            [6, 7, 8],
            [7, 8, 9],
        ]
    )
    assert np.array_equal(result, expected)

    result = sliding_window(a, ws, ss=2)
    expected = np.array(
        [
            [0, 1, 2],
            [2, 3, 4],
            [4, 5, 6],
            [6, 7, 8],
        ]
    )
    assert np.array_equal(result, expected)

    # ss = ws, in this case (ss=3)
    result = sliding_window(a, ws, ss=None)
    expected = np.array(
        [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
        ]
    )
    assert np.array_equal(result, expected)

    # ss is greater than ws
    result = sliding_window(a, ws, ss=5)
    expected = np.array(
        [
            [0, 1, 2],
            [5, 6, 7],
        ]
    )
    assert np.array_equal(result, expected)

    # ss is greater than array size
    # Then, only one window should be returned
    result = sliding_window(a, ws, ss=100)
    expected = np.array(
        [
            [0, 1, 2],
        ]
    )
    assert np.array_equal(result, expected)

    # A 1D tuple is passed as ws and ss instead of an int
    result = sliding_window(a, ws=(ws,), ss=(100,))
    assert np.array_equal(result, expected)


def test_sliding_window_error():
    a = np.arange(10)

    # Window size is greater than the array size
    with pytest.raises(ValueError):
        sliding_window(a, ws=11, ss=1)

    # Window size is greater than the array size (and ss=ws)
    with pytest.raises(ValueError):
        sliding_window(a, ws=11, ss=None)

    # Window size is 0 (invalid)
    with pytest.raises(ValueError):
        sliding_window(a, ws=0, ss=1)

    # ss is 0 (invalid)
    with pytest.raises(ValueError):
        sliding_window(a, ws=5, ss=0)

    # ws and ss are int (will be normalized as a 1-element tuple), but a is 2D
    a = np.arange(10).reshape(2, 5)
    with pytest.raises(ValueError):
        sliding_window(a, ws=1, ss=1)

    # ws has 3 elements and array is 2D
    with pytest.raises(ValueError):
        sliding_window(a, ws=(1, 1, 1), ss=(1,))

    # ws has 2 elements and array is 1D
    a = np.arange(10)
    with pytest.raises(ValueError):
        sliding_window(a, ws=(1, 1), ss=(1,))

    # ss has 3 elements and array is 1D
    with pytest.raises(ValueError):
        sliding_window(a, ws=1, ss=(1, 1, 1))

    # ws has one dim greater than a
    a = np.arange(10)
    with pytest.raises(ValueError):
        sliding_window(a, ws=(11,), ss=None)


def test_opp_sliding_window():
    data_x = np.random.rand(100, 6)
    data_y = np.random.randint(0, 5, 100)
    ws = 10
    ss = 5
    data_x_windowed, data_y_windowed = opp_sliding_window(data_x, data_y, ws, ss)
    assert data_x_windowed.shape == (19, 10, 6)
    assert data_y_windowed.shape == (19,)
    assert data_x_windowed.dtype == np.float32
    assert data_y_windowed.dtype == np.uint8

    assert isinstance(data_x_windowed, np.ndarray)
    assert isinstance(data_y_windowed, np.ndarray)


def test_hardatasetcpc_init(sample_data):
    dataset = HARDatasetCPC(
        data_path=sample_data,
        input_size=6,
        window=10,
        overlap=5,
        phase="train",
        label="activity code",
    )
    assert len(dataset) > 0
    assert dataset.data.shape[1] == 6
    assert dataset.data.shape[2] == 10


def test_hardatasetcpc_getitem(sample_data):
    dataset = HARDatasetCPC(
        data_path=sample_data,
        input_size=6,
        window=10,
        overlap=5,
        phase="train",
        label="activity code",
    )
    data, label = dataset[0]
    assert data.shape == (6, 10)
    np.testing.assert_allclose(data, dataset.data[0])
    np.testing.assert_allclose(label, dataset.labels[0])


def test_hardatasetcpc_len(sample_data):
    dataset = HARDatasetCPC(
        data_path=sample_data,
        input_size=6,
        window=10,
        overlap=5,
        phase="train",
        label="activity code",
    )
    assert len(dataset) == dataset.data.shape[0]


def test_hardatasetcpc_return_index_as_label(sample_data):
    dataset = HARDatasetCPC(
        data_path=sample_data,
        input_size=6,
        window=10,
        overlap=5,
        phase="train",
        label="return_index_as_label",
    )
    data, label = dataset[0]
    assert data.shape == (6, 10)
    assert label.shape == ()


def test_hardatasetcpc_transpose_data_true(sample_data):
    dataset = HARDatasetCPC(
        data_path=sample_data,
        input_size=6,
        window=10,
        overlap=5,
        phase="train",
        transpose_data=True,
        label="activity code",
    )
    data, _ = dataset[0]
    assert data.shape == (6, 10)


def test_hardatasetcpc_transpose_data_false(sample_data):
    dataset = HARDatasetCPC(
        data_path=sample_data,
        input_size=6,
        window=10,
        overlap=5,
        phase="train",
        transpose_data=False,
        label="activity code",
    )
    data, _ = dataset[0]
    assert data.shape == (10, 6)


def test_hardatasetcpc_use_train_as_val(sample_data):
    # use_train_as_val = True
    dataset = HARDatasetCPC(
        data_path=sample_data,
        input_size=6,
        window=10,
        overlap=5,
        phase="val",
        use_train_as_val=True,
        label="activity code",
    )

    np.testing.assert_allclose(
        dataset.data_raw["val"]["data"], dataset.data_raw["train"]["data"]
    )


def test_hardatasetcpc_use_val_with_train(sample_data):
    dataset = HARDatasetCPC(
        data_path=sample_data,
        input_size=6,
        window=10,
        overlap=5,
        phase="train",
        use_val_with_train=True,
        label="activity code",
    )

    # Directly load raw train and val data
    df_train = pd.read_csv(sample_data / "train/sample.csv")
    df_val = pd.read_csv(sample_data / "val/sample.csv")

    # Manually concatenate data as expected
    expected_data = np.concatenate(
        [df_train.iloc[:, :-1].values, df_val.iloc[:, :-1].values], axis=0
    )
    expected_labels = np.concatenate(
        [df_train.iloc[:, -1].values, df_val.iloc[:, -1].values], axis=0
    )

    # Final comparison
    np.testing.assert_allclose(dataset.data_raw["train"]["data"], expected_data)
    np.testing.assert_allclose(dataset.data_raw["train"]["labels"], expected_labels)
