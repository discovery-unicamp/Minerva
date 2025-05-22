import numpy as np
import pandas as pd
import pytest
from minerva.data.readers.csv_reader import CSVReader


@pytest.fixture
def sample_csv(tmp_path):
    df = pd.DataFrame(
        {
            "accel-x-0": np.arange(10),
            "accel-x-1": np.arange(10) + 10,
            "accel-x-2": np.arange(10) + 100,
            "accel-x-3": np.arange(10) + 1000,
            "accel-y-0": np.arange(10),
            "accel-y-1": np.arange(10) * 2,
            "accel-y-2": np.arange(10) * 3,
            "accel-y-3": np.arange(10) * 4,
            "gyro-x-0": np.arange(10) - 10,
            "gyro-x-1": np.arange(10) - 20,
            "gyro-x-2": np.arange(10) - 30,
            "gyro-x-3": np.arange(10) - 40,
            "label": [f"label-{i}" for i in range(10)],
        }
    )
    csv_path = tmp_path / "sample.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_len(sample_csv):
    reader = CSVReader(sample_csv, ["accel-x-0"])
    assert len(reader) == 10

    reader = CSVReader(sample_csv, ["accel-x-0", "gyro-x-0"])
    assert len(reader) == 10


def test_using_dataframe(sample_csv):
    df = pd.read_csv(sample_csv)
    reader = CSVReader(df, ["accel-x-0"])
    assert len(reader) == 10

    reader = CSVReader(df, ["accel-x-0", "gyro-x-0"])
    assert len(reader) == 10


def test_pattern_match(sample_csv):
    index = 8

    reader = CSVReader(sample_csv, ["accel-x-.*", "gyro-x-.*"])
    result = reader[index]
    expected_cols = [
        "accel-x-0",
        "accel-x-1",
        "accel-x-2",
        "accel-x-3",
        "gyro-x-0",
        "gyro-x-1",
        "gyro-x-2",
        "gyro-x-3",
    ]
    expected = pd.read_csv(sample_csv).loc[index, expected_cols].to_numpy()  # type: ignore
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (8,)
