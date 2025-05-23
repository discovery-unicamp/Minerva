import numpy as np
import pandas as pd
import pytest
from minerva.data.readers.tabular_reader import TabularReader


@pytest.fixture
def sample_df():
    return pd.DataFrame(
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


def test_len(sample_df):
    reader = TabularReader(sample_df, ["accel-x-0"])
    assert len(reader) == 10

    reader = TabularReader(sample_df, ["accel-x-0", "gyro-x-0"])
    assert len(reader) == 10


def test_column_pattern_match(sample_df):
    index = 8

    reader = TabularReader(sample_df, ["accel-x-.*", "gyro-x-.*"])
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
    expected = sample_df.loc[index, expected_cols].to_numpy()
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (8,)

    # Inverting the pattern
    reader = TabularReader(sample_df, ["gyro-x-.*", "accel-x-.*"])
    result = reader[index]
    expected_cols = [
        "gyro-x-0",
        "gyro-x-1",
        "gyro-x-2",
        "gyro-x-3",
        "accel-x-0",
        "accel-x-1",
        "accel-x-2",
        "accel-x-3",
    ]
    expected = sample_df.loc[index, expected_cols].to_numpy()
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (8,)

    # all accel/gyro columns
    reader = TabularReader(sample_df, ["accel-.*", "gyro-.*"])
    result = reader[index]
    expected_cols = [
        "accel-x-0",
        "accel-x-1",
        "accel-x-2",
        "accel-x-3",
        "accel-y-0",
        "accel-y-1",
        "accel-y-2",
        "accel-y-3",
        "gyro-x-0",
        "gyro-x-1",
        "gyro-x-2",
        "gyro-x-3",
    ]
    expected = sample_df.loc[index, expected_cols].to_numpy()
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (12,)

    # all accel/gyro columns inverted order
    reader = TabularReader(sample_df, ["gyro.*", "accel.*"])
    result = reader[index]
    expected_cols = [
        "gyro-x-0",
        "gyro-x-1",
        "gyro-x-2",
        "gyro-x-3",
        "accel-x-0",
        "accel-x-1",
        "accel-x-2",
        "accel-x-3",
        "accel-y-0",
        "accel-y-1",
        "accel-y-2",
        "accel-y-3",
    ]
    expected = sample_df.loc[index, expected_cols].to_numpy()
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (12,)

    # all accel-y/gyro/accel-x columns
    reader = TabularReader(sample_df, ["accel-y-.*", "gyro-.*", "accel-x-.*"])
    result = reader[index]
    expected_cols = [
        "accel-y-0",
        "accel-y-1",
        "accel-y-2",
        "accel-y-3",
        "gyro-x-0",
        "gyro-x-1",
        "gyro-x-2",
        "gyro-x-3",
        "accel-x-0",
        "accel-x-1",
        "accel-x-2",
        "accel-x-3",
    ]
    expected = sample_df.loc[index, expected_cols].to_numpy()
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (12,)

    # Mixed pattern and names
    cols = [
        "accel-y-1",
        "gyro-x-2",
        "accel-y-0",
        "accel-y-3",
        "accel-y-2",
        "gyro-x-1",
        "gyro-x-3",
        "accel-x-*",
    ]
    expected_cols = [
        "accel-y-1",
        "gyro-x-2",
        "accel-y-0",
        "accel-y-3",
        "accel-y-2",
        "gyro-x-1",
        "gyro-x-3",
        "accel-x-0",
        "accel-x-1",
        "accel-x-2",
        "accel-x-3",
    ]
    reader = TabularReader(sample_df, cols)
    result = reader[index]
    expected = sample_df.loc[index, expected_cols].to_numpy()
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (11,)


def test_data_reshape(sample_df):
    index = 7

    reader = TabularReader(sample_df, ["accel-x-.*", "gyro-x-.*"], data_shape=(2, -1))
    result = reader[index]
    assert result.shape == (2, 4)
    np.testing.assert_array_equal(
        result,
        np.array(
            [
                sample_df.loc[
                    index, ["accel-x-0", "accel-x-1", "accel-x-2", "accel-x-3"]
                ].to_list(),
                sample_df.loc[
                    index, ["gyro-x-0", "gyro-x-1", "gyro-x-2", "gyro-x-3"]
                ].to_list(),
            ]
        ),
    )

    reader = TabularReader(sample_df, ["accel-x-.*", "gyro-x-.*"], data_shape=(-1, 2))
    result = reader[index]
    assert result.shape == (4, 2)
    np.testing.assert_array_equal(
        result,
        np.array(
            [
                sample_df.loc[index, ["accel-x-0", "accel-x-1"]].to_list(),
                sample_df.loc[index, ["accel-x-2", "accel-x-3"]].to_list(),
                sample_df.loc[index, ["gyro-x-0", "gyro-x-1"]].to_list(),
                sample_df.loc[index, ["gyro-x-2", "gyro-x-3"]].to_list(),
            ]
        ),
    )


def test_single_column_return(sample_df):
    index = 7

    reader = TabularReader(sample_df, "label")
    result = reader[index]
    assert isinstance(result, str)
    assert result == f"label-{index}"


def test_data_type_casting(sample_df):
    reader = TabularReader(sample_df, ["accel-x-.*"], cast_to="float32")
    result = reader[0]
    assert result.dtype == np.float32


def test_index_access_out_of_bounds_raises(sample_df):
    reader = TabularReader(sample_df, ["accel-x-.*"])
    with pytest.raises(IndexError):
        _ = reader[100]


def test_invalid_pattern_raises(sample_df):
    with pytest.raises(ValueError):
        reader = TabularReader(sample_df, ["invalid-col"])
        _ = reader[0]

    with pytest.raises(ValueError):
        reader = TabularReader(sample_df, ["accel-x-0", "invalid-col"])
        _ = reader[0]

    with pytest.raises(ValueError):
        reader = TabularReader(sample_df, ["accel-x-0", "accel-y-*", "invalid-col"])
        _ = reader[0]

    with pytest.raises(ValueError):
        reader = TabularReader(sample_df, ["accel-*", "invalid-pattern-*"])
        _ = reader[0]
