import numpy as np

from minerva.data.readers import (
    PatchedArrayReader,
    LazyPaddedPatchedArrayReader,
    NumpyArrayReader,
)
import pytest


@pytest.mark.parametrize(
    "reader_class", [PatchedArrayReader, LazyPaddedPatchedArrayReader]
)
def test_patched_array_reader_no_stride_1(reader_class):
    data = np.arange(100).reshape(10, 10)
    reader = reader_class(
        data,
        data_shape=(5, 5),
    )

    assert len(reader) == 4, "The number of patches is incorrect"
    assert reader[0].shape == (
        5,
        5,
    ), "The shape of the first patch is incorrect"
    assert np.all(
        reader[0]
        == np.array(
            [
                [0, 1, 2, 3, 4],
                [10, 11, 12, 13, 14],
                [20, 21, 22, 23, 24],
                [30, 31, 32, 33, 34],
                [40, 41, 42, 43, 44],
            ]
        )
    ), "The content of the first patch is incorrect"

    assert np.all(
        reader[3]
        == np.array(
            [
                [55, 56, 57, 58, 59],
                [65, 66, 67, 68, 69],
                [75, 76, 77, 78, 79],
                [85, 86, 87, 88, 89],
                [95, 96, 97, 98, 99],
            ]
        )
    ), "The content of the last patch is incorrect"


@pytest.mark.parametrize(
    "reader_class", [PatchedArrayReader, LazyPaddedPatchedArrayReader]
)
def test_patched_array_reader_stride(reader_class):
    data = np.arange(100).reshape(10, 10)
    reader = reader_class(data, data_shape=(5, 5), stride=(2, 5))

    assert len(reader) == 6, "The number of patches is incorrect"
    assert np.all(
        reader[0]
        == np.array(
            [
                [0, 1, 2, 3, 4],
                [10, 11, 12, 13, 14],
                [20, 21, 22, 23, 24],
                [30, 31, 32, 33, 34],
                [40, 41, 42, 43, 44],
            ]
        )
    ), "The content of the first patch is incorrect"

    assert np.all(
        reader[2]
        == np.array(
            [
                [20, 21, 22, 23, 24],
                [30, 31, 32, 33, 34],
                [40, 41, 42, 43, 44],
                [50, 51, 52, 53, 54],
                [60, 61, 62, 63, 64],
            ]
        )
    ), "The content of the third patch is incorrect"


@pytest.mark.parametrize(
    "reader_class", [PatchedArrayReader, LazyPaddedPatchedArrayReader]
)
def test_patched_array_reader_stride_and_pad(reader_class):
    data = np.arange(100).reshape(10, 10)
    reader = reader_class(
        data,
        data_shape=(5, 5),
        stride=(2, 5),
        pad_width=((2, 2), (2, 2)),
        pad_mode="constant",
        pad_kwargs={"constant_values": -1},
    )
    assert len(reader) == 10, "The number of patches is incorrect"
    assert np.all(
        reader[0]
        == np.array(
            [
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1],
                [-1, -1, 0, 1, 2],
                [-1, -1, 10, 11, 12],
                [-1, -1, 20, 21, 22],
            ]
        )
    ), "The content of the first patch is incorrect"

    assert np.all(
        reader[-1]
        == np.array(
            [
                [63, 64, 65, 66, 67],
                [73, 74, 75, 76, 77],
                [83, 84, 85, 86, 87],
                [93, 94, 95, 96, 97],
                [-1, -1, -1, -1, -1],
            ]
        )
    ), "The content of the last patch is incorrect"


@pytest.mark.parametrize(
    "reader_class", [PatchedArrayReader, LazyPaddedPatchedArrayReader]
)
def test_patched_array_reader_index_bounds(reader_class):
    data = np.arange(100).reshape(10, 10)

    # Define index bounds to select a 5x5 subarray starting at (2, 2)
    index_bounds = ((2, 2), (7, 7))
    reader = reader_class(
        data,
        data_shape=(1, 1),  # Smaller patch size for the subarray
        stride=(1, 1),  # Stride of 1 for more patches
        index_bounds=index_bounds,
    )

    # Verify the subarray was correctly selected
    expected_subarray = data[2:7, 2:7]
    assert reader.data.shape == (5, 5), "The subarray shape is incorrect"
    assert np.all(reader.data == expected_subarray), "The subarray content is incorrect"

    # Verify patch
    # generation works correctly on the subarray
    assert len(reader) == 25, "The number of patches is incorrect"


def test_loading_numpy_array_reader(tmp_path):
    data = np.arange(100).reshape(10, 10)

    # ---------- numpy array file ---------
    reader = NumpyArrayReader(data, data_shape=(1, 10))
    assert len(reader) == 10, "The number of patches is incorrect"
    assert reader[0].shape == (1, 10), "The shape of patch is incorrect"

    # --------- .npy file ---------

    # Save the array to a .npy file
    # and create a NumpyArrayReader instance
    array_file = tmp_path / "test.npy"
    np.save(array_file, data)
    reader = NumpyArrayReader(array_file, data_shape=(1, 10))
    assert len(reader) == 10, "The number of patches is incorrect"
    assert reader[0].shape == (1, 10), "The shape of patch is incorrect"

    # ---------- .npz file ---------
    array_file = tmp_path / "test.npz"
    np.savez(array_file, data=data)
    reader = NumpyArrayReader(array_file, data_shape=(1, 10), npz_key="data")
    assert len(reader) == 10, "The number of patches is incorrect"
    assert reader[0].shape == (1, 10), "The shape of patch is incorrect"

    with pytest.raises(KeyError):
        reader = NumpyArrayReader(array_file, data_shape=(1, 10), npz_key="invalid_key")


def test_loading_numpy_array_reader_invalid_file(tmp_path):
    # Create a temporary file with invalid content
    invalid_file = tmp_path / "invalid.npy"
    with pytest.raises(FileNotFoundError):
        NumpyArrayReader(invalid_file, data_shape=(1, 10))

    # Write invalid content to the file
    with open(invalid_file, "w") as f:
        f.write("This is not a valid numpy array file.")

    # Attempt to load the invalid file
    with pytest.raises(Exception):
        NumpyArrayReader(invalid_file, data_shape=(1, 10))

    invalid_file = tmp_path / "test.invalid"
    # Write invalid content to the file
    with open(invalid_file, "w") as f:
        f.write("This is not a valid numpy array file.")
    with pytest.raises(ValueError):
        NumpyArrayReader(invalid_file, data_shape=(1, 10))
