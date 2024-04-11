import numpy as np
from minerva.data.readers.patched_array_reader import PatchedArrayReader


def test_patched_array_reader_no_stride_1():
    data = np.arange(100).reshape(10, 10)
    reader = PatchedArrayReader(
        data,
        data_shape=(5, 5),
    )

    assert len(reader) == 4, "The number of patches is incorrect"
    assert reader[0].shape == (5, 5), "The shape of the first patch is incorrect"
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
    
def test_patched_array_reader_stride():
    data = np.arange(100).reshape(10, 10)
    reader = PatchedArrayReader(
        data,
        data_shape=(5, 5),
        stride=(2, 5)
    )

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


def test_patched_array_reader_stride_and_pad():
    data = np.arange(100).reshape(10, 10)
    reader = PatchedArrayReader(
        data,
        data_shape=(5, 5),
        stride=(2, 5),
        pad_width=2,
        pad_mode="constant",
        pad_kwargs={"constant_values": -1}
    )
    assert len(reader) == 10, "The number of patches is incorrect"
    assert np.all(
        reader[0]
        == np.array(
            [
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1],
                [-1, -1,  0,  1,  2],
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
                [-1, -1, -1, -1, -1]
            ]
        )
    ), "The content of the last patch is incorrect"
