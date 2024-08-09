import numpy as np

from minerva.data.readers import MultiReader, PatchedArrayReader


def test_multi_reader_identity():

    reader1 = PatchedArrayReader(
        np.arange(15**2).reshape(1, 15, 15),
        data_shape=(1, 5, 5),
    )

    reader2 = PatchedArrayReader(
        np.arange(10**2).reshape(1, 10, 10), data_shape=(1, 5, 5)
    )

    multireader = MultiReader([reader1, reader2])

    assert len(multireader) == min(
        len(reader1), len(reader2)
    ), "Reader has incorrect length"

    assert np.all(
        multireader[0] == np.stack([reader1[0], reader2[0]])
    ), "Reader's first element is incorrect"

    assert np.all(
        multireader[len(multireader) - 1]
        == np.stack([reader1[len(multireader) - 1], reader2[len(multireader) - 1]])
    ), "Reader's last element is incorrect"


def test_multi_reader_squeeze():

    reader1 = PatchedArrayReader(
        np.arange(15**2).reshape(1, 15, 15),
        data_shape=(1, 5, 5),
    )

    reader2 = PatchedArrayReader(
        np.arange(10**2).reshape(1, 10, 10), data_shape=(1, 5, 5)
    )

    multireader = MultiReader([reader1, reader2], np.squeeze)

    assert len(multireader) == min(
        len(reader1), len(reader2)
    ), "Reader has incorrect length"

    assert np.all(
        multireader[0] == np.stack([np.squeeze(reader1[0]), np.squeeze(reader2[0])])
    ), "Reader's first element is incorrect"

    assert np.all(
        multireader[len(multireader) - 1]
        == np.stack(
            [
                np.squeeze(reader1[len(multireader) - 1]),
                np.squeeze(reader2[len(multireader) - 1]),
            ]
        )
    ), "Reader's last element is incorrect"
