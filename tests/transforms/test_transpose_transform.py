import numpy as np
import pytest
import torch

from minerva.transforms.transpose_transform import TransposeTransform


def test_transpose_numpy():
    x = np.array([[1, 2, 3], [4, 5, 6]])
    transform = TransposeTransform(0, 1)

    result = transform(x)

    np.testing.assert_array_equal(result, [[1, 4], [2, 5], [3, 6]])


def test_transpose_tensor():
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    transform = TransposeTransform(0, 1)

    result = transform(x)

    torch.testing.assert_close(result, torch.tensor([[1, 4], [2, 5], [3, 6]]))


@pytest.mark.parametrize("dims", [(-1, 0), (0, -1)])
def test_transpose_negative_dimension(dims):
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        TransposeTransform(*dims)


def test_transpose_invalid_input():
    transform = TransposeTransform(0, 1)

    with pytest.raises(TypeError, match="numpy array or a Pytorch tensor"):
        transform("invalid")


def test_transpose_dimension_out_of_range():
    transform = TransposeTransform(0, 2)
    x = torch.zeros(2, 3)

    with pytest.raises(ValueError, match="must be less than 2"):
        transform(x)
