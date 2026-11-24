import numpy as np
import pytest
import torch

from minerva.transforms import TransposeTransform


def test_transpose_numpy_swap_two_axes():
    x = np.array([[1, 2, 3], [4, 5, 6]])
    result = TransposeTransform(["x"], (1, 0))({"x": x})["x"]
    np.testing.assert_array_equal(result, [[1, 4], [2, 5], [3, 6]])


def test_transpose_tensor():
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    result = TransposeTransform(["x"], (1, 0))({"x": x})["x"]
    torch.testing.assert_close(result, torch.tensor([[1, 4], [2, 5], [3, 6]]))


def test_hwc_to_chw_for_numpy_and_tensor():
    x = np.zeros((4, 5, 3))
    assert TransposeTransform(["x"], (2, 0, 1))({"x": x})["x"].shape == (3, 4, 5)
    assert TransposeTransform(["x"], (2, 0, 1))({"x": torch.zeros(4, 5, 3)})["x"].shape == (3, 4, 5)


def test_transpose_invalid_input():
    with pytest.raises(TypeError, match="numpy array or a torch tensor"):
        TransposeTransform(["x"], (1, 0))({"x": "invalid"})


def test_transpose_axes_out_of_range():
    with pytest.raises(Exception):
        TransposeTransform(["x"], (0, 2))({"x": torch.zeros(2, 3)})
