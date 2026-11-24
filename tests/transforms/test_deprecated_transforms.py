import numpy as np
import pytest
import torch

from minerva.transforms import SwapAxesTransform


def test_transpose_numpy():
    x = np.array([[1, 2, 3], [4, 5, 6]])
    transform = SwapAxesTransform(['x'], 0, 1)

    result = transform({'x': x})['x']

    np.testing.assert_array_equal(result, [[1, 4], [2, 5], [3, 6]])


def test_transpose_tensor():
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    transform = SwapAxesTransform(['x'], 0, 1)

    result = transform({'x': x})['x']

    torch.testing.assert_close(result, torch.tensor([[1, 4], [2, 5], [3, 6]]))


@pytest.mark.parametrize("dims", [(-1, 0), (0, -1)])
def test_transpose_negative_dimension(dims):
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        SwapAxesTransform(['x'], *dims)


def test_transpose_invalid_input():
    transform = SwapAxesTransform(['x'], 0, 1)

    with pytest.raises(TypeError, match="numpy array or a Pytorch tensor"):
        transform({"x": "invalid"})


def test_transpose_dimension_out_of_range():
    transform = SwapAxesTransform(['x'], 0, 2)
    x = torch.zeros(2, 3)

    with pytest.raises(ValueError, match="must be less than 2"):
        transform({'x': x})


# ---- deprecation warnings ---------------------------------------------------
def test_deprecated_transforms_warn_and_still_work():
    from minerva.transforms import ClassRatioCropTransform, PadCropTransform

    with pytest.warns(FutureWarning, match="next major release"):
        pad_crop = PadCropTransform(["x"], 6, 6, seed=0)
    assert pad_crop({"x": np.ones((4, 8, 3))})["x"].shape == (6, 6, 3)

    # deterministic: same output every call, centered crop (no random position)
    x = np.arange(100, dtype="float32").reshape(10, 10)
    outs = [pad_crop({"x": x.copy()})["x"] for _ in range(5)]
    assert all(np.array_equal(outs[0], o) for o in outs)
    with pytest.warns(FutureWarning):
        crop4 = PadCropTransform(["x"], 4, 4)
    np.testing.assert_array_equal(crop4({"x": x.copy()})["x"], x[3:7, 3:7])

    with pytest.warns(FutureWarning, match="RandomCropTransform"):
        ratio = ClassRatioCropTransform(["x"], "x", (4, 4), seed=0)
    assert ratio({"x": np.arange(64).reshape(8, 8) % 3})["x"].shape == (4, 4)

    with pytest.warns(FutureWarning, match="TransposeTransform"):
        SwapAxesTransform(["x"], 0, 1)


def test_old_module_paths_still_import():
    from minerva.transforms.context_transform import ClassRatioCropTransform
    from minerva.transforms.transpose_transform import SwapAxesTransform as S

    assert S is SwapAxesTransform and ClassRatioCropTransform
