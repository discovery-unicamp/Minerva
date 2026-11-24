import numpy as np
import pytest
import torch

from minerva.transforms import (
    RandomCropTransform,
    RandomFlipTransform,
    RandomGrayScaleTransform,
    RandomResizedCropTransform,
    RandomResizeTransform,
    RandomRotationTransform,
    RandomSolarizeTransform,
    GaussianBlurTransform,
    RandomVerticalPatchSwapTransform,
)


def _seg_sample(h=40, w=40):
    image = np.random.rand(3, h, w).astype(np.float32)
    return {"image": image, "label": image[0].copy()}  # label mirrors channel 0


def _in_sync(out):
    return np.array_equal(out["image"][0], out["label"])


def test_random_flip_prob_extremes():
    x = np.random.rand(10, 20)
    always = RandomFlipTransform(["x"], 0, prob=1.0)({"x": x})["x"]
    never = RandomFlipTransform(["x"], 0, prob=0.0)({"x": x})["x"]
    np.testing.assert_array_equal(always, np.flip(x, axis=0))
    np.testing.assert_array_equal(never, x)


def test_random_flip_multiple_axes_prob_1():
    x = np.random.rand(4, 5, 6)
    out = RandomFlipTransform(["x"], [0, 2], prob=1.0)({"x": x})["x"]
    np.testing.assert_array_equal(out, np.flip(x, axis=[0, 2]))


def test_random_flip_invalid_prob():
    with pytest.raises(AssertionError):
        RandomFlipTransform(["x"], 0, prob=1.5)


def test_random_flip_is_reproducible_with_torch_seed():
    x = np.random.rand(10, 20)
    results = []
    for _ in range(2):
        torch.manual_seed(7)
        results.append([RandomFlipTransform(["x"], 0)({"x": x})["x"] for _ in range(8)])
    for a, b in zip(*results):
        np.testing.assert_array_equal(a, b)


def test_random_flip_keeps_image_and_label_in_sync():
    t = RandomFlipTransform(["image", "label"], [-1, -2], prob=0.5)
    for _ in range(30):
        assert _in_sync(t(_seg_sample(8, 8)))


def test_random_crop_shape_and_sync():
    t = RandomCropTransform(["image", "label"], (30, 30))
    for _ in range(20):
        out = t(_seg_sample())
        assert out["image"].shape == (3, 30, 30) and out["label"].shape == (30, 30)
        assert _in_sync(out)


def test_random_crop_varies_across_calls():
    t = RandomCropTransform(["x"], (5, 5))
    base = np.arange(400, dtype=np.float32).reshape(20, 20)
    crops = {t({"x": base})["x"].tobytes() for _ in range(20)}
    assert len(crops) > 1


def test_random_grayscale_and_solarize_prob_extremes():
    x = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)
    g = RandomGrayScaleTransform(["x"], prob=1.0, method="average")({"x": x})["x"]
    assert np.all(g[..., 0] == g[..., 1])
    assert np.array_equal(RandomGrayScaleTransform(["x"], prob=0.0)({"x": x})["x"], x)
    s = RandomSolarizeTransform(["x"], threshold=100, prob=1.0)({"x": x})["x"]
    assert not np.array_equal(s, x)
    assert np.array_equal(RandomSolarizeTransform(["x"], prob=0.0)({"x": x})["x"], x)


def test_random_rotation_prob_and_label_modes():
    image = np.random.randint(0, 256, size=(50, 50, 3), dtype=np.uint8)
    label = np.random.randint(0, 4, size=(50, 50), dtype=np.uint8)
    out = RandomRotationTransform(
        ["image", "label"], 45, prob=1.0, mode=["bilinear", "nearest"]
    )({"image": image, "label": label})
    assert out["image"].shape == image.shape
    assert set(np.unique(out["label"])) <= set(np.unique(label))
    same = RandomRotationTransform(["image"], 45, prob=0.0)({"image": image})["image"]
    np.testing.assert_array_equal(same, image)


def test_random_resize_height_width_not_swapped():
    t = RandomResizeTransform(["x"], target_scale=(20, 40), ratio_range=(1.0, 1.0))
    out = t({"x": np.random.rand(10, 10)})["x"]
    assert out.shape == (20, 40)  # (height, width)


def test_random_resized_crop_sync_and_label_values():
    label = np.random.randint(0, 5, size=(64, 64), dtype=np.uint8)
    t = RandomResizedCropTransform(
        ["image", "label"], size=(32, 32), mode=["bilinear", "nearest"]
    )
    for _ in range(10):
        image = np.broadcast_to(label.astype(np.float32), (3, 64, 64)).copy()
        out = t({"image": image, "label": label})
        assert out["image"].shape == (3, 32, 32) and out["label"].shape == (32, 32)
        assert set(np.unique(out["label"])) <= set(np.unique(label))


def test_random_resized_crop_same_box_for_image_and_label():
    # image holds the same value as the label, so identical boxes keep them equal
    # under nearest resizing of both
    label = np.random.randint(0, 255, size=(64, 64)).astype(np.float32)
    t = RandomResizedCropTransform(
        ["image", "label"], size=(16, 16), mode=["nearest", "nearest"]
    )
    for _ in range(10):
        out = t({"image": label[None].copy(), "label": label.copy()})
        np.testing.assert_array_equal(out["image"][0], out["label"])


def test_gaussian_blur():
    x = np.random.rand(3, 16, 16).astype(np.float32)
    blurred = GaussianBlurTransform(["x"], kernel_size=5, prob=1.0)({"x": x})["x"]
    assert blurred.shape == x.shape and not np.array_equal(blurred, x)
    np.testing.assert_array_equal(
        GaussianBlurTransform(["x"], kernel_size=5, prob=0.0)({"x": x})["x"], x
    )
    with pytest.raises(ValueError):
        GaussianBlurTransform(["x"], kernel_size=4)


def test_random_vertical_patch_swap_keeps_image_and_label_in_sync():
    label = np.arange(8 * 6, dtype=np.float32).reshape(8, 6)
    data = {"image": label[None].copy(), "label": label.copy(), "meta": "keep"}
    out = RandomVerticalPatchSwapTransform(
        ["image", "label"], patch_height=2, prob=1.0, seed=42
    )(data)
    np.testing.assert_array_equal(out["image"][0], out["label"])
    assert out["meta"] == "keep"


def test_random_vertical_patch_swap_prob_zero_is_identity():
    image = np.arange(8 * 6).reshape(1, 8, 6)
    label = image[0].copy()
    out = RandomVerticalPatchSwapTransform(
        ["image", "label"], patch_height=2, prob=0.0
    )({"image": image.copy(), "label": label.copy()})
    np.testing.assert_array_equal(out["image"], image)
    np.testing.assert_array_equal(out["label"], label)
