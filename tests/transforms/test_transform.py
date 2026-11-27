import numpy as np
import pytest

from minerva.transforms import (
    AmplitudeScalingTransform,
    CastToTransform,
    ColorJitterTransform,
    CropTransform,
    FlipTransform,
    GradientTransform,
    GrayScaleTransform,
    IdentityTransform,
    IndexerTransform,
    NormalizeTransform,
    PaddingTransform,
    PerlinMaskerTransform,
    RandomCropTransform,
    RepeatTransform,
    ReshapeTransform,
    ResizeTransform,
    RotationTransform,
    SolarizeTransform,
    SqueezeTransform,
    TransposeTransform,
    UnsqueezeTransform,
    _TransformsPipeline,
    VerticalPatchSwapTransform,
)


def test_pipeline_and_add_radd():
    x = np.random.rand(10, 20)
    p1 = _TransformsPipeline([FlipTransform(["x"], 0), FlipTransform(["x"], 1)])
    p2 = _TransformsPipeline([FlipTransform(["x"], [0, 1])])

    assert len((p1 + p2).transforms) == 3
    assert len((p1 + FlipTransform(["x"], [0, 1])).transforms) == 3
    assert len((FlipTransform(["x"], 0) + p2).transforms) == 2
    out = (p1 + p2)({"x": x})
    assert out["x"].shape == x.shape


def test_pipeline_str():
    p = _TransformsPipeline([IdentityTransform()])
    assert str(p).startswith("_TransformsPipeline(transforms=[IdentityTransform(")


def test_flip_single_and_multiple_axes():
    x = np.random.rand(4, 5, 6)
    np.testing.assert_array_equal(
        FlipTransform(["x"], 1)({"x": x})["x"], np.flip(x, axis=1)
    )
    np.testing.assert_array_equal(
        FlipTransform(["x"], [0, 2])({"x": x})["x"], np.flip(x, axis=[0, 2])
    )


def test_flip_invalid_axes():
    with pytest.raises(AssertionError):
        FlipTransform(["x"], [0, 1, 2])({"x": np.random.rand(10, 20)})


@pytest.mark.parametrize(
    "kwargs", [{"octaves": 0}, {"octaves": -1}, {"octaves": 1, "scale": 0}]
)
def test_perlin_masker_invalid(kwargs):
    with pytest.raises(ValueError):
        PerlinMaskerTransform(["x"], **kwargs)


def test_perlin_masker_shares_noise_across_keys():
    x = np.ones((8, 8))
    out = PerlinMaskerTransform(["a", "b"], octaves=2)({"a": x, "b": x.copy()})
    np.testing.assert_array_equal(out["a"], out["b"])
    assert set(np.unique(out["a"])) <= {0.0, 1.0}


def test_transpose():
    x = np.random.rand(2, 3, 4)
    out = TransposeTransform(["x"], (2, 0, 1))({"x": x})["x"]
    assert out.shape == (4, 2, 3)


def test_squeeze_unsqueeze():
    x = np.random.rand(1, 3)
    assert SqueezeTransform(["x"], 0)({"x": x})["x"].shape == (3,)
    assert UnsqueezeTransform(["x"], 1)({"x": x})["x"].shape == (1, 1, 3)
    with pytest.raises(ValueError):
        SqueezeTransform(["x"], 1)({"x": x})


def test_unsqueeze_only_listed_keys():
    data = {"image": np.zeros((3, 4)), "label": np.zeros((4,), dtype=np.uint8)}
    out = UnsqueezeTransform(["label"], 0)(data)
    assert out["label"].shape == (1, 4)
    assert out["image"].shape == (3, 4)


def test_normalize():
    x = np.random.rand(10, 20, 1)
    out = NormalizeTransform(["x"], [np.mean(x)], [np.std(x)])({"x": x.copy()})
    assert np.allclose(out["x"].mean(), 0) and np.allclose(out["x"].std(), 1)


def test_amplitude_scaling_scales_listed_signal_and_preserves_mask():
    signal = np.arange(12, dtype=np.float32).reshape(3, 4)
    mask = np.array([[0, 1], [2, 255]], dtype=np.uint8)
    data = {"signal": signal.copy(), "mask": mask.copy(), "metadata": "keep"}
    out = AmplitudeScalingTransform(["signal"], 2.5)(data)
    np.testing.assert_array_equal(out["signal"], signal * 2.5)
    np.testing.assert_array_equal(out["mask"], mask)
    assert out["metadata"] == "keep"


def test_amplitude_scaling_rejects_non_finite_factor():
    with pytest.raises(ValueError, match="finite"):
        AmplitudeScalingTransform(["signal"], float("inf"))


def test_amplitude_scaling_normalizes_integer_factor_before_multiplication():
    signal = np.array([20000], dtype=np.int16)
    integer_factor = AmplitudeScalingTransform(["signal"], 2)({"signal": signal})[
        "signal"
    ]
    float_factor = AmplitudeScalingTransform(["signal"], 2.0)({"signal": signal})[
        "signal"
    ]

    np.testing.assert_array_equal(integer_factor, np.array([40000.0]))
    np.testing.assert_array_equal(integer_factor, float_factor)


def test_normalize_does_not_mutate_input_and_skips_unlisted_label():
    image = np.random.rand(5, 5, 1)
    label = np.arange(25, dtype=np.uint8).reshape(5, 5, 1)
    image_copy = image.copy()
    out = NormalizeTransform(["image"], [0.5], [2.0])({"image": image, "label": label})
    np.testing.assert_array_equal(image, image_copy)
    np.testing.assert_array_equal(out["label"], np.arange(25).reshape(5, 5, 1))


def test_normalize_to_rgb():
    out = NormalizeTransform(["x"], [0] * 3, [1] * 3, to_rgb=True)(
        {"x": np.ones((4, 4, 1))}
    )
    assert out["x"].shape == (4, 4, 3)


def test_identity():
    data = {"x": np.zeros(3)}
    assert IdentityTransform()(data) is data


def test_indexer_repeat_cast_reshape():
    x = np.random.rand(3, 4, 5)
    assert IndexerTransform(["x"], 1)({"x": x})["x"].shape == (4, 5)
    assert RepeatTransform(["x"], 0, 2)({"x": x})["x"].shape == (6, 4, 5)
    assert CastToTransform(["x"], np.float32)({"x": x})["x"].dtype == np.float32
    assert ReshapeTransform(["x"], (12, 5))({"x": x})["x"].shape == (12, 5)
    with pytest.raises(IndexError):
        IndexerTransform(["x"], 5)({"x": x})
    with pytest.raises(IndexError):
        RepeatTransform(["x"], 5, 2)({"x": x})


def test_padding_per_key_value():
    data = {"image": np.ones((4, 4, 3)), "label": np.ones((4, 4), dtype=np.uint8)}
    out = PaddingTransform(
        ["image", "label"], 6, 6, padding_mode="constant", padding_value=[0, 255]
    )(data)
    assert out["image"].shape == (6, 6, 3)
    assert out["label"].shape == (6, 6)  # rank is preserved
    assert out["image"][5, 5, 0] == 0
    assert out["label"][5, 5] == 255


def test_padding_reflect_default_and_bad_value_length():
    out = PaddingTransform(["x"], 6, 6)({"x": np.random.rand(4, 4, 3)})
    assert out["x"].shape == (6, 6, 3)
    with pytest.raises(ValueError):
        PaddingTransform(["a", "b"], 6, 6, padding_value=[0, 1, 2])


def test_gradient():
    out = GradientTransform(["x"], 0)({"x": np.zeros((5, 6))})["x"]
    assert out.shape == (5, 6, 2)
    assert out[0, 0, 1] == 0 and out[0, -1, 1] == 1
    with pytest.raises(AssertionError):
        GradientTransform(["x"], 2)


def test_color_jitter():
    x = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    y = ColorJitterTransform(["x"], 0.8, 1.2, 1.1, 10)({"x": x})["x"]
    assert y.shape == x.shape and not np.array_equal(x, y)


def test_crop_hwc_2d_padding_and_bbox():
    x = np.random.randint(0, 256, size=(50, 50, 3), dtype=np.uint8)
    assert CropTransform(["x"], (30, 30), coords=(0.5, 0.5))({"x": x})["x"].shape == (
        30,
        30,
        3,
    )
    assert CropTransform(["x"], (40, 40))({"x": x[:20, :20]})["x"].shape == (40, 40, 3)
    assert CropTransform(["x"], (30, 30))({"x": x[..., 0]})["x"].shape == (30, 30)

    z = np.zeros((50, 50, 3), dtype=np.uint8)
    z[10:20, 10:20] = 255
    y = CropTransform(["x"], (10, 10), bbox=(10, 20, 10, 20))({"x": z})["x"]
    assert y.shape == (10, 10, 3) and np.all(y == 255)


def test_crop_same_region_for_all_keys():
    image = np.arange(100).reshape(10, 10, 1)
    out = CropTransform(["image", "label"], (4, 4), coords=(0.5, 0.5))(
        {"image": image, "label": image.copy()}
    )
    np.testing.assert_array_equal(out["image"], out["label"])


def test_grayscale_and_solarize():
    x = np.random.randint(0, 256, size=(40, 40, 3), dtype=np.uint8)
    g = GrayScaleTransform(["x"], "average")({"x": x})["x"]
    assert np.all(g[..., 0] == g[..., 1]) and np.all(g[..., 1] == g[..., 2])
    s = SolarizeTransform(["x"], 128)({"x": x})["x"]
    assert s.shape == x.shape and not np.array_equal(s, x)
    with pytest.raises(ValueError):
        GrayScaleTransform(["x"], "bad")  # type: ignore


def test_rotation_modes():
    image = np.random.randint(0, 256, size=(60, 80, 3), dtype=np.uint8)
    label = np.random.randint(0, 5, size=(60, 80), dtype=np.uint8)
    out = RotationTransform(["image", "label"], 33, mode=["bilinear", "nearest"])(
        {"image": image, "label": label}
    )
    assert out["image"].shape == image.shape and not np.array_equal(out["image"], image)
    # nearest never creates new label values
    assert set(np.unique(out["label"])) <= set(np.unique(label))


def test_mode_length_validated():
    with pytest.raises(ValueError, match="one entry per key"):
        RotationTransform(["image", "label"], 10, mode=["nearest"])
    with pytest.raises(ValueError, match="Unknown mode"):
        ResizeTransform(["image"], 4, 4, mode=["lanczos"])


def test_resize_modes_and_aspect_ratio():
    label = np.random.randint(0, 5, size=(20, 40), dtype=np.uint8)
    out = ResizeTransform(["image", "label"], 10, 10, mode=["bilinear", "nearest"])(
        {"image": np.random.rand(20, 40, 3).astype(np.float32), "label": label}
    )
    assert out["image"].shape == (10, 10, 3) and out["label"].shape == (10, 10)
    assert set(np.unique(out["label"])) <= set(np.unique(label))
    kept = ResizeTransform(["x"], 10, 10, keep_aspect_ratio=True)(
        {"x": np.random.rand(20, 40)}
    )
    assert kept["x"].shape == (5, 10)


def test_random_crop_pads_when_too_small_and_shares_position():
    # one axis needs padding (20 -> 30), the other cropping (20 -> 10)
    image = np.arange(3 * 20 * 20).reshape(20, 20, 3)
    out = RandomCropTransform(["image", "label"], (10, 30), seed=0)(
        {"image": image, "label": image.copy()}
    )
    assert out["image"].shape == (10, 30, 3)
    np.testing.assert_array_equal(out["image"], out["label"])


def test_vertical_patch_swap_keeps_image_and_mask_in_sync():
    mask = np.arange(9 * 6, dtype=np.float32).reshape(9, 6)
    data = {
        "image": mask[..., None].copy(),
        "mask": mask.copy(),
        "extra": "unchanged",
    }
    out = VerticalPatchSwapTransform(
        ["image", "mask"], patch_height=4, seed=11
    )(data)
    np.testing.assert_array_equal(out["image"][..., 0], out["mask"])
    assert out["extra"] == "unchanged"


def test_vertical_patch_swap_rejects_mismatched_spatial_shapes():
    with pytest.raises(ValueError, match="matching spatial dimensions"):
        VerticalPatchSwapTransform(
            ["image", "mask"], patch_height=2, seed=11
        )({"image": np.zeros((1, 8, 6)), "mask": np.zeros((7, 6))})


def test_crop_constant_pad_value_per_key():
    data = {"image": np.ones((4, 4, 3)), "label": np.ones((4, 4), dtype=np.uint8)}
    out = CropTransform(
        ["image", "label"], (6, 6), pad_mode="constant", pad_value=[0, 255]
    )(data)
    assert out["image"].shape == (6, 6, 3) and out["label"].shape == (6, 6)
    assert out["image"][0, 0, 0] == 0 and out["label"][0, 0] == 255  # centered padding
