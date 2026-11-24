import numpy as np
import pytest
import torch

from minerva.transforms import (
    DINOTransform,
    MoCoAugmentationTransform,
    RandomResizeTransform,
    RandomResizedCropTransform,
    ResizeTransform,
    RotationTransform,
    TFCTransform,
)
from minerva.transforms.transform import GaussianBlurTransform


@pytest.mark.parametrize(
    "dtype", ["uint8", "int32", "int64", "uint16", "float16", "bool"]
)
def test_geometric_nearest_preserves_dtype_and_classes(dtype):
    label = (np.arange(64).reshape(8, 8) % 3).astype(dtype)
    for t, x in [
        (ResizeTransform(["l"], 16, 16, mode=["nearest"]), label),
        (RotationTransform(["l"], 15, mode=["nearest"]), label),
        (RandomResizedCropTransform(["l"], (8, 8), mode=["nearest"], seed=0), label),
    ]:
        out = t({"l": x.copy()})["l"]
        assert out.dtype == label.dtype
        assert set(np.unique(out)) <= set(np.unique(label))


def test_int64_label_stays_int64_for_cross_entropy():
    label = np.random.randint(0, 3, (8, 8)).astype("int64")
    out = ResizeTransform(["l"], 4, 4, mode=["nearest"])({"l": label})["l"]
    assert out.dtype == np.int64


def test_blur_int64_does_not_crash():
    x = np.ones((3, 8, 8), "int64") * 5
    out = GaussianBlurTransform(["x"], 3, seed=1)({"x": x})["x"]
    assert out.dtype == np.int64 and (out == 5).all()


def test_random_resize_tiny_factor_does_not_crash():
    x = np.ones((8, 8, 3), "float32")
    out = RandomResizeTransform(["x"], (8, 8), (0.01, 0.02), seed=1)({"x": x})["x"]
    assert out.shape[0] >= 1 and out.shape[1] >= 1


def _img():
    from PIL import Image

    return Image.fromarray(
        (np.random.RandomState(0).rand(64, 64, 3) * 255).astype("uint8")
    )


def test_moco_seed_reproducible_and_restores_global_rng():
    img = _img()
    a = MoCoAugmentationTransform(["x"], seed=3)({"x": img})["x"]
    b = MoCoAugmentationTransform(["x"], seed=3)({"x": img})["x"]
    c = MoCoAugmentationTransform(["x"], seed=4)({"x": img})["x"]
    assert torch.equal(a, b) and not torch.equal(a, c)
    torch.manual_seed(9)
    before = torch.rand(1)
    torch.manual_seed(9)
    MoCoAugmentationTransform(["x"], seed=3)({"x": img})
    assert torch.equal(torch.rand(1), before)  # global RNG state untouched


def test_tfc_seed_reproducible():
    x = np.random.RandomState(0).rand(64, 3, 32).astype("float32")
    a = TFCTransform(["x"], seed=5).augment(x)
    b = TFCTransform(["x"], seed=5).augment(x)
    c = TFCTransform(["x"], seed=6).augment(x)
    assert all(torch.equal(u, v) for u, v in zip(a, b))
    assert not torch.equal(a[1], c[1])


def test_dino_seed_reproducible():
    kw = dict(local_crops_number=2, global_crops_size=32, local_crops_size=16)
    img = _img()
    a = DINOTransform(["x"], seed=1, **kw)({"x": img})
    b = DINOTransform(["x"], seed=1, **kw)({"x": img})
    c = DINOTransform(["x"], seed=2, **kw)({"x": img})
    assert torch.equal(a["x_global_crops"], b["x_global_crops"])
    assert not torch.equal(a["x_global_crops"], c["x_global_crops"])


def test_spatial_transforms_share_one_hwc_layout():
    """(H, W) labels and (H, W, C) images are cropped/flipped/resized identically."""
    from minerva.transforms import (
        CropTransform,
        RandomCropTransform,
        RandomFlipTransform,
    )

    base = np.random.RandomState(0).rand(20, 24).astype("float32")
    image = np.repeat(base[..., None], 3, axis=-1)
    for t in [
        RandomFlipTransform(["image", "label"], prob=1.0),  # default axis 1
        RandomCropTransform(["image", "label"], (10, 12), seed=0),
        CropTransform(["image", "label"], (10, 12), coords=(0.3, 0.7)),
        ResizeTransform(["image", "label"], 10, 12, mode=["nearest", "nearest"]),
    ]:
        out = t({"image": image.copy(), "label": base.copy()})
        assert out["image"].ndim == 3 and out["image"].shape[-1] == 3
        np.testing.assert_array_equal(out["image"][..., 0], out["label"])


def _flip_pipe(seed):
    from minerva.transforms import RandomFlipTransform, _TransformsPipeline

    return _TransformsPipeline(
        [RandomFlipTransform(["image"], possible_axis=1, prob=0.5, seed=seed)]
    )


def _view_bits(multi, n=40):
    base = np.arange(6, dtype="float32").reshape(1, 6)
    v1, v2 = [], []
    for _ in range(n):
        out = multi({"image": base.copy()})
        v1.append(int(out["view1_image"][0, 0] == 5))
        v2.append(int(out["view2_image"][0, 0] == 5))
    return v1, v2


def test_multiview_separate_objects_with_same_seed_give_different_views():
    from minerva.transforms import MultiViewTransform

    v1, v2 = _view_bits(
        MultiViewTransform(["image"], {"view1": _flip_pipe(0), "view2": _flip_pipe(0)})
    )
    assert v1 != v2


def test_multiview_with_seeds_is_reproducible():
    from minerva.transforms import MultiViewTransform

    def make():
        return MultiViewTransform(
            ["image"], {"view1": _flip_pipe(0), "view2": _flip_pipe(0)}
        )

    assert _view_bits(make()) == _view_bits(make())


def test_multiview_shared_object_views_still_differ():
    from minerva.transforms import MultiViewTransform

    shared = _flip_pipe(3)
    v1, v2 = _view_bits(
        MultiViewTransform(["image"], {"view1": shared, "view2": shared})
    )
    assert v1 != v2


def test_multiview_first_view_keeps_plain_seed():
    from minerva.transforms import MultiViewTransform

    alone = _flip_pipe(0)
    base = np.arange(6, dtype="float32").reshape(1, 6)
    expected = [
        int(alone({"image": base.copy()})["image"][0, 0] == 5) for _ in range(40)
    ]
    v1, _ = _view_bits(
        MultiViewTransform(["image"], {"view1": _flip_pipe(0), "view2": _flip_pipe(0)})
    )
    assert v1 == expected


def test_solarize_float_images_use_unit_range():
    from minerva.transforms import SolarizeTransform

    x = np.array([[0.2, 0.5, 0.9]], dtype="float32")
    out = SolarizeTransform(["x"])({"x": x.copy()})["x"]
    np.testing.assert_allclose(
        out, [[0.2, 0.5, 0.1]], atol=1e-6
    )  # >= 0.5 inverted as 1 - x
    assert out.dtype == np.float32


def test_solarize_uint8_keeps_historical_behaviour():
    from minerva.transforms import SolarizeTransform

    x = np.array([[100, 127, 128, 200]], dtype="uint8")
    out = SolarizeTransform(["x"])({"x": x.copy()})["x"]
    np.testing.assert_array_equal(out, [[100, 127, 127, 55]])
    assert out.dtype == np.uint8
    out3 = SolarizeTransform(["x"], 128)({"x": np.full((2, 2, 3), 200, "uint8")})["x"]
    assert (out3 == 55).all()


def test_solarize_float_0_255_with_explicit_max_value():
    from minerva.transforms import SolarizeTransform

    x = np.array([[100.0, 200.0]], dtype="float32")
    out = SolarizeTransform(["x"], 128, max_value=255)({"x": x})["x"]
    np.testing.assert_allclose(out, [[100.0, 55.0]])


def test_solarize_uint16_uses_dtype_range():
    from minerva.transforms import SolarizeTransform

    x = np.array([[1000, 60000]], dtype="uint16")
    out = SolarizeTransform(["x"])({"x": x})["x"]
    np.testing.assert_array_equal(out, [[1000, 5535]])


def test_random_solarize_float():
    from minerva.transforms import RandomSolarizeTransform

    x = np.full((4, 4, 3), 0.8, dtype="float32")
    out = RandomSolarizeTransform(["x"], prob=1.0, seed=0)({"x": x})["x"]
    np.testing.assert_allclose(out, 0.2, atol=1e-6)


# ---- round 2 (review feedback) ---------------------------------------------
def test_resize_keeps_one_channel_axis():
    from minerva.transforms import RandomResizeTransform, ResizeTransform

    x = np.random.rand(16, 16, 1).astype("float32")
    assert ResizeTransform(["x"], 8, 8)({"x": x})["x"].shape == (8, 8, 1)
    assert ResizeTransform(["x"], 8, 8)({"x": x[..., 0]})["x"].shape == (8, 8)
    out = RandomResizeTransform(["x"], (8, 8), (1.0, 1.0), seed=0)({"x": x})["x"]
    assert out.shape == (8, 8, 1)


def test_random_crop_can_pick_last_legal_position():
    from minerva.transforms import CropTransform, RandomCropTransform

    base = np.arange(25).reshape(5, 5)
    t = RandomCropTransform(["x"], (4, 4), seed=0)
    starts = {int(t({"x": base.copy()})["x"][0, 0]) for _ in range(200)}
    assert starts == {0, 1, 5, 6}  # offsets (0,0) (0,1) (1,0) (1,1)
    last = CropTransform(["x"], (4, 4), coords=(1.0, 1.0))({"x": base})["x"]
    assert last[0, 0] == 6


def test_class_ratio_check_does_not_count_padding_as_a_class():
    from minerva.transforms import RandomCropTransform

    label = np.ones(
        (2, 4), dtype=np.uint8
    )  # padded to 4x4, so half the crop is padding
    # label not cropped itself: padding is ignored, the 8 valid pixels are all class 1
    t = RandomCropTransform(
        ["image"], (4, 4), label_key="label", max_ratio=0.75, ignore_index=255, seed=0
    )
    assert not t._accepts((0.0, 0.0), label)
    # label cropped too, constant fill == ignore_index: same answer
    t2 = RandomCropTransform(
        ["image", "label"],
        (4, 4),
        pad_mode="constant",
        pad_value=[0, 255],
        label_key="label",
        max_ratio=0.75,
        ignore_index=255,
        seed=0,
    )
    assert not t2._accepts((0.0, 0.0), label)


def test_seed_scope_does_not_touch_cuda_rng(monkeypatch):
    from minerva.transforms import MoCoAugmentationTransform

    def boom(*a, **k):
        raise AssertionError("torch.manual_seed also reseeds CUDA")

    monkeypatch.setattr(torch, "manual_seed", boom)
    MoCoAugmentationTransform(["x"], seed=1)({"x": _img()})


def test_multiview_nested_keeps_inner_streams_distinct():
    from minerva.transforms import MultiViewTransform, _TransformsPipeline

    inner = MultiViewTransform(["image"], {"a": _flip_pipe(0), "b": _flip_pipe(0)})
    outer = MultiViewTransform(
        ["image"], {"v1": _TransformsPipeline([inner]), "v2": _flip_pipe(0)}
    )
    base = np.arange(6, dtype="float32").reshape(1, 6)
    bits = {k: [] for k in ("v1_a_image", "v1_b_image")}
    for _ in range(40):
        out = outer({"image": base.copy()})
        for k in bits:
            bits[k].append(int(out[k][0, 0] == 5))
    assert bits["v1_a_image"] != bits["v1_b_image"]


def test_multiview_does_not_reset_the_wrapped_transforms_stream():
    from minerva.transforms import MultiViewTransform

    base = np.arange(6, dtype="float32").reshape(1, 6)

    def draws(p, n):
        return [int(p({"image": base.copy()})["image"][0, 0] == 5) for _ in range(n)]

    original, twin = _flip_pipe(0), _flip_pipe(0)
    draws(original, 3), draws(twin, 3)
    MultiViewTransform(["image"], {"view1": original, "view2": _flip_pipe(1)})
    assert draws(original, 20) == draws(twin, 20)  # stream continues, not restarted
