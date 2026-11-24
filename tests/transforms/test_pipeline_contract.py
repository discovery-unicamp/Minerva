"""Tests for the Minerva 1.0 transform contract (dict in, dict out)."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from minerva.transforms import (
    CropTransform,
    ExternalTransform,
    FlipTransform,
    MultiViewTransform,
    RandomCropTransform,
    RandomFlipTransform,
    RandomResizedCropTransform,
    ResizeTransform,
    _TransformsPipeline,
)


def _sample():
    image = np.random.rand(32, 32, 3).astype(np.float32)
    return {
        "image": image,
        "label": image[..., 0].copy(),
        "metadata": {"id": 1},
    }


UNLISTED_PASS_THROUGH = [
    lambda: FlipTransform(["image"], 1),
    lambda: CropTransform(["image"], (8, 8)),
    lambda: RandomFlipTransform(["image"], 1, prob=1.0),
    lambda: RandomCropTransform(["image"], (8, 8)),
    lambda: RandomResizedCropTransform(["image"], (8, 8)),
    lambda: ResizeTransform(["image"], 8, 8),
]


@pytest.mark.parametrize("make", UNLISTED_PASS_THROUGH)
def test_unlisted_keys_pass_through_unchanged(make):
    data = _sample()
    label, metadata = data["label"].copy(), data["metadata"]
    out = make()(data)
    np.testing.assert_array_equal(out["label"], label)
    assert out["metadata"] is metadata


def test_random_params_shared_across_listed_keys():
    t = RandomCropTransform(["image", "label"], (8, 8))
    for _ in range(20):
        out = t(_sample())
        np.testing.assert_array_equal(out["image"][..., 0], out["label"])


def test_class_ratio_crop_reads_label_and_crops_all_keys():
    label = np.zeros((32, 32), dtype=np.int64)
    label[:, 16:] = 1
    image = np.repeat(label.astype(np.float32)[..., None], 3, axis=-1)
    t = RandomCropTransform(
        ["image", "label"], (16, 16), label_key="label", max_ratio=0.9
    )
    out = t({"image": image, "label": label, "metadata": "m"})
    assert out["image"].shape == (16, 16, 3) and out["label"].shape == (16, 16)
    np.testing.assert_array_equal(out["image"][..., 0], out["label"])
    assert out["metadata"] == "m"


def test_external_transform_applies_callable_per_key():
    out = ExternalTransform(["x"], lambda v: v + 1)({"x": 1, "y": 5})
    assert out == {"x": 2, "y": 5}


def _views(keys=("image",)):
    pipe = _TransformsPipeline([RandomCropTransform(list(keys), (8, 8))])
    return MultiViewTransform(list(keys), views={"view1": pipe, "view2": pipe})


def test_multi_view_output_keys():
    out = _views()({"image": np.random.rand(32, 32, 1), "metadata": 3})
    assert set(out) == {"view1_image", "view2_image", "metadata"}


def test_multi_view_views_differ_but_stay_in_sync_within_view():
    t = _views(("image", "label"))
    differing = 0
    for _ in range(20):
        out = t(_sample() | {})
        for view in ("view1", "view2"):
            np.testing.assert_array_equal(
                out[f"{view}_image"][..., 0], out[f"{view}_label"]
            )
        differing += not np.array_equal(out["view1_image"], out["view2_image"])
    assert differing > 0


def test_multi_view_does_not_alias_source_or_views():
    src = np.ones((1, 4, 4), dtype=np.float32)
    pipe = _TransformsPipeline([IdentityLike()])
    out = MultiViewTransform(["image"], {"a": pipe, "b": pipe})({"image": src})
    out["a_image"][...] = 7
    assert np.all(src == 1) and np.all(out["b_image"] == 1)


class IdentityLike:
    keys_to_transform = []

    def __call__(self, data):
        return data


class _DictDataset(Dataset):
    def __init__(self, transform):
        self.transform = transform

    def __len__(self):
        return 16

    def __getitem__(self, idx):
        image = np.full((16, 16, 1), idx, dtype=np.float32)
        image += np.arange(256, dtype=np.float32).reshape(16, 16, 1)
        return self.transform({"image": image, "label": image[..., 0].copy()})


def test_dataloader_with_workers_keeps_image_and_label_in_sync():
    transform = _TransformsPipeline(
        [
            RandomCropTransform(["image", "label"], (8, 8)),
            RandomFlipTransform(
                ["image", "label"], 1, prob=0.5
            ),  # axis 1 = width for HW and HWC
        ]
    )
    loader = DataLoader(_DictDataset(transform), batch_size=4, num_workers=2)
    n_batches = 0
    for batch in loader:
        assert isinstance(batch, dict)
        np.testing.assert_array_equal(
            batch["image"][..., 0].numpy(), batch["label"].numpy()
        )
        n_batches += 1
    assert n_batches == 4


class _Constant(Dataset):
    def __len__(self):
        return 16

    def __getitem__(self, idx):
        x = np.arange(400, dtype=np.float32).reshape(1, 20, 20)
        return RandomCropTransform(["x"], (4, 4))({"x": x})


def test_workers_draw_different_params():
    # batch_size=1 with 2 workers: consecutive batches come from different
    # workers, so a repeated per-worker RNG would make batch 0 equal batch 1.
    loader = DataLoader(_Constant(), batch_size=1, num_workers=2)
    firsts = [batch["x"].flatten()[0].item() for batch in loader]
    assert firsts[0] != firsts[1]
    assert len(set(firsts)) > 4


# ---- seeds ------------------------------------------------------------------
import pickle

from minerva.transforms import (
    GaussianBlurTransform,
    PerlinMaskerTransform,
    RandomGrayScaleTransform,
    RandomResizeTransform,
    RandomRotationTransform,
    RandomSolarizeTransform,
)

SEEDED = [
    lambda s: RandomFlipTransform(["x"], 0, seed=s),
    lambda s: RandomCropTransform(["x"], (6, 6), seed=s),
    lambda s: RandomResizedCropTransform(["x"], (6, 6), seed=s),
    lambda s: RandomRotationTransform(["x"], 40, prob=0.9, seed=s),
    lambda s: RandomResizeTransform(["x"], (10, 10), (0.5, 1.5), seed=s),
    lambda s: GaussianBlurTransform(["x"], 3, seed=s),
    lambda s: RandomCropTransform(["x"], (6, 6), seed=s, label_key="x", max_ratio=0.5),
]


def _run(transform, n=12):
    x = np.random.RandomState(0).rand(16, 16).astype(np.float32)
    return [transform({"x": x.copy()})["x"] for _ in range(n)]


@pytest.mark.parametrize("make", SEEDED)
def test_seed_is_reproducible_and_seeds_differ(make):
    a, b, c = _run(make(1)), _run(make(1)), _run(make(2))
    for u, v in zip(a, b):
        np.testing.assert_array_equal(u, v)
    assert any(u.shape != v.shape or not np.array_equal(u, v) for u, v in zip(a, c))


def test_seed_ignores_global_rng_and_none_follows_it():
    t1, t2 = RandomCropTransform(["x"], (6, 6), seed=5), RandomCropTransform(
        ["x"], (6, 6), seed=5
    )
    torch.manual_seed(0)
    a = _run(t1)
    torch.manual_seed(123)
    b = _run(t2)
    for u, v in zip(a, b):
        np.testing.assert_array_equal(u, v)
    # seed=None follows torch.manual_seed
    outs = []
    for _ in range(2):
        torch.manual_seed(9)
        outs.append(_run(RandomCropTransform(["x"], (6, 6))))
    for u, v in zip(*outs):
        np.testing.assert_array_equal(u, v)


def test_other_random_transforms_accept_seed():
    x = np.random.randint(0, 256, size=(16, 16, 3), dtype=np.uint8)
    for cls in (RandomGrayScaleTransform, RandomSolarizeTransform):
        a = [cls(["x"], prob=0.5, seed=3)({"x": x.copy()})["x"] for _ in range(1)]
        b = [cls(["x"], prob=0.5, seed=3)({"x": x.copy()})["x"] for _ in range(1)]
        np.testing.assert_array_equal(a[0], b[0])
    p = lambda: PerlinMaskerTransform(["x"], octaves=2, seed=4)({"x": np.ones((6, 6))})[
        "x"
    ]
    np.testing.assert_array_equal(p(), p())


def test_seeded_transform_survives_pickle_and_str_shows_seed():
    t = RandomCropTransform(["x"], (6, 6), seed=11)
    _run(t, 3)
    t2 = pickle.loads(pickle.dumps(t))
    assert t2.seed == 11 and "seed=11" in str(t2)
    for u, v in zip(_run(t2, 5), _run(RandomCropTransform(["x"], (6, 6), seed=11), 5)):
        np.testing.assert_array_equal(u, v)


class _Seeded(Dataset):
    def __init__(self, seed):
        self.t = RandomCropTransform(["x"], (4, 4), seed=seed)

    def __len__(self):
        return 8

    def __getitem__(self, idx):
        x = np.arange(400, dtype=np.float32).reshape(1, 20, 20)
        return self.t({"x": x})


def _loader_firsts(seed):
    loader = DataLoader(_Seeded(seed), batch_size=1, num_workers=2)
    return [b["x"].flatten()[0].item() for b in loader]


def test_seeded_workers_are_reproducible_but_not_identical():
    a, b = _loader_firsts(3), _loader_firsts(3)
    assert a == b  # same seed, same run-to-run result
    # worker 0 serves even batches, worker 1 odd ones; they must not repeat each other
    assert a[0::2] != a[1::2]
