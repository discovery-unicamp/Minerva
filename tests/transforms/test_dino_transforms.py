import numpy as np
import pytest
import torch
from PIL import Image

from minerva.transforms.dino import (
    DINOTransform,
    GaussianBlur,
    MaskingGenerator,
    make_normalize_transform,
)


@pytest.fixture
def dummy_pil_image():
    arr = np.random.randint(0, 256, size=(256, 256, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_gaussian_blur(dummy_pil_image):
    blur = GaussianBlur(p=1.0, radius_min=0.5, radius_max=1.5)
    # Applied to PIL or tensor
    img_tensor = torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8)
    out = blur(img_tensor)
    assert isinstance(out, torch.Tensor)
    assert out.shape == img_tensor.shape


def test_make_normalize_transform():
    normalize = make_normalize_transform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    x = torch.ones((3, 32, 32), dtype=torch.float32)
    out = normalize(x)
    assert torch.allclose(out, torch.ones_like(x))


def test_data_augmentation_dino_default(dummy_pil_image):
    aug = DINOTransform(['image'],
        global_crops_scale=[0.5, 1.0],
        local_crops_scale=[0.1, 0.5],
        local_crops_number=4,
        global_crops_size=96,
        local_crops_size=48,
    )
    res = aug.augment(dummy_pil_image)

    assert "global_crops" in res
    assert "global_crops_teacher" in res
    assert "local_crops" in res
    assert "offsets" in res
    assert "weak_flag" in res

    assert len(res["global_crops"]) == 2
    assert len(res["global_crops_teacher"]) == 2
    assert len(res["local_crops"]) == 4

    for crop in res["global_crops"]:
        assert isinstance(crop, torch.Tensor)
        assert crop.shape == (3, 96, 96)

    for crop in res["local_crops"]:
        assert isinstance(crop, torch.Tensor)
        assert crop.shape == (3, 48, 48)


def test_data_augmentation_dino_gram_teacher(dummy_pil_image):
    aug = DINOTransform(['image'],
        global_crops_scale=[0.5, 1.0],
        local_crops_scale=[0.1, 0.5],
        local_crops_number=2,
        global_crops_size=96,
        local_crops_size=48,
        gram_teacher_crops_size=64,
        gram_teacher_no_distortions=True,
    )
    res = aug.augment(dummy_pil_image)

    assert "gram_teacher_crops" in res
    assert len(res["gram_teacher_crops"]) == 2
    for crop in res["gram_teacher_crops"]:
        assert crop.shape == (3, 64, 64)


def test_data_augmentation_dino_teacher_no_color_jitter(dummy_pil_image):
    aug = DINOTransform(['image'],
        global_crops_size=64,
        local_crops_size=32,
        local_crops_number=2,
        teacher_no_color_jitter=True,
    )
    res = aug.augment(dummy_pil_image)
    assert len(res["global_crops_teacher"]) == 2
    assert res["global_crops_teacher"][0].shape == (3, 64, 64)


def test_data_augmentation_dino_share_color_jitter(dummy_pil_image):
    aug = DINOTransform(['image'],
        global_crops_size=64,
        local_crops_size=32,
        local_crops_number=2,
        share_color_jitter=True,
    )
    res = aug.augment(dummy_pil_image)
    assert len(res["global_crops"]) == 2
    assert len(res["local_crops"]) == 2


def test_data_augmentation_dino_local_subset(dummy_pil_image):
    aug = DINOTransform(['image'],
        global_crops_size=96,
        local_crops_size=32,
        local_crops_number=4,
        local_crops_subset_of_global_crops=True,
        patch_size=16,
    )
    res = aug.augment(dummy_pil_image)
    assert len(res["local_crops"]) == 4
    assert len(res["offsets"]) == 4
    for crop in res["local_crops"]:
        assert crop.shape == (3, 32, 32)


def test_masking_generator():
    gen = MaskingGenerator(
        input_size=(14, 14),
        num_masking_patches=50,
        min_num_patches=4,
        max_num_patches=20,
    )
    assert gen.get_shape() == (14, 14)
    assert "Generator" in repr(gen)

    mask = gen(num_masking_patches=50)
    assert isinstance(mask, np.ndarray)
    assert mask.shape == (14, 14)
    assert mask.dtype == bool
    assert mask.sum() == 50


def test_masking_generator_complete_randomly():
    gen = MaskingGenerator(input_size=10)
    mask = np.zeros((10, 10), dtype=bool)
    mask[0, 0] = True
    completed = gen.complete_mask_randomly(mask, num_masking_patches=25)
    assert completed.shape == (10, 10)
    assert completed.sum() == 25
    assert completed[0, 0] is np.True_ or completed[0, 0] == True


def test_dino_transform_dict_call(dummy_pil_image):
    aug = DINOTransform(
        ["image"],
        local_crops_number=2,
        global_crops_size=64,
        local_crops_size=32,
    )
    out = aug({"image": dummy_pil_image, "label": 7})

    assert "image" not in out
    assert out["label"] == 7
    assert out["image_global_crops"].shape == (2, 3, 64, 64)
    assert out["image_global_crops_teacher"].shape == (2, 3, 64, 64)
    assert out["image_local_crops"].shape == (2, 3, 32, 32)
    assert "image_offsets" in out
