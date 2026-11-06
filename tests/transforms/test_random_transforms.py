import numpy as np
import pytest

from minerva.transforms import (
    GrayScale,
    RandomCrop,
    RandomGrayScale,
    RandomRotation,
    RandomSolarize,
    Rotation,
    Solarize,
    RandomGaussianBlur,
    GaussianBlur,
    RandomVerticalPatchSwap,
    VerticalPatchSwap
)
from minerva.transforms.random_transform import Identity


def test_random_crop_shape():
    # CHW format: (C, H, W)
    x = np.random.randint(0, 256, size=(3, 50, 50), dtype=np.uint8)
    transform = RandomCrop(crop_size=(30, 30), seed=42)
    crop_transform = transform.select_transform()
    y = crop_transform(x)

    assert y.shape == (3, 30, 30)


def test_random_grayscale_prob_1():
    x = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
    transform = RandomGrayScale(prob=1.0, method="average", seed=123)
    selected = transform.select_transform()

    assert isinstance(selected, GrayScale)
    y = selected(x)
    assert np.all(y[..., 0] == y[..., 1]) and np.all(y[..., 1] == y[..., 2])


def test_random_grayscale_prob_0():
    transform = RandomGrayScale(prob=0.0, method="luminosity", seed=123)
    selected = transform.select_transform()

    assert isinstance(selected, Identity)


def test_random_solarize_prob_1():
    x = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)
    transform = RandomSolarize(threshold=100, prob=1.0, seed=0)
    selected = transform.select_transform()

    assert isinstance(selected, Solarize)
    y = selected(x)
    assert y.shape == x.shape
    assert not np.array_equal(x, y)


def test_random_solarize_prob_0():
    transform = RandomSolarize(threshold=100, prob=0.0, seed=0)
    selected = transform.select_transform()

    assert isinstance(selected, Identity)


def test_random_rotation_applies_transform():
    x = np.random.randint(0, 256, size=(60, 80, 3), dtype=np.uint8)
    transform = RandomRotation(degrees=45, prob=1.0, seed=1337)
    selected = transform.select_transform()

    assert isinstance(selected, Rotation)
    y = selected(x)
    assert y.shape == x.shape
    assert not np.array_equal(x, y)


def test_random_rotation_skips_transform():
    transform = RandomRotation(degrees=45, prob=0.0, seed=1337)
    selected = transform.select_transform()

    assert isinstance(selected, Identity)


def test_random_rotation_within_degree_range():
    transform = RandomRotation(degrees=30, prob=1.0, seed=123)
    selected = transform.select_transform()

    assert isinstance(selected, Rotation)
    assert -30 <= selected.degrees <= 30

def test_random_gaussian_blur_prob_1():
    x = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
    transform = RandomGaussianBlur(kernel_size=5, sigma_range=(0.5, 1.5), prob=1.0, seed=42)
    selected = transform.select_transform()

    assert isinstance(selected, GaussianBlur)
    y = selected(x)
    assert y.shape == x.shape
    assert not np.array_equal(x, y)


def test_random_gaussian_blur_prob_0():
    transform = RandomGaussianBlur(kernel_size=5, prob=0.0, seed=42)
    selected = transform.select_transform()

    assert isinstance(selected, Identity)


def test_random_gaussian_blur_within_sigma_range():
    transform = RandomGaussianBlur(kernel_size=5, sigma_range=(1.0, 3.0), prob=1.0, seed=123)
    selected = transform.select_transform()

    assert isinstance(selected, GaussianBlur)
    assert 1.0 <= selected.sigma <= 3.0


def test_random_vertical_patch_swap_prob_1():
    x = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
    transform = RandomVerticalPatchSwap(patch_height=16, prob=1.0, seed=42)
    selected = transform.select_transform()

    # Verify the correct transform is selected
    assert isinstance(selected, VerticalPatchSwap)
    
    # Verify shape and effect
    y = selected(x)
    assert y.shape == x.shape
    assert not np.array_equal(x, y)


def test_random_vertical_patch_swap_prob_0():
    transform = RandomVerticalPatchSwap(patch_height=16, prob=0.0, seed=42)
    selected = transform.select_transform()

    # Verify the identity transform is returned when probability fails
    assert isinstance(selected, Identity)