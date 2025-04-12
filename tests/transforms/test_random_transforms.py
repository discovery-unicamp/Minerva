import numpy as np
import pytest

from minerva.transforms import GrayScale, Solarize, Rotation
from minerva.transforms import (
    RandomCrop,
    RandomGrayScale,
    RandomSolarize,
    RandomRotation,
)
from minerva.transforms.random_transform import EmptyTransform


def test_random_crop_shape():
    x = np.random.randint(0, 256, size=(50, 50, 3), dtype=np.uint8)
    transform = RandomCrop(crop_size=(30, 30), seed=42)
    crop_transform = transform.select_transform()
    y = crop_transform(x)

    assert y.shape == (30, 30, 3)


def test_random_grayscale_prob_1():
    x = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
    transform = RandomGrayScale(prob=1.0, gray=77, seed=123)
    selected = transform.select_transform()

    assert isinstance(selected, GrayScale)
    y = selected(x)
    assert np.all(y[..., 0] == 77)


def test_random_grayscale_prob_0():
    transform = RandomGrayScale(prob=0.0, gray=77, seed=123)
    selected = transform.select_transform()

    assert isinstance(selected, EmptyTransform)


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

    assert isinstance(selected, EmptyTransform)


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

    assert isinstance(selected, EmptyTransform)


def test_random_rotation_within_degree_range():
    transform = RandomRotation(degrees=30, prob=1.0, seed=123)
    selected = transform.select_transform()

    assert isinstance(selected, Rotation)
    assert -30 <= selected.degrees <= 30
