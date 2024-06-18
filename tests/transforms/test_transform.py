from typing import Sequence

import numpy as np
import pytest

from minerva.transforms import Flip, PerlinMasker, TransformPipeline, _Transform


def test_transform_pipeline():
    # Define some dummy transforms
    transforms: Sequence[_Transform] = [
        Flip(axis=0),
        Flip(axis=1),
        Flip(axis=[0, 1]),
    ]
    pipeline = TransformPipeline(transforms)

    # Create a dummy input
    x = np.random.rand(10, 20)

    # Apply the transform pipeline
    transformed_x = pipeline(x)

    # Check if the transformed data has the same shape as the input
    assert transformed_x.shape == x.shape


def test_flip_single_axis():
    # Create a dummy input
    x = np.random.rand(10, 20)

    # Apply the flip transform along the first axis
    flip_transform = Flip(axis=0)
    flipped_x = flip_transform(x)

    # Check if the flipped data has the same shape as the input
    assert flipped_x.shape == x.shape


def test_flip_multiple_axes():
    # Create a dummy input
    x = np.random.rand(10, 20, 30)

    # Apply the flip transform along multiple axes
    flip_transform = Flip(axis=[0, 1])
    flipped_x = flip_transform(x)

    # Check if the flipped data has the same shape as the input
    assert flipped_x.shape == x.shape


def test_flip_invalid_axes():
    # Create a dummy input
    x = np.random.rand(10, 20)

    # Apply the flip transform with invalid axes
    flip_transform = Flip(axis=[0, 1, 2])

    # Check if an AssertionError is raised when applying the transform
    with pytest.raises(AssertionError):
        flipped_x = flip_transform(x)


def test_perlin_masker_invalid_octaves():
    # Create a dummy input
    x = np.random.rand(10, 20)

    # Check if a ValueError is raised when using invalid octaves
    with pytest.raises(ValueError):
        perlin_masker = PerlinMasker(octaves=-1)
        masked_x = perlin_masker(x)


def test_perlin_masker_invalid_scale():
    # Create a dummy input
    x = np.random.rand(10, 20)

    # Check if a ValueError is raised when using invalid scale
    with pytest.raises(ValueError):
        perlin_masker = PerlinMasker(octaves=3, scale=0)
        masked_x = perlin_masker(x)
