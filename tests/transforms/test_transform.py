from typing import Sequence

import numpy as np

from sslt.transforms import Flip, TransformPipeline, _Transform


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
    try:
        flipped_x = flip_transform(x)
    except AssertionError:
        pass
    else:
        assert False, "Expected an AssertionError, but no exception was raised."


# Run the tests
test_transform_pipeline()
test_flip_single_axis()
test_flip_multiple_axes()
test_flip_invalid_axes()
