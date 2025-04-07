import numpy as np
import pytest

from minerva.transforms.random_transform import RandomFlip


def test_random_flip_single_axis_with_flip():
    # Create a dummy input
    x = np.random.rand(10, 20)

    # Apply the flip transform along the first axis
    flip_transform = RandomFlip(possible_axis=0, num_samples=1, seed=1)
    flipped_x = flip_transform(x)

    # Check if the flipped data has the same shape as the input
    assert flipped_x.shape == x.shape

    # Check if the flipped data is different from the input
    assert np.allclose(flipped_x, np.flip(x, axis=0))


def test_random_flip_single_axis_without_flip():
    # Create a dummy input
    x = np.random.rand(10, 20)

    # Apply the flip transform along the first axis
    flip_transform = RandomFlip(possible_axis=0, num_samples=1, seed=0)
    flipped_x = flip_transform(x)

    # Check if the flipped data has the same shape as the input
    assert flipped_x.shape == x.shape

    # Check if the flipped data is different from the input
    assert np.allclose(flipped_x, x)


def test_random_flip_first_axis():
    # Create a dummy input
    x = np.random.rand(10, 20, 30)

    # Apply the flip transform along multiple axes
    flip_transform = RandomFlip(possible_axis=[0, 1], num_samples=1, seed=1)
    flipped_x = flip_transform(x)

    # Check if the flipped data has the same shape as the input
    assert flipped_x.shape == x.shape

    # check if only the first axis is flipped
    assert np.allclose(flipped_x, np.flip(x, axis=0))


def test_random_flip_second_axis():
    # Create a dummy input
    x = np.random.rand(10, 20, 30)

    # Apply the flip transform along multiple axes
    flip_transform = RandomFlip(possible_axis=[0, 1], num_samples=1, seed=3)
    flipped_x = flip_transform(x)

    # Check if the flipped data has the same shape as the input
    assert flipped_x.shape == x.shape

    # check if the second axis is flipped
    assert np.allclose(flipped_x, np.flip(x, axis=1))


def test_random_flip_two_axis():
    # Create a dummy input
    x = np.random.rand(10, 20, 30)

    # Apply the flip transform along multiple axes
    flip_transform = RandomFlip(possible_axis=[0, 1], num_samples=1, seed=11)
    flipped_x = flip_transform(x)

    # Check if the flipped data has the same shape as the input
    assert flipped_x.shape == x.shape

    # check if both axis are flipped
    assert np.allclose(flipped_x, np.flip(x, axis=(0, 1)))


def test_random__dont_flip_any_axis():
    # Create a dummy input
    x = np.random.rand(10, 20, 30)

    # Apply the flip transform along multiple axes
    flip_transform = RandomFlip(possible_axis=[0, 1], num_samples=1, seed=0)
    flipped_x = flip_transform(x)

    # Check if the flipped data has the same shape as the input
    assert flipped_x.shape == x.shape

    # check if both axis are flipped
    assert np.allclose(flipped_x, x)

def test_random_flip_different_transforms():
    # Create a dummy input
    x1 = np.random.rand(10, 20, 30)
    x2 = x1.copy()
    
    flip_transform = RandomFlip(possible_axis=[0, 1], num_samples=1, seed=1)
    flipped_x1 = flip_transform(x1)
    flipped_x2 = flip_transform(x2)
    
    # Check if the flipped data has the same shape as the input
    assert flipped_x1.shape == x1.shape
    assert flipped_x2.shape == x2.shape
    
    # Check if the flipped data is different from the input
    assert not np.array_equal(flipped_x1, x1)
    assert not np.array_equal(flipped_x2, x2)

    #Check if the flipped data is different from each other
    assert not np.array_equal(flipped_x1, flipped_x2)

def test_random_flip_equal_transforms():
    # Create a dummy input
    x1 = np.random.rand(10, 20, 30)
    x2 = x1.copy()
    
    flip_transform = RandomFlip(possible_axis=[0, 1], num_samples=2, seed=0)
    flipped_x1 = flip_transform(x1)
    flipped_x2 = flip_transform(x2)
    
    # Check if the flipped data has the same shape as the input
    assert flipped_x1.shape == x1.shape
    assert flipped_x2.shape == x2.shape
    
    # Check if the flipped data is different from the input
    assert not np.array_equal(flipped_x1, x1)
    assert not np.array_equal(flipped_x2, x2)

    #Check if the flipped data is equal to each other
    assert np.array_equal(flipped_x1, flipped_x2)


