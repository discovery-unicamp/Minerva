import pytest

from minerva.transforms.split_transform import SplitTransform

import numpy as np
import torch


def test_split_transform_invalid_parameters():
    erroneus_value = 0
    with pytest.raises(
        ValueError,
        match=f"Expected input 'num_splits' to be a positive integer greater than 0, but received {erroneus_value}.",
    ):
        # Define an invalid split transform
        SplitTransform(num_splits=erroneus_value, split_dimension=1)

    erroneus_value = -1
    with pytest.raises(
        ValueError,
        match=f"Expected input 'num_splits' to be a positive integer greater than 0, but received {erroneus_value}.",
    ):
        # Define an invalid split transform
        SplitTransform(num_splits=erroneus_value, split_dimension=1)

    erroneus_value = -1
    with pytest.raises(
        ValueError,
        match=f"Expected input 'split_dimension' to be a positive integer greater than or equal to 0, but received {erroneus_value}.",
    ):
        # Define an invalid split transform
        SplitTransform(num_splits=2, split_dimension=erroneus_value)


def test_split_transform_invalid_input():
    # Define a valid split transform
    split_transform = SplitTransform(num_splits=3, split_dimension=1)
    erroneus_value = "invalid input"
    with pytest.raises(
        TypeError,
        match=f"Expected input 'x' to be either a numpy array or a Pytorch tensor, but received an object of type {type(erroneus_value)}.",
    ):
        # Apply the split transform with an invalid input
        split_transform(erroneus_value)


def test_split_transform_invalid_split():
    random_input = np.random.rand(10, 20, 30)
    # Define a valid split transform
    split_transform = SplitTransform(num_splits=3, split_dimension=1)
    with pytest.raises(
        ValueError,
        match=f"Invalid split: expected {split_transform.num_splits} to divide equally the dimension {random_input.shape[split_transform.split_dimension]}.",
    ):
        # Apply the split transform with an invalid split
        split_transform(random_input)

    # Define a valid split transform
    split_transform = SplitTransform(num_splits=50, split_dimension=2)
    with pytest.raises(
        ValueError,
        match=f"Invalid split: expected {split_transform.num_splits} to divide equally the dimension {random_input.shape[split_transform.split_dimension]}.",
    ):
        # Apply the split transform with an invalid split
        split_transform(random_input)

    # Define a valid split transform
    split_transform = SplitTransform(num_splits=50, split_dimension=5)
    with pytest.raises(
        ValueError,
        match=f"Invalid split dimension: expected the split dimension to be less than {len(random_input.shape)}, but received {split_transform.split_dimension}.",
    ):
        # Apply the split transform with an invalid split
        split_transform(random_input)


def test_split_transform_numpy():
    # Create a dummy input
    x = np.random.rand(10, 20, 30)

    # Apply the split transform
    split_transform = SplitTransform(num_splits=2, split_dimension=1)
    splits = split_transform(x)

    # Check if the split data has the correct shape
    assert len(splits) == 2
    assert splits[0].shape == (10, 10, 30)
    assert splits[1].shape == (10, 10, 30)

    # Check if the split data is correct
    assert np.allclose(splits[0], x[:, :10])
    assert np.allclose(splits[1], x[:, 10:])


def test_split_transform_torch():
    # Create a dummy input
    x = torch.rand(10, 20, 30)

    # Apply the split transform
    split_transform = SplitTransform(num_splits=2, split_dimension=1)
    splits = split_transform(x)

    # Check if the split data has the correct shape
    assert len(splits) == 2
    assert splits[0].shape == (10, 10, 30)
    assert splits[1].shape == (10, 10, 30)

    # Check if the split data is correct
    assert torch.allclose(splits[0], x[:, :10])
    assert torch.allclose(splits[1], x[:, 10:])
