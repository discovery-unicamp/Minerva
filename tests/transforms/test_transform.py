from typing import Sequence

import numpy as np
import pytest

from minerva.transforms import (
    _Transform,
    TransformPipeline,
    Flip,
    PerlinMasker,
    Squeeze,
    Unsqueeze,
    Transpose,
    CastTo,
    Identity,
    Indexer,
    Repeat,
    Normalize,
    ContrastiveTransform,
    Crop,
    ColorJitter,
    GrayScale,
    Solarize,
    Rotation,
)
from minerva.transforms.random_transform import EmptyTransform


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


def test_transform_pipeline_add_and_radd():
    x = np.random.rand(10, 20)

    pipeline_1 = TransformPipeline([Flip(axis=0), Flip(axis=1)])
    pipeline_2 = TransformPipeline([Flip(axis=[0, 1])])

    pipeline_3 = pipeline_1 + pipeline_2
    assert len(pipeline_3.transforms) == 3
    transformed_x = pipeline_3(x)
    assert transformed_x.shape == x.shape

    pipeline_4 = pipeline_1 + Flip(axis=[0, 1])
    assert len(pipeline_4.transforms) == 3
    transformed_x = pipeline_4(x)
    assert transformed_x.shape == x.shape

    pipeline_1 += Flip(axis=[0, 1])
    assert len(pipeline_1.transforms) == 3
    transformed_x = pipeline_1(x)
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


def test_flip_str():
    flip_transform = Flip(axis=[0, 1, 2])
    assert str(flip_transform) == "Flip(axis=[0, 1, 2])"


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


def test_transpose():
    # Create a dummy input
    x = np.random.rand(3, 2, 5, 4)

    # Apply the transform
    transpose_transform = Transpose([2, 3, 0, 1])
    transposed_x = transpose_transform(x)

    # Check if the axes have the correct length
    assert transposed_x.shape == (5, 4, 3, 2)


def test_squeeze():
    # Create a dummy input with a single-dimensional axis
    x = np.random.rand(1, 10, 20)

    # Apply the squeeze transform
    squeeze_transform = Squeeze(axis=0)
    squeezed_x = squeeze_transform(x)

    # Check if the squeezed data has the correct shape
    assert squeezed_x.shape == (10, 20)

    # Check if the data content remains unchanged
    assert np.array_equal(squeezed_x, np.squeeze(x, axis=0))


def test_squeeze_invalid_axis():
    # Create a dummy input without a single-dimensional axis
    x = np.random.rand(10, 20)

    # Apply the squeeze transform with an invalid axis
    squeeze_transform = Squeeze(axis=0)

    # Check if an exception is raised
    with pytest.raises(ValueError):
        squeezed_x = squeeze_transform(x)


def test_unsqueeze():
    # Create a dummy input
    x = np.random.rand(10, 20)

    # Apply the unsqueeze transform
    unsqueeze_transform = Unsqueeze(axis=0)
    unsqueezed_x = unsqueeze_transform(x)

    # Check if the unsqueezed data has the correct shape
    assert unsqueezed_x.shape == (1, 10, 20)

    # Check if the data content remains unchanged
    assert np.array_equal(unsqueezed_x, np.expand_dims(x, axis=0))


def test_unsqueeze_invalid_axis():
    # Create a dummy input
    x = np.random.rand(10, 20)

    # Apply the unsqueeze transform with an invalid axis
    unsqueeze_transform = Unsqueeze(axis=3)

    # Check if an exception is raised
    with pytest.raises(IndexError):
        unsqueezed_x = unsqueeze_transform(x)


def test_squeeze_str():
    squeeze_transform = Squeeze(axis=0)
    assert str(squeeze_transform) == "Squeeze(axis=0)"


def test_unsqueeze_str():
    unsqueeze_transform = Unsqueeze(axis=1)
    assert str(unsqueeze_transform) == "Unsqueeze(axis=1)"


def test_normalize():
    # Create a dummy input
    x = np.random.rand(1, 10, 20)

    mean = np.mean(x[0])
    std = np.std(x[0])

    # Apply the normalize transform
    normalize_transform = Normalize([mean], [std])
    normalized_x = normalize_transform(x)

    # Check if the normalized data has the same shape as the input
    assert normalized_x.shape == x.shape

    # Check if the mean and standard deviation are close to 0 and 1, respectively
    assert np.allclose(np.mean(normalized_x), 0)
    assert np.allclose(np.std(normalized_x), 1)


def test_contrastive_transform():
    # Create a dummy input
    x = np.random.rand(10, 20)

    # Apply the contrastive transform
    contrastive_transform = ContrastiveTransform(EmptyTransform())
    contrastive_x = contrastive_transform(x)

    # Check if the contrastive transform returns a tuple of 2 elements
    assert isinstance(contrastive_x, tuple)
    assert len(contrastive_x) == 2

    # Check if the contrastive data has the same shape as the input
    assert contrastive_x[0].shape == x.shape and contrastive_x[1].shape == x.shape


def test_identity():
    # Create a dummy input
    x = np.random.rand(10, 20)

    # Apply the identity transform
    identity_transform = Identity()
    transformed_x = identity_transform(x)

    # Check if the transformed data is identical to the input
    assert np.array_equal(transformed_x, x)

    # Check the string representation
    assert str(identity_transform) == "Identity()"


def test_indexer():
    # Create a dummy input with multiple channels
    x = np.random.rand(3, 10, 20)

    # Apply the indexer transform
    indexer_transform = Indexer(index=1)
    indexed_x = indexer_transform(x)

    # Check if the indexed data has the correct shape
    assert indexed_x.shape == (10, 20)

    # Check if the indexed data matches the expected channel
    assert np.array_equal(indexed_x, x[1])

    # Check the string representation
    assert str(indexer_transform) == "Indexer(index=1)"


def test_indexer_invalid_index():
    # Create a dummy input with multiple channels
    x = np.random.rand(3, 10, 20)

    # Apply the indexer transform with an invalid index
    indexer_transform = Indexer(index=5)

    # Check if an exception is raised
    with pytest.raises(IndexError):
        indexed_x = indexer_transform(x)


def test_repeat():
    # Create a dummy input
    x = np.random.rand(10, 20)

    # Apply the repeat transform
    repeat_transform = Repeat(axis=0, n_repetitions=2)
    repeated_x = repeat_transform(x)

    # Check if the repeated data has the correct shape
    assert repeated_x.shape == (20, 20)

    # Check if the repeated data matches the expected output
    assert np.array_equal(repeated_x, np.repeat(x, 2, axis=0))

    # Check the string representation
    assert str(repeat_transform) == "Repeat(axis=0, n_repetitions=2)"


def test_repeat_invalid_axis():
    # Create a dummy input
    x = np.random.rand(10, 20)

    # Apply the repeat transform with an invalid axis
    repeat_transform = Repeat(axis=2, n_repetitions=2)

    # Check if an exception is raised
    with pytest.raises(IndexError):
        repeated_x = repeat_transform(x)


def test_cast_to():
    # Create a dummy input
    x = np.random.rand(10, 20)

    # Apply the CastTo transform to cast to float32
    cast_to_transform = CastTo(dtype=np.float32)
    casted_x = cast_to_transform(x)

    # Check if the casted data has the same shape as the input
    assert casted_x.shape == x.shape

    # Check if the casted data has the correct dtype
    assert casted_x.dtype == np.float32

    # Check if the data content remains unchanged
    assert np.allclose(casted_x, x)

    # Check the string representation
    assert str(cast_to_transform) == "CastTo(dtype=<class 'numpy.float32'>)"


def test_cast_to_invalid_dtype():
    # Create a dummy input
    x = np.random.rand(10, 20)

    # Apply the CastTo transform with an invalid dtype
    cast_to_transform = CastTo(dtype="invalid_dtype")

    # Check if an exception is raised
    with pytest.raises(TypeError):
        casted_x = cast_to_transform(x)


def test_color_jitter_output_shape_and_effect():
    x = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    transform = ColorJitter(brightness=0.8, contrast=1.2, saturation=1.1, hue=10)
    y = transform(x)

    assert y.shape == x.shape
    assert not np.array_equal(x, y)  # Check that the image has changed


def test_crop_output_shape():
    x = np.random.randint(0, 256, size=(50, 50, 3), dtype=np.uint8)
    transform = Crop(output_size=(30, 30), coords=(0.5, 0.5))
    y = transform(x)

    assert y.shape == (30, 30, 3)


def test_crop_with_padding():
    x = np.random.randint(0, 256, size=(20, 20, 3), dtype=np.uint8)
    transform = Crop(output_size=(40, 40), coords=(0.0, 0.0))
    y = transform(x)

    assert y.shape == (40, 40, 3)


def test_grayscale_output():
    x = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    transform = GrayScale(gray=128)
    y = transform(x)

    assert y.shape == x.shape
    assert np.all(y[..., 0] == 128)
    assert np.all(y[..., 1] == 128)
    assert np.all(y[..., 2] == 128)


def test_solarize_output_shape_and_effect():
    x = np.random.randint(0, 256, size=(50, 50, 3), dtype=np.uint8)
    transform = Solarize(threshold=128)
    y = transform(x)

    assert y.shape == x.shape
    # Some values should be inverted
    assert not np.array_equal(x, y)


def test_rotation_output_shape_and_visual_change():
    x = np.random.randint(0, 256, size=(60, 80, 3), dtype=np.uint8)
    transform = Rotation(degrees=45)
    y = transform(x)

    assert y.shape == x.shape
    assert not np.array_equal(x, y)
