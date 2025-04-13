from typing import Sequence

import numpy as np
import pytest

from minerva.transforms import ActivityImageTransforms


def test_activity_image_transforms():
    transform = ActivityImageTransforms(9, 68)

    # Create a dummy input
    x = np.random.rand(9, 128)

    # Apply the transform
    x, transformed_x = transform(x)

    # Output shape
    y = np.random.rand(37, 68)

    assert transformed_x.shape == y.shape
