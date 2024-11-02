from itertools import product
from typing import Any, List, Literal, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from perlin_noise import PerlinNoise


class _Transform:
    """This class is a base class for all transforms. Transforms is just a
    fancy word for a function that takes an input and returns an output. The
    input and output can be anything. However, transforms operates over a
    single sample of data and does not require any additional information to
    perform the transformation. The __call__ method should be overridden in
    subclasses to define the transformation logic.
    """

    def __call__(self, *args, **kwargs) -> Any:
        """Implement the transformation logic in this method. Usually, the
        transformation is applyied on a single sample of data.
        """
        raise NotImplementedError


class TransformPipeline(_Transform):
    """Apply a sequence of transforms to a single sample of data and return the
    transformed data.
    """

    def __init__(self, transforms: Sequence[_Transform]):
        """Apply a sequence of transforms to a single sample of data and return
        the transformed data.

        Parameters
        ----------
        transforms : List[_Transform]
            A list of transforms to be applied to the input data.
        """
        self.transforms = transforms

    def __call__(self, x: Any) -> Any:
        """Apply a sequence of transforms to a single sample of data and return
        the transformed data.
        """
        for transform in self.transforms:
            x = transform(x)
        return x


class Flip(_Transform):
    """Flip the input data along the specified axis."""

    def __init__(self, axis: Union[int, List[int]] = 0):
        """Flip the input data along the specified axis.

        Parameters
        ----------
        axis : int | List[int], optional
            One or more axis to flip the input data along, by default 0.
            If a list of axis is provided, the input data is flipped along all the specified axis in the order they are provided.
        """
        self.axis = axis

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Flip the input data along the specified axis.
        if axis is an integer, the input data is flipped along the specified axis.
        if axis is a list of integers, the input data is flipped along all the specified axis in the order they are provided.
        The input must have the same, or less, number of dimensions as the length of the list of axis.
        """

        if isinstance(self.axis, int):
            return np.flip(x, axis=self.axis).copy()

        assert (
            len(self.axis) <= x.ndim
        ), "Axis list has more dimentions than input data. The lenth of axis needs to be less or equal to input dimentions."

        for axis in self.axis:
            x = np.flip(x, axis=axis)

        return x.copy()


class PerlinMasker(_Transform):
    """Zeroes entries of a tensor according to the sign of Perlin noise. Seed for the noise generator given by torch.randint"""

    def __init__(self, octaves: int, scale: float = 1):
        """Zeroes entries of a tensor according to the sign of Perlin noise. Seed for the noise generator given by torch.randint

        Parameters
        ----------
        octaves: int
            Level of detail for the Perlin noise generator
        scale: float = 1
            Optionally rescale the Perlin noise. Default is 1 (no rescaling)
        """
        if octaves <= 0:
            raise ValueError(f"Number of octaves must be positive, but got {octaves=}")
        if scale == 0:
            raise ValueError(f"Scale can't be 0")
        self.octaves = octaves
        self.scale = scale

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Zeroes entries of a tensor according to the sign of Perlin noise.

        Parameters
        ----------
        x: np.ndarray
            The tensor whose entries to zero.
        """

        mask = np.empty_like(x, dtype=bool)
        noise = PerlinNoise(self.octaves, torch.randint(0, 2**32, (1,)).item())
        denom = self.scale * max(x.shape)

        for pos in product(*[range(i) for i in mask.shape]):
            mask[pos] = noise([i / denom for i in pos]) < 0

        return x * mask


class Squeeze(_Transform):
    """Remove single-dimensional entries from the shape of an array."""

    def __init__(self, axis: int):
        """Remove single-dimensional entries from the shape of an array.

        Parameters
        ----------
        axis : int
            The position of the axis to be removed.
        """
        self.axis = axis

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Remove single-dimensional entries from the shape of an array."""
        return np.squeeze(x, axis=self.axis)


class Unsqueeze(_Transform):
    """Add a new axis to the input data at the specified position."""

    def __init__(self, axis: int):
        """Add a new axis to the input data at the specified position.

        Parameters
        ----------
        axis : int
            The position of the new axis to be added.
        """
        self.axis = axis

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Add a new axis to the input data at the specified position."""
        return np.expand_dims(x, axis=self.axis)


class CastTo(_Transform):
    """Cast the input data to the specified data type."""

    def __init__(self, dtype: Union[type, str]):
        """Cast the input data to the specified data type.

        Parameters
        ----------
        dtype : type
            The data type to which the input data will be cast.
        """
        self.dtype = dtype

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Cast the input data to the specified data type."""
        return x.astype(self.dtype)


class Padding(_Transform):
    def __init__(
        self,
        target_h_size: int,
        target_w_size: int,
        padding_mode: Literal["reflect", "constant"] = "reflect",
        constant_value: int = 0,
        mask_value: int = 255,
    ):
        self.target_h_size = target_h_size
        self.target_w_size = target_w_size
        self.padding_mode = padding_mode
        self.constant_value = constant_value
        self.mask_value = mask_value

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h, w = x.shape[:2]
        pad_h = max(0, self.target_h_size - h)
        pad_w = max(0, self.target_w_size - w)
        is_label = True if x.dtype == np.uint8 else False

        if len(x.shape) == 2:
            if self.padding_mode == "reflect":
                padded = np.pad(x, ((0, pad_h), (0, pad_w)), mode="reflect")
            elif self.padding_mode == "constant":
                if is_label:
                    padded = np.pad(
                        x,
                        ((0, pad_h), (0, pad_w)),
                        mode="constant",
                        constant_values=self.mask_value,
                    )
                else:
                    padded = np.pad(
                        x,
                        ((0, pad_h), (0, pad_w)),
                        mode="constant",
                        constant_values=self.constant_value,
                    )
            padded = np.expand_dims(padded, axis=2)

        else:
            if self.padding_mode == "reflect":
                padded = np.pad(x, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            elif self.padding_mode == "constant":
                if is_label:
                    padded = np.pad(
                        x,
                        ((0, pad_h), (0, pad_w), (0, 0)),
                        mode="constant",
                        constant_values=self.mask_value,
                    )
                else:
                    padded = np.pad(
                        x,
                        ((0, pad_h), (0, pad_w), (0, 0)),
                        mode="constant",
                        constant_values=self.constant_value,
                    )
        return padded


class Normalize(_Transform):
    def __init__(self, mean, std, to_rgb=False, normalize_labels=False):
        """
        Initialize the Normalize transform.

        Args:
            means (list or tuple): A list or tuple containing the mean for each channel.
            stds (list or tuple): A list or tuple containing the standard deviation for each channel.
            to_rgb (bool): If True, convert the data from BGR to RGB.
        """
        assert len(mean) == len(
            std
        ), "Means and standard deviations must have the same length."
        self.mean = mean
        self.std = std
        self.to_rgb = to_rgb
        self.normalize_labels = normalize_labels

    def __call__(self, data):
        """
        Normalize the input data using the provided means and standard deviations.

        Args:
            data (numpy.ndarray): Input data array of shape (C, H, W) where C is the number of channels.

        Returns:
            numpy.ndarray: Normalized data.
        """

        is_label = True if data.dtype == np.uint8 else False

        if is_label and self.normalize_labels:
            # Convert from gray scale (1 channel) to RGB (3 channels) if to_rgb is True
            if self.to_rgb and data.shape[0] == 1:
                data = np.repeat(data, 3, axis=0)

            assert data.shape[0] == len(
                self.mean
            ), f"Number of channels in data does not match the number of provided mean/std. {data.shape}"

            # Normalize each channel
            for i in range(len(self.mean)):
                data[i, :, :] = (data[i, :, :] - self.mean[i]) / self.std[i]

        return data


class Crop(_Transform):
    def __init__(
        self,
        target_h_size: int,
        target_w_size: int,
        start_coord: Tuple[int, int] = (0, 0),
    ):
        self.target_h_size = target_h_size
        self.target_w_size = target_w_size
        self.start_coord = start_coord

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h, w = x.shape[:2]
        start_h = (h - self.target_h_size) // 2
        start_w = (w - self.target_w_size) // 2
        end_h = start_h + self.target_h_size
        end_w = start_w + self.target_w_size
        if len(x.shape) == 2:
            cropped = x[start_h:end_h, start_w:end_w]
            cropped = np.expand_dims(cropped, axis=2)
        else:
            cropped = x[start_h:end_h, start_w:end_w]

        return cropped


class Transpose(_Transform):
    """Reorder the axes of numpy arrays."""

    def __init__(self, axes: Sequence[int]):
        """Reorder the axes of numpy arrays.

        Parameters
        ----------
        axes : int
            The order of the new axes
        """
        self.axes = axes

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Reorder the axes of numpy arrays."""
        return np.transpose(x, self.axes)


class Resize(_Transform):

    def __init__(
        self,
        target_h_size: int,
        target_w_size: int,
        keep_aspect_ratio: bool = False,
    ):
        self.target_h_size = target_h_size
        self.target_w_size = target_w_size
        self.keep_aspect_ratio = keep_aspect_ratio

    def __call__(self, x: np.ndarray) -> np.ndarray:

        if not self.keep_aspect_ratio:
            return cv2.resize(
                x,
                (self.target_w_size, self.target_h_size),
                interpolation=cv2.INTER_NEAREST,
            )

        original_height, original_width = x.shape[:2]

        # Calculate scaling factors to fit within max_size, preserving aspect ratio
        width_scale = self.target_w_size / original_width
        height_scale = self.target_h_size / original_height
        scale = min(
            width_scale, height_scale
        )  # Choose the smaller scale to fit within dimensions

        # Calculate new dimensions based on the correct scale factor
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        return cv2.resize(x, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    # Convert the resized PIL Image back to a NumPy array
