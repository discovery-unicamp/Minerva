from itertools import product
from typing import Any, List, Optional, Sequence, Tuple, Union, Literal

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
        transformation is applied on a single sample of data.
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

    def __add__(self, other: _Transform) -> "TransformPipeline":
        """Add a transform to the pipeline."""
        if isinstance(other, TransformPipeline):
            return TransformPipeline(list(self.transforms) + list(other.transforms))
        return TransformPipeline(list(self.transforms) + [other])

    def __radd__(self, other: _Transform) -> "TransformPipeline":
        """Add a transform to the pipeline."""
        return self.__add__(other)

    def __str__(self) -> str:
        return f"TransformPipeline(transforms=[{', '.join([str(t) for t in self.transforms])}])"


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
        ), "Axis list has more dimensions than input data. The length of axis needs to be less or equal to input dimensions."

        for axis in self.axis:
            x = np.flip(x, axis=axis)

        return x.copy()

    def __str__(self) -> str:
        return f"Flip(axis={self.axis})"


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

    def __str__(self) -> str:
        return f"PerlinMasker(octaves={self.octaves}, scale={self.scale})"


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

    def __str__(self) -> str:
        return f"Squeeze(axis={self.axis})"


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

    def __str__(self) -> str:
        return f"Unsqueeze(axis={self.axis})"


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

    def __str__(self) -> str:
        return f"Transpose(axes={self.axes})"


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

    def __str__(self) -> str:
        return f"CastTo(dtype={self.dtype})"


class Padding(_Transform):
    def __init__(self, target_h_size: int, target_w_size: int):
        self.target_h_size = target_h_size
        self.target_w_size = target_w_size

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h, w = x.shape[:2]
        pad_h = max(0, self.target_h_size - h)
        pad_w = max(0, self.target_w_size - w)
        if len(x.shape) == 2:
            padded = np.pad(x, ((0, pad_h), (0, pad_w)), mode="reflect")
            padded = np.expand_dims(padded, axis=2)
        else:
            padded = np.pad(x, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

        padded = np.transpose(padded, (2, 0, 1))
        return padded

    def __str__(self) -> str:
        return f"Padding(target_h_size={self.target_h_size}, target_w_size={self.target_w_size})"


class Gradient(_Transform):
    directions = {0: "x (width)", 1: "y (height)"}

    def __init__(self, direction: int):
        """
        direction:
            0 -> Gradient along the x-axis (width)
            1 -> Gradient along the y-axis (height)
        """

        assert direction in [0, 1], "Direction must be 0 (x-axis) or 1 (y-axis)"
        self.direction = direction

    def generate_gradient(self, shape: tuple[int, int]) -> np.ndarray:
        """
        Inputs in format (H, W)
        Outputs a gradient from 0 to 1 in either x or y direction based on the direction parameter
        """

        xx, yy = np.meshgrid(np.linspace(0, 1, shape[1]), np.linspace(0, 1, shape[0]))

        if self.direction == 0:  # Gradient along the x-axis
            return xx
        elif self.direction == 1:  # Gradient along the y-axis
            return yy

    def __call__(self, x):
        if x.ndim == 2:
            shape = x.shape
        else:
            shape = x.shape[1:]
        gradient = self.generate_gradient(
            shape
        )  # Generate gradient in the specified direction

        x_expanded = np.expand_dims(x, axis=0) if x.ndim == 2 else x
        gradient_expanded = np.expand_dims(gradient, axis=0)

        output = np.concatenate([x_expanded, gradient_expanded], axis=0)

        assert output.shape == (
            x_expanded.shape[0] + 1,
            shape[0],
            shape[1],
        ), f"Output shape {output.shape} does not match expected shape {(shape[0], shape[1], x_expanded.shape[0] + 1)}"

        return output

    def __str__(self) -> str:
        return (
            f"Gradient(direction={self.direction} - {self.directions[self.direction]})"
        )


class ColorJitter(_Transform):
    def __init__(
        self,
        brightness: float = 1.0,
        contrast: float = 1.0,
        saturation: float = 1.0,
        hue: float = 0.0,
    ):
        """
        Applies fixed adjustments to brightness, contrast, saturation, and hue to an input image.

        Parameters
        ----------
        brightness : float, optional
            Fixed factor for brightness adjustment. A value of 1.0 means no change. Defaults to 1.0.
        contrast : float, optional
            Fixed factor for contrast adjustment. A value of 1.0 means no change. Defaults to 1.0.
        saturation : float, optional
            Fixed factor for saturation adjustment. A value of 1.0 means no change. Defaults to 1.0.
        hue : float, optional
            Fixed degree shift for hue adjustment, in the range [-180, 180]. Defaults to 0.0.

        Returns
        -------
        np.ndarray
            The transformed image with fixed brightness, contrast, saturation, and hue adjustments applied.
        """
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, image: np.ndarray) -> np.ndarray:
        # Convert to HSV for hue/saturation adjustment
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)

        # Brightness adjustment
        image[..., 2] = np.clip(image[..., 2] * self.brightness, 0, 255)

        # Saturation adjustment
        image[..., 1] = np.clip(image[..., 1] * self.saturation, 0, 255)

        # Contrast adjustment
        mean = image[..., 2].mean()
        image[..., 2] = np.clip((image[..., 2] - mean) * self.contrast + mean, 0, 255)

        # Hue adjustment
        image[..., 0] = (image[..., 0] + self.hue) % 180

        return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_HSV2RGB)

    def __str__(self) -> str:
        return f"ColorJitter(brightness={self.brightness}, contrast={self.contrast}, saturation={self.saturation}, hue={self.hue})"


class Crop(_Transform):
    def __init__(
        self,
        output_size: Tuple[int, int],
        pad_mode: str = "reflect",
        coords: Tuple[float, float] = (0, 0),
    ):
        """
        Crops the input image to a specified output size, with optional padding if needed.

        Parameters
        ----------
        output_size : Tuple[int, int]
            Desired output size as (height, width).
        pad_mode : str, optional
            Padding mode used if output size is larger than input size. Defaults to 'reflect'.
        coords : Tuple[int, int], optional
            Top-left coordinates for the crop box.
            Values must go from 0 to 1 indicating the relative position on where the
            new top-left corner can be set, taking in consideration the new size

        Returns
        -------
        np.ndarray
            Cropped image, padded as necessary.
        """
        self.output_size = output_size
        self.pad_mode = pad_mode
        self.coords = coords

    def __call__(self, image: np.ndarray) -> np.ndarray:
        X, Y = self.coords
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        # Apply padding if output size is larger than input size
        if new_h > h or new_w > w:
            pad_h = max(new_h - h, 0)
            pad_w = max(new_w - w, 0)
            image = np.pad(
                image,
                (
                    (pad_h // 2, pad_h - pad_h // 2),
                    (pad_w // 2, pad_w - pad_w // 2),
                    (0, 0),
                ),
                mode=self.pad_mode,
            )

        # Update dimensions after padding
        h, w = image.shape[:2]

        x = int((h - new_h) * X)
        y = int((w - new_w) * Y)

        return image[x : x + new_h, y : y + new_w]

    def __str__(self) -> str:
        return f"Crop(output_size={self.output_size}, pad_mode={self.pad_mode}, coords={self.coords})"


class GrayScale(_Transform):
    def __init__(self, method: Literal["average", "luminosity"] = "luminosity"):
        """
        Converts an image to grayscale using the specified method.

        Parameters
        ----------
        method : {'average', 'luminosity'}, optional
            The method to compute grayscale:
            - 'average': (R + G + B) / 3
            - 'luminosity': 0.299R + 0.587G + 0.114B
            Defaults to 'luminosity'.
        """
        if method not in {"average", "luminosity"}:
            raise ValueError("method must be 'average' or 'luminosity'")
        self.method = method

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Applies grayscale conversion to the input RGB image.

        Parameters
        ----------
        image : np.ndarray
            Input image in RGB format with shape (H, W, 3).

        Returns
        -------
        np.ndarray
            Grayscale image with shape (H, W, 3) where all channels are equal.
        """
        assert (
            image.ndim == 3 and image.shape[2] == 3
        ), "Input must have shape (H, W, 3)"

        if self.method == "average":
            gray = image.mean(axis=2)
        else:  # luminosity
            weights = np.array([0.299, 0.587, 0.114])
            gray = np.dot(image[..., :3], weights)

        return np.stack([gray, gray, gray], axis=-1).astype(image.dtype)

    def __str__(self) -> str:
        return f"GrayScale(method='{self.method}')"


class Solarize(_Transform):
    def __init__(self, threshold: int = 128):
        """
        Solarizes the image by inverting pixel values above a specified threshold.

        Parameters
        ----------
        threshold : int, optional
            Intensity threshold for inversion, default is 128.

        Returns
        -------
        np.ndarray
            Solarized image with inverted pixel values above threshold.
        """
        self.threshold = threshold

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:  # Color image
            channels = cv2.split(image)
            solarized_channels = [
                np.where(channel < self.threshold, channel, 255 - channel)
                for channel in channels
            ]
            solarized_image = cv2.merge(solarized_channels)
        else:  # Grayscale image
            solarized_image = np.where(image < self.threshold, image, 255 - image)

        return solarized_image

    def __str__(self):
        return f"Solarize(threshold={self.threshold})"


class Rotation(_Transform):
    def __init__(self, degrees: float):
        """
        Rotates an image by a specified angle using reflection padding.

        Parameters
        ----------
        degrees : float
            Angle in degrees to rotate the image counterclockwise.

        Notes
        -----
        - Accepts input with shape (H, W) or (H, W, C), where C can be any number of channels.
        - For multi-channel images, the same transformation is applied to all channels.
        - Uses OpenCV's warpAffine with reflection padding.
        """
        self.degrees = degrees

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if image.ndim not in (2, 3):
            raise ValueError(
                f"Unsupported image shape: {image.shape}. Expected 2D or 3D with channels last."
            )

        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, self.degrees, 1.0)

        if image.ndim == 2:
            # Single-channel 2D image
            return cv2.warpAffine(
                image, rotation_matrix, (w, h), borderMode=cv2.BORDER_REFLECT
            )

        # Multi-channel image (H, W, C)
        channels = [
            cv2.warpAffine(
                image[:, :, c], rotation_matrix, (w, h), borderMode=cv2.BORDER_REFLECT
            )
            for c in range(image.shape[2])
        ]
        return np.stack(channels, axis=-1)

    def __str__(self) -> str:
        return f"Rotation(degrees={self.degrees})"


class PadCrop(_Transform):
    def __init__(
        self,
        target_h_size: int,
        target_w_size: int,
        padding_mode: str = "reflect",
        seed: Optional[int] = None,
        constant_values: int = 0,
    ):
        """Transforms image and pads or crops it to the target size. If the
        target size is larger than the input size, the image is padded, else,
        the image is cropped. The same happens for both height and width.
        The padding mode can be specified, as well as the seed for the random
        number generator.

        For padding, the padding is applied symmetrically on both sides of the
        image, thus, image will be centered in the padded image. For cropping,
        the crop is applied from a random position in the image.

        Image is expected to be in C x H x W, or H x W format.

        Parameters
        ----------
        target_h_size : int
            Desired height size.
        target_w_size : int
            Desired width size.
        padding_mode : str, optional
            The padding mode to use, by default "reflect"
        seed : int, optional
            The seed for the random number generator. It is used to generate
            the random crop position. By default, None.
        constant_values : int, optional
            If padding mode is 'constant', the value to use for padding. By
            default 0.
        """
        self.target_h_size = target_h_size
        self.target_w_size = target_w_size
        self.padding_mode = padding_mode
        self.seed = seed
        self.rng = np.random.default_rng(
            seed
        )  # Random number generator with the provided seed
        self.constant_values = constant_values

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # Input is expected to be in C x H x W format or H x W format

        # If input is in C x H x W format, convert to H x W x C format
        if len(x.shape) == 3:
            x = np.transpose(x, (1, 2, 0))

        # Get the height and width of the input image (H and W)
        h, w = x.shape[:2]

        #### HEIGHT ####
        # Handle height dimension independently: pad if target_h_size > h, else crop
        if self.target_h_size > h:
            pad_h = self.target_h_size - h
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_args = {
                "array": x,
                "pad_width": (
                    ((pad_top, pad_bottom), (0, 0), (0, 0))
                    if len(x.shape) == 3
                    else ((pad_top, pad_bottom), (0, 0))
                ),
                "mode": self.padding_mode,
            }
            if self.padding_mode == "constant":
                pad_args["constant_values"] = self.constant_values

            x = np.pad(**pad_args)

        elif self.target_h_size < h:
            crop_h_start = self.rng.integers(0, h - self.target_h_size + 1)
            x = x[crop_h_start : crop_h_start + self.target_h_size, ...]

        #### WIDTH ####
        # Handle width dimension independently: pad if target_w_size > w, else crop
        if self.target_w_size > w:
            pad_w = self.target_w_size - w
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left

            pad_args = {
                "array": x,
                "pad_width": (
                    ((0, 0), (pad_left, pad_right), (0, 0))
                    if len(x.shape) == 3
                    else ((0, 0), (pad_left, pad_right))
                ),
                "mode": self.padding_mode,
            }

            if self.padding_mode == "constant":
                pad_args["constant_values"] = self.constant_values

            x = np.pad(**pad_args)

        elif self.target_w_size < w:
            crop_w_start = self.rng.integers(0, w - self.target_w_size + 1)
            x = x[:, crop_w_start : crop_w_start + self.target_w_size, ...]

        #  If input is 3D, convert back to C x H x W format
        if len(x.shape) == 3:
            x = np.transpose(x, (2, 0, 1))
        return x

    def __str__(self) -> str:
        return f"PadCrop(target_h_size={self.target_h_size}, target_w_size={self.target_w_size}, padding_mode={self.padding_mode}, constant_values={self.constant_values}, seed={self.seed})"


class Identity(_Transform):
    """This class is a dummy transform that does nothing. It is useful when
    you want to skip a transform in a pipeline.
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x

    def __str__(self) -> str:
        return "Identity()"


class Indexer(_Transform):
    def __init__(self, index: int):
        """This transform extracts a single channel from a multi-channel image.

        Parameters
        ----------
        index : int
            The index of the channel to extract.
        """
        self.index = index

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x[self.index]

    def __str__(self) -> str:
        return f"Indexer(index={self.index})"


class Repeat(_Transform):
    def __init__(self, axis: int, n_repetitions: int):
        """This transform repeats the input data along the specified axis.

        Parameters
        ----------
        axis : int
            The axis along which to repeat the input data.
        n_repetitions : int
            The number of repetitions.
        """
        self.axis = axis
        self.n_repetitions = n_repetitions

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.repeat(x, self.n_repetitions, axis=self.axis)

    def __str__(self) -> str:
        return f"Repeat(axis={self.axis}, n_repetitions={self.n_repetitions})"


class Normalize(_Transform):
    def __init__(self, mean, std, to_rgb=False, normalize_labels=False):
        """
        Normalize the input data using the provided means and standard deviations.

        Parameters
        ----------
        mean : List[float]
            List of means for each channel.
        std : List[float]
            List of standard deviations for each channel.
        to_rgb : bool, optional
            Convert grayscale images to RGB format, by default False.
        normalize_labels : bool, optional
            Normalize label images, by default False.

        """
        assert len(mean) == len(
            std
        ), "Means and standard deviations must have the same length."
        self.mean = mean
        self.std = std
        self.to_rgb = to_rgb
        self.normalize_labels = normalize_labels

    def __call__(self, data):

        is_label = True if data.dtype == np.uint8 else False

        if (is_label and self.normalize_labels) or not is_label:
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

    def __str__(self) -> str:
        return f"Normalize(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb}, normalize_labels={self.normalize_labels})"


class ContrastiveTransform(_Transform):
    def __init__(self, transform: _Transform):
        self.transform = transform

    def __call__(self, x: np.ndarray) -> Tuple:
        return self.transform(x), self.transform(x)

    def __str__(self) -> str:
        return f"ContrastiveTransform(transform={self.transform})"
