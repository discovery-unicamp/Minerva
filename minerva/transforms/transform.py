from itertools import product
from typing import Any, List, Sequence, Union, Tuple
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
            return np.flip(x, axis=self.axis)

        assert (
            len(self.axis) <= x.ndim
        ), "Axis list has more dimentions than input data. The lenth of axis needs to be less or equal to input dimentions."

        for axis in self.axis:
            x = np.flip(x, axis=axis)

        return x
    

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
    
    
class Gradient(_Transform):
    def __init__(self, direction: int):
        
        '''
        direction: 
            0 -> Gradient along the x-axis (width)
            1 -> Gradient along the y-axis (height)
        '''
        
        assert direction in [0, 1], "Direction must be 0 (x-axis) or 1 (y-axis)"
        self.direction = direction

    def generate_gradient(self, shape: tuple[int, int]) -> np.ndarray:              
        
        '''
        Inputs in format (H, W) 
        Outputs a gradient from 0 to 1 in either x or y direction based on the direction parameter
        '''
        
        xx, yy = np.meshgrid(np.linspace(0, 1, shape[1]), np.linspace(0, 1, shape[0]))

        if self.direction == 0:  # Gradient along the x-axis
            return xx
        elif self.direction == 1:  # Gradient along the y-axis
            return yy

    def __call__(self, x):
        if x.ndim == 2: 
            shape = x.shape
        else: shape = x.shape[1:]
        gradient = self.generate_gradient(shape)  # Generate gradient in the specified direction
        
        x_expanded = np.expand_dims(x, axis=0) if x.ndim == 2 else x
        gradient_expanded = np.expand_dims(gradient, axis=0)
        
        output = np.concatenate([x_expanded, gradient_expanded], axis=0)

        assert output.shape == (x_expanded.shape[0] + 1, shape[0], shape[1]), \
            f"Output shape {output.shape} does not match expected shape {(shape[0], shape[1], x_expanded.shape[0] + 1)}"
        
        return output


class ColorJitter(_Transform):
    def __init__(self, brightness: float = 1.0, contrast: float = 1.0, saturation: float = 1.0, hue: float = 0.0):
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


class Crop(_Transform):
    def __init__(self, output_size: Tuple[int, int], pad_mode: str = 'reflect', coords: Tuple[int, int] = (0, 0)):
        """
        Crops the input image to a specified output size, with optional padding if needed.

        Parameters
        ----------
        output_size : Tuple[int, int]
            Desired output size as (height, width).
        pad_mode : str, optional
            Padding mode used if output size is larger than input size. Defaults to 'reflect'.
        coords : Tuple[int, int], optional
            Top-left coordinates for the crop box. Defaults to (0, 0).

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
            image = np.pad(image, ((pad_h // 2, pad_h - pad_h // 2), 
                                   (pad_w // 2, pad_w - pad_w // 2), 
                                   (0, 0)), mode=self.pad_mode)

        # Update dimensions after padding
        h, w = image.shape[:2]

        return image[X:X + new_h, Y:Y + new_w]   
    
    
class GrayScale(_Transform):
    def __init__(self, gray: float = 0.0):
        """
        Converts an image to grayscale with a specified gray value.

        Parameters
        ----------
        gray : float, optional
            Gray value to use when converting the image. Defaults to 0.0.

        Returns
        -------
        np.ndarray
            Grayscale image in RGB format with all channels set to `gray`.
        """
        self.gray = gray

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return np.stack([self.gray] * 3, axis=-1)  # Convert grayscale to RGB format   


class SolarizeTransform(_Transform):
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
            solarized_channels = [np.where(channel < self.threshold, channel, 255 - channel) for channel in channels]
            solarized_image = cv2.merge(solarized_channels)
        else:  # Grayscale image
            solarized_image = np.where(image < self.threshold, image, 255 - image)
        
        return solarized_image  
    
    
class Rotation(_Transform):
    def __init__(self, degrees: float):
        """
        Rotates the image by a specified angle.

        Parameters
        ----------
        degrees : float
            Angle in degrees to rotate the image.

        Returns
        -------
        np.ndarray
            Rotated image with reflection padding.
        """
        self.degrees = degrees

    def __call__(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, self.degrees, 1.0)
        return cv2.warpAffine(image, rotation_matrix, (w, h), 
                              borderMode=cv2.BORDER_REFLECT)  
    