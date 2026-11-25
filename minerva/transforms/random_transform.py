"""Random transforms.

Each call draws its random parameters once (with torch's RNG, which
``DataLoader`` seeds per worker) and applies the same parameters to every key
in ``keys_to_transform``. This is what keeps image and label in sync. Never
list several view keys on one random transform to get different augmentations
per view; use :class:`~minerva.transforms.transform.MultiViewTransform`.
"""

import math
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from minerva.transforms.transform import (
    CropTransform,
    AmplitudeScalingTransform,
    FlipTransform,
    GrayScaleTransform,
    ResizeTransform,
    RotationTransform,
    SolarizeTransform,
    VerticalPatchSwapTransform,
    _as_keys,
    _interpolation,
    _resize_spatial,
    _resolve_modes,
    _RandomTransform,
)


class RandomFlipTransform(_RandomTransform):
    """Flip every listed key with the same random decision."""

    def __init__(
        self,
        keys_to_transform: List[str],
        possible_axis: Union[int, List[int]] = 1,
        prob: float = 0.5,
        seed: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        keys_to_transform : List[str]
            Keys of the entries to flip. All listed keys share one random draw.
        possible_axis : int | List[int], optional
            Axis (or axes) that may be flipped, by default 1. With a list,
            each axis is flipped independently with probability ``prob``.
        prob : float, optional
            Probability of flipping, by default 0.5.
        seed : int, optional
            Seed for reproducible draws, by default None (torch's global RNG).
        """
        assert 0.0 <= prob <= 1.0, "prob must be between 0 and 1"
        self._init_rng(seed)
        self.keys_to_transform = _as_keys(keys_to_transform)
        self.possible_axis = possible_axis
        self.prob = prob

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(self.possible_axis, int):
            axis: Union[int, List[int], None] = (
                self.possible_axis if self._rand() < self.prob else None
            )
        else:
            chosen = [a for a in self.possible_axis if self._rand() < self.prob]
            axis = chosen or None
        if axis is not None:
            FlipTransform(self.keys_to_transform, axis=axis)(data)
        return data


class RandomAmplitudeScalingTransform(_RandomTransform):
    """Scale listed continuous signals by one random factor per call.

    This simulates variation in sensor gain without changing the timing or
    shape of the signal pattern.

    Categorical masks and labels should be left out of ``keys_to_transform``;
    scaling their integer class values would corrupt their meaning.
    """

    def __init__(
        self,
        keys_to_transform: List[str],
        scale_range: Tuple[float, float] = (0.8, 1.2),
        prob: float = 1.0,
        seed: Optional[int] = None,
    ):
        """Initialize random amplitude scaling.

        Parameters
        ----------
        keys_to_transform : List[str]
            Continuous signal keys to scale. All listed keys share one factor;
            unlisted entries, including masks, pass through unchanged.
        scale_range : Tuple[float, float], optional
            Bounds for the uniformly sampled factor, by default (0.8, 1.2).
        prob : float, optional
            Probability of scaling, by default 1.0.
        seed : int, optional
            Seed for reproducible draws, by default None (torch's global RNG).
        """
        low, high = scale_range
        if not math.isfinite(low) or not math.isfinite(high) or low < 0 or low > high:
            raise ValueError(
                "scale_range must be finite, non-negative, and ordered."
            )
        if not 0.0 <= prob <= 1.0:
            raise ValueError("prob must be between 0 and 1.")
        self._init_rng(seed)
        self.keys_to_transform = _as_keys(keys_to_transform)
        self.scale_range = scale_range
        self.prob = prob

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self._rand() < self.prob:
            factor = self._uniform(*self.scale_range)
            AmplitudeScalingTransform(self.keys_to_transform, factor)(data)
        return data


class RandomCropTransform(_RandomTransform):
    """Crop every listed key at the same random position."""

    def __init__(
        self,
        keys_to_transform: List[str],
        crop_size: Tuple[int, int],
        pad_mode: str = "reflect",
        seed: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        keys_to_transform : List[str]
            Keys of the entries to crop. All listed keys share one position.
        crop_size : Tuple[int, int]
            Output size as (height, width).
        pad_mode : str, optional
            Padding mode if the crop is larger than the input, by default "reflect".
        seed : int, optional
            Seed for reproducible draws, by default None (torch's global RNG).
        """
        self._init_rng(seed)
        self.keys_to_transform = _as_keys(keys_to_transform)
        self.crop_size = crop_size
        self.pad_mode = pad_mode

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        coords = (self._rand(), self._rand())
        return CropTransform(
            self.keys_to_transform, self.crop_size, self.pad_mode, coords
        )(data)


class RandomGrayScaleTransform(_RandomTransform):
    """Convert every listed key to grayscale with the same random decision."""

    def __init__(
        self,
        keys_to_transform: List[str],
        prob: float = 0.1,
        method: Literal["average", "luminosity"] = "luminosity",
        seed: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        keys_to_transform : List[str]
            Keys of the images to convert. They share one random draw.
        prob : float, optional
            Probability of converting, by default 0.1.
        method : {'average', 'luminosity'}, optional
            Grayscale method, by default 'luminosity'.
        seed : int, optional
            Seed for reproducible draws, by default None (torch's global RNG).
        """
        self._init_rng(seed)
        self.keys_to_transform = _as_keys(keys_to_transform)
        self.prob = prob
        self.method = method

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self._rand() < self.prob:
            GrayScaleTransform(self.keys_to_transform, self.method)(data)
        return data


class RandomSolarizeTransform(_RandomTransform):
    """Solarize every listed key with the same random decision."""

    def __init__(
        self,
        keys_to_transform: List[str],
        threshold: int = 128,
        prob: float = 1.0,
        seed: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        keys_to_transform : List[str]
            Keys of the images to solarize. They share one random draw.
        threshold : int, optional
            Intensity threshold for inversion, by default 128.
        prob : float, optional
            Probability of solarizing, by default 1.0.
        seed : int, optional
            Seed for reproducible draws, by default None (torch's global RNG).
        """
        self._init_rng(seed)
        self.keys_to_transform = _as_keys(keys_to_transform)
        self.threshold = threshold
        self.prob = prob

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self._rand() < self.prob:
            SolarizeTransform(self.keys_to_transform, self.threshold)(data)
        return data


class RandomRotationTransform(_RandomTransform):
    """Rotate every listed key by the same random angle."""

    def __init__(
        self,
        keys_to_transform: List[str],
        degrees: float,
        prob: float,
        mode: Optional[Sequence[str]] = None,
        seed: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        keys_to_transform : List[str]
            Keys of the entries to rotate. They share one angle.
        degrees : float
            Maximum absolute angle; the angle is drawn uniformly from
            [-degrees, +degrees].
        prob : float
            Probability of rotating.
        mode : Sequence[str], optional
            One interpolation mode per key. Use "nearest" for labels.
            By default "bilinear" for all.
        seed : int, optional
            Seed for reproducible draws, by default None (torch's global RNG).
        """
        self._init_rng(seed)
        self.keys_to_transform = _as_keys(keys_to_transform)
        self.degrees = degrees
        self.prob = prob
        self.mode = _resolve_modes(self.keys_to_transform, mode)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self._rand() < self.prob:
            angle = self._uniform(-self.degrees, self.degrees)
            RotationTransform(self.keys_to_transform, angle, self.mode)(data)
        return data


class RandomResizeTransform(_RandomTransform):
    """Resize every listed key by the same random scale factor."""

    def __init__(
        self,
        keys_to_transform: List[str],
        target_scale: Tuple[int, int],
        ratio_range: Tuple[float, float],
        mode: Optional[Sequence[str]] = None,
        seed: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        keys_to_transform : List[str]
            Keys of the entries to resize. They share one scale factor.
        target_scale : Tuple[int, int]
            Base target size as (height, width).
        ratio_range : Tuple[float, float]
            Range of the scale factor, drawn uniformly and applied to both
            dimensions of ``target_scale``.
        mode : Sequence[str], optional
            One interpolation mode per key. Use "nearest" for labels.
            By default "bilinear" for all.
        seed : int, optional
            Seed for reproducible draws, by default None (torch's global RNG).
        """
        self._init_rng(seed)
        self.keys_to_transform = _as_keys(keys_to_transform)
        self.target_scale = target_scale
        self.ratio_range = ratio_range
        self.mode = _resolve_modes(self.keys_to_transform, mode)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        factor = self._uniform(*self.ratio_range)
        new_h = int(self.target_scale[0] * factor)
        new_w = int(self.target_scale[1] * factor)
        return ResizeTransform(
            self.keys_to_transform, new_h, new_w, mode=self.mode
        )(data)


class RandomResizedCropTransform(_RandomTransform):
    """Crop a random area and aspect ratio, then resize, identically for every listed key.

    Expects (C, H, W) or (H, W) arrays; the last two axes are spatial.
    """

    def __init__(
        self,
        keys_to_transform: List[str],
        size: Tuple[int, int],
        scale: Tuple[float, float] = (0.08, 1.0),
        ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        mode: Optional[Sequence[str]] = None,
        seed: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        keys_to_transform : List[str]
            Keys of the entries to crop and resize. They share one crop box.
        size : Tuple[int, int]
            Output size as (height, width).
        scale : Tuple[float, float], optional
            Range of the crop area as a fraction of the input area, by default (0.08, 1.0).
        ratio : Tuple[float, float], optional
            Range of the crop aspect ratio (width / height), by default (3/4, 4/3).
        mode : Sequence[str], optional
            One interpolation mode per key, e.g. ``["bilinear", "nearest"]``.
            By default "bilinear" for all.
        seed : int, optional
            Seed for reproducible draws, by default None (torch's global RNG).
        """
        self._init_rng(seed)
        self.keys_to_transform = _as_keys(keys_to_transform)
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.mode = _resolve_modes(self.keys_to_transform, mode)

    def _draw_box(self, h: int, w: int) -> Tuple[float, float, float, float]:
        """Draw (top, left, height, width) as fractions of the input size."""
        for _ in range(10):
            area = self._uniform(*self.scale)
            log_ratio = self._uniform(np.log(self.ratio[0]), np.log(self.ratio[1]))
            aspect = float(np.exp(log_ratio))
            # fractions of the input: ch * cw == area, cw / ch (pixels) == aspect
            ch = (area * w / (h * aspect)) ** 0.5
            cw = ch * aspect * h / w
            if ch <= 1.0 and cw <= 1.0:
                return self._uniform(0, 1 - ch), self._uniform(0, 1 - cw), ch, cw
        return 0.0, 0.0, 1.0, 1.0  # fallback: whole image

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        first = data[self.keys_to_transform[0]]
        h, w = first.shape[-2:]
        top, left, ch, cw = self._draw_box(h, w)
        for key, m in zip(self.keys_to_transform, self.mode):
            x = data[key]
            xh, xw = x.shape[-2:]
            t, l = int(round(top * xh)), int(round(left * xw))
            bh, bw = max(1, int(round(ch * xh))), max(1, int(round(cw * xw)))
            patch = x[..., t : t + bh, l : l + bw]
            data[key] = _resize_spatial(
                np.ascontiguousarray(patch), self.size[0], self.size[1], _interpolation(m)
            )
        return data


class RandomVerticalPatchSwapTransform(_RandomTransform):
    """Randomly shuffle horizontal stripes, keeping listed keys aligned."""

    def __init__(
        self,
        keys_to_transform: List[str],
        patch_height: int = 50,
        prob: float = 0.5,
        seed: Optional[int] = None,
    ):
        """Initialize random vertical patch swapping.

        Parameters
        ----------
        keys_to_transform : List[str]
            Keys to shuffle together. Unlisted entries pass through unchanged.
        patch_height : int, optional
            Height of each horizontal stripe, by default 50.
        prob : float, optional
            Probability of shuffling, by default 0.5.
        seed : int, optional
            Seed for reproducible draws, by default None (torch's global RNG).
        """
        if not 0.0 <= prob <= 1.0:
            raise ValueError("prob must be between 0 and 1.")
        self._init_rng(seed)
        self.keys_to_transform = _as_keys(keys_to_transform)
        self.patch_height = patch_height
        self.prob = prob
        if patch_height <= 0:
            raise ValueError("patch_height must be greater than 0.")

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self._rand() >= self.prob or not self.keys_to_transform:
            return data
        first = data[self.keys_to_transform[0]]
        h, w = first.shape[-2:]
        permutation = torch.randperm(
            (h + self.patch_height - 1) // self.patch_height,
            generator=self._rng.generator(),
        ).numpy()
        for key in self.keys_to_transform:
            value = data[key]
            if value.shape[-2:] != (h, w):
                raise ValueError(
                    "All transformed keys must have matching spatial dimensions."
                )
            data[key] = VerticalPatchSwapTransform._apply_permutation(
                value, self.patch_height, permutation
            )
        return data
