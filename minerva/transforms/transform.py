"""Minerva 1.0 transforms.

Every transform takes a flat ``dict[str, Any]`` and returns a flat dict. A
transform reads and writes only the keys listed in ``keys_to_transform`` (plus
any new keys it documents as outputs); every other key passes through
unchanged. Random transforms draw their parameters once per call (with torch's
RNG, which ``DataLoader`` seeds per worker) and share them across all listed
keys. Independent randomness across views comes only from
:class:`MultiViewTransform`.

A listed key that is missing from the dict raises ``KeyError``. This is not
settled in the 1.0 standard (open question 3); do not depend on it.
"""

import copy
import math
import os
from itertools import product
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import cv2
import numpy as np
import torch
from perlin_noise import PerlinNoise

_INTERPOLATION = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "linear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "cubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
}


def _as_keys(keys_to_transform: Union[str, Sequence[str]]) -> List[str]:
    """Normalize ``keys_to_transform`` to a list of keys."""
    if isinstance(keys_to_transform, str):
        return [keys_to_transform]
    return list(keys_to_transform)


def _resolve_modes(
    keys_to_transform: Sequence[str], mode: Optional[Sequence[str]]
) -> List[str]:
    """Validate the per-key interpolation ``mode`` of a geometric transform.

    ``mode=None`` means ``"bilinear"`` for every key. Pass ``"nearest"`` for
    labels and masks, they must never be interpolated like images.
    """
    if mode is None:
        return ["bilinear"] * len(keys_to_transform)
    if isinstance(mode, str):
        mode = [mode]
    mode = list(mode)
    if len(mode) != len(keys_to_transform):
        raise ValueError(
            f"`mode` must have one entry per key in `keys_to_transform`, "
            f"got {len(mode)} modes for {len(keys_to_transform)} keys."
        )
    for m in mode:
        if m not in _INTERPOLATION:
            raise ValueError(
                f"Unknown mode {m!r}. Valid modes: {sorted(_INTERPOLATION)}."
            )
    return mode


def _interpolation(mode: str) -> int:
    return _INTERPOLATION[mode]


class _RNG:
    """Torch random source of a random transform.

    With ``seed=None`` it uses torch's global RNG, which ``DataLoader`` seeds
    differently in every worker. With an integer seed it owns a
    ``torch.Generator`` that is created lazily in the process that uses it and
    seeded with ``seed + worker_id``, so runs are reproducible while workers
    still draw different parameters. The generator is rebuilt after a fork or
    pickle, so a given worker repeats the same sequence on every run (and in
    every epoch if workers are recreated each epoch).
    """

    def __init__(self, seed: Optional[int]):
        self.seed = seed
        self._generator: Optional[torch.Generator] = None
        self._owner: Optional[Tuple[int, int]] = None

    def generator(self) -> Optional[torch.Generator]:
        if self.seed is None:
            return None
        info = torch.utils.data.get_worker_info()
        worker_id = info.id if info is not None else 0
        owner = (os.getpid(), worker_id)
        if self._generator is None or self._owner != owner:
            self._generator = torch.Generator()
            self._generator.manual_seed(int(self.seed) + worker_id)
            self._owner = owner
        return self._generator

    def __getstate__(self):
        return {"seed": self.seed, "_generator": None, "_owner": None}


def _resize_spatial(x: np.ndarray, height: int, width: int, flag: int) -> np.ndarray:
    """Resize the last two axes of a (H, W) or (C, H, W) array."""
    if x.ndim == 2:
        return cv2.resize(x, (width, height), interpolation=flag)
    return np.stack(
        [cv2.resize(c, (width, height), interpolation=flag) for c in x], axis=0
    )


class _Transform:
    """Base class for all transforms.

    A transform takes a flat dict and returns a flat dict. It reads and writes
    only the keys in ``keys_to_transform`` (plus documented output keys), keeps
    every unlisted key unchanged and holds no per-sample state between calls.
    The dict is updated in place and returned.
    """

    keys_to_transform: List[str]

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Implement the transformation logic in this method."""
        raise NotImplementedError

    def _map_keys(
        self, data: Dict[str, Any], fn: Callable[[Any], Any]
    ) -> Dict[str, Any]:
        for key in self.keys_to_transform:
            data[key] = fn(data[key])
        return data

    def __str__(self) -> str:
        args = ", ".join(
            f"{k}={v}" for k, v in vars(self).items() if not k.startswith("_")
        )
        return f"{type(self).__name__}({args})"


class _RandomTransform(_Transform):
    """Base class of transforms that draw random parameters.

    Subclasses call ``self._init_rng(seed)`` in their constructor and draw
    with ``self._rand()``, ``self._uniform()`` and ``self._randint()`` once per
    call. ``seed=None`` (default) uses torch's global RNG, so
    ``torch.manual_seed`` also gives reproducible runs. An integer ``seed``
    makes this transform reproducible on its own (see :class:`_RNG`).
    """

    seed: Optional[int]

    def _init_rng(self, seed: Optional[int]) -> None:
        self.seed = seed
        self._rng = _RNG(seed)

    def _rand(self) -> float:
        """Uniform float in [0, 1)."""
        return torch.rand(1, generator=self._rng.generator()).item()

    def _uniform(self, low: float, high: float) -> float:
        return low + (high - low) * self._rand()

    def _randint(self, low: int, high: int) -> int:
        """Integer in [low, high)."""
        return int(torch.randint(low, high, (1,), generator=self._rng.generator()).item())


class _TransformsPipeline(_Transform):
    """Apply a sequence of transforms (dict in, dict out) in order.

    A pipeline is itself a transform, so it can be nested or used as a view
    sub-pipeline in :class:`MultiViewTransform`.
    """

    def __init__(self, transforms: Sequence[_Transform]):
        """
        Parameters
        ----------
        transforms : Sequence[_Transform]
            The transforms to apply, in order.
        """
        self.transforms = transforms

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for transform in self.transforms:
            data = transform(data)
        return data

    def __add__(self, other: _Transform) -> "_TransformsPipeline":
        if isinstance(other, _TransformsPipeline):
            return _TransformsPipeline(list(self.transforms) + list(other.transforms))
        return _TransformsPipeline(list(self.transforms) + [other])

    def __radd__(self, other: _Transform) -> "_TransformsPipeline":
        if isinstance(other, _TransformsPipeline):
            return _TransformsPipeline(list(other.transforms) + list(self.transforms))
        return _TransformsPipeline([other] + list(self.transforms))

    def __str__(self) -> str:
        return f"_TransformsPipeline(transforms=[{', '.join(str(t) for t in self.transforms)}])"


class FlipTransform(_Transform):
    """Flip every listed key along the given axis."""

    def __init__(self, keys_to_transform: List[str], axis: Union[int, List[int]] = 1):
        """
        Parameters
        ----------
        keys_to_transform : List[str]
            Keys of the entries to flip.
        axis : int | List[int], optional
            One or more axes to flip along, by default 1. If a list is given,
            the data is flipped along all of them.
        """
        self.keys_to_transform = _as_keys(keys_to_transform)
        self.axis = axis

    def _flip(self, x: np.ndarray) -> np.ndarray:
        if isinstance(self.axis, list):
            assert len(self.axis) <= x.ndim, (
                "Axis list has more dimensions than input data. The length of "
                "axis needs to be less or equal to input dimensions."
            )
        return np.flip(x, axis=self.axis).copy()

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return self._map_keys(data, self._flip)


class PerlinMaskerTransform(_RandomTransform):
    """Zero entries of every listed key according to the sign of Perlin noise.

    The noise seed is drawn once per call and shared by all listed keys.
    """

    def __init__(
        self,
        keys_to_transform: List[str],
        octaves: int,
        scale: float = 1,
        seed: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        keys_to_transform : List[str]
            Keys of the entries to mask.
        octaves : int
            Level of detail for the Perlin noise generator.
        scale : float, optional
            Rescale the Perlin noise, by default 1 (no rescaling).
        seed : int, optional
            Seed for reproducible noise, by default None (torch's global RNG).
        """
        if octaves <= 0:
            raise ValueError(f"Number of octaves must be positive, but got {octaves=}")
        if scale == 0:
            raise ValueError("Scale can't be 0")
        self.keys_to_transform = _as_keys(keys_to_transform)
        self.octaves = octaves
        self.scale = scale
        self._init_rng(seed)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        seed = self._randint(0, 2**32)
        for key in self.keys_to_transform:
            x = data[key]
            mask = np.empty_like(x, dtype=bool)
            noise = PerlinNoise(self.octaves, seed)
            denom = self.scale * max(x.shape)
            for pos in product(*[range(i) for i in mask.shape]):
                mask[pos] = noise([i / denom for i in pos]) < 0
            data[key] = x * mask
        return data


class SqueezeTransform(_Transform):
    """Remove a single-dimensional axis from every listed key."""

    def __init__(self, keys_to_transform: List[str], axis: int):
        """
        Parameters
        ----------
        keys_to_transform : List[str]
            Keys of the entries to squeeze.
        axis : int
            The position of the axis to be removed.
        """
        self.keys_to_transform = _as_keys(keys_to_transform)
        self.axis = axis

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return self._map_keys(data, lambda x: np.squeeze(x, axis=self.axis))


class UnsqueezeTransform(_Transform):
    """Add a new axis to every listed key.

    To unsqueeze only the label, list only the label key.
    """

    def __init__(self, keys_to_transform: List[str], axis: int):
        """
        Parameters
        ----------
        keys_to_transform : List[str]
            Keys of the entries to unsqueeze.
        axis : int
            The position of the new axis.
        """
        self.keys_to_transform = _as_keys(keys_to_transform)
        self.axis = axis

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return self._map_keys(data, lambda x: np.expand_dims(x, axis=self.axis))


class TransposeTransform(_Transform):
    """Reorder the axes of every listed key (``np.transpose``).

    To exchange just two axes use :class:`SwapAxesTransform`.
    """

    def __init__(self, keys_to_transform: List[str], axes: Sequence[int]):
        """
        Parameters
        ----------
        keys_to_transform : List[str]
            Keys of the entries to transpose.
        axes : Sequence[int]
            The order of the new axes.
        """
        self.keys_to_transform = _as_keys(keys_to_transform)
        self.axes = axes

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return self._map_keys(data, lambda x: np.transpose(x, self.axes))


class CastToTransform(_Transform):
    """Cast every listed key to the given data type."""

    def __init__(self, keys_to_transform: List[str], dtype: Union[type, str]):
        """
        Parameters
        ----------
        keys_to_transform : List[str]
            Keys of the entries to cast.
        dtype : type | str
            The data type to cast to.
        """
        self.keys_to_transform = _as_keys(keys_to_transform)
        self.dtype = dtype

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return self._map_keys(data, lambda x: x.astype(self.dtype))


class PaddingTransform(_Transform):
    """Pad every listed key to reach the target height and width.

    Expects (H, W) or (H, W, C) arrays. A 2D input becomes (H, W, 1).
    """

    def __init__(
        self,
        keys_to_transform: List[str],
        target_h_size: int,
        target_w_size: int,
        padding_mode: str = "reflect",
        padding_value: Union[int, Sequence[int]] = 0,
    ):
        """
        Parameters
        ----------
        keys_to_transform : List[str]
            Keys of the entries to pad.
        target_h_size : int
            Target height in pixels after padding.
        target_w_size : int
            Target width in pixels after padding.
        padding_mode : str, optional
            Padding mode ("reflect", "constant", "edge", "wrap"), by default "reflect".
        padding_value : int | Sequence[int], optional
            Value used when ``padding_mode="constant"``. Either one value for
            all keys or one value per key (e.g. 0 for the image and 255 for
            an ignore-index label), by default 0.
        """
        self.keys_to_transform = _as_keys(keys_to_transform)
        self.target_h_size = target_h_size
        self.target_w_size = target_w_size
        self.padding_mode = padding_mode
        if isinstance(padding_value, (int, float, np.integer, np.floating)):
            values = [padding_value] * len(self.keys_to_transform)
        else:
            values = list(padding_value)
            if len(values) != len(self.keys_to_transform):
                raise ValueError(
                    "`padding_value` must be a scalar or have one entry per key."
                )
        self.padding_value = values

    def _pad(self, x: np.ndarray, value: Union[int, float]) -> np.ndarray:
        h, w = x.shape[:2]
        pad_h = max(0, self.target_h_size - h)
        pad_w = max(0, self.target_w_size - w)
        pad_width = ((0, pad_h), (0, pad_w)) + (((0, 0),) if x.ndim > 2 else ())
        kwargs = {"constant_values": value} if self.padding_mode == "constant" else {}
        padded = np.pad(x, pad_width, mode=self.padding_mode, **kwargs)  # type: ignore
        if x.ndim == 2:
            padded = np.expand_dims(padded, axis=2)
        return padded

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in zip(self.keys_to_transform, self.padding_value):
            data[key] = self._pad(data[key], value)
        return data


class GradientTransform(_Transform):
    """Append a 0-to-1 gradient channel to every listed key."""

    directions = {0: "x (width)", 1: "y (height)"}

    def __init__(self, keys_to_transform: List[str], direction: int):
        """
        Parameters
        ----------
        keys_to_transform : List[str]
            Keys of the entries to which the gradient channel is appended.
        direction : int
            0 for a gradient along x (width), 1 along y (height).
        """
        assert direction in [0, 1], "Direction must be 0 (x-axis) or 1 (y-axis)"
        self.keys_to_transform = _as_keys(keys_to_transform)
        self.direction = direction

    def generate_gradient(self, shape: Tuple[int, int]) -> np.ndarray:
        """Gradient from 0 to 1 for a (H, W) shape along ``direction``."""
        xx, yy = np.meshgrid(np.linspace(0, 1, shape[1]), np.linspace(0, 1, shape[0]))
        return xx if self.direction == 0 else yy

    def _gradient(self, x: np.ndarray) -> np.ndarray:
        shape = x.shape if x.ndim == 2 else x.shape[1:]
        gradient = np.expand_dims(self.generate_gradient(shape), axis=0)
        x_expanded = np.expand_dims(x, axis=0) if x.ndim == 2 else x
        output = np.concatenate([x_expanded, gradient], axis=0)
        assert output.shape == (x_expanded.shape[0] + 1, shape[0], shape[1])
        return output

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return self._map_keys(data, self._gradient)


class ColorJitterTransform(_Transform):
    """Apply fixed brightness, contrast, saturation and hue adjustments.

    Expects uint8 RGB images with shape (H, W, 3). The adjustments are
    deterministic.
    """

    def __init__(
        self,
        keys_to_transform: List[str],
        brightness: float = 1.0,
        contrast: float = 1.0,
        saturation: float = 1.0,
        hue: float = 0.0,
    ):
        """
        Parameters
        ----------
        keys_to_transform : List[str]
            Keys of the images to adjust.
        brightness : float, optional
            Brightness factor, 1.0 means no change.
        contrast : float, optional
            Contrast factor, 1.0 means no change.
        saturation : float, optional
            Saturation factor, 1.0 means no change.
        hue : float, optional
            Hue shift in the OpenCV HSV scale, by default 0.0.
        """
        self.keys_to_transform = _as_keys(keys_to_transform)
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def _jitter(self, image: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        image[..., 2] = np.clip(image[..., 2] * self.brightness, 0, 255)
        image[..., 1] = np.clip(image[..., 1] * self.saturation, 0, 255)
        mean = image[..., 2].mean()
        image[..., 2] = np.clip((image[..., 2] - mean) * self.contrast + mean, 0, 255)
        image[..., 0] = (image[..., 0] + self.hue) % 180
        return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_HSV2RGB)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return self._map_keys(data, self._jitter)


class CropTransform(_Transform):
    """Crop every listed key to the same region.

    Expects (C, H, W) or (H, W) arrays; the last two axes are spatial.
    """

    def __init__(
        self,
        keys_to_transform: List[str],
        output_size: Tuple[int, int],
        pad_mode: str = "reflect",
        coords: Tuple[float, float] = (0, 0),
        bbox: Optional[Tuple[int, int, int, int]] = None,
    ):
        """
        Parameters
        ----------
        keys_to_transform : List[str]
            Keys of the entries to crop. Unlisted keys pass through unchanged.
        output_size : Tuple[int, int]
            Desired output size as (height, width).
        pad_mode : str, optional
            Padding mode used if the output size is larger than the input,
            by default "reflect". Any ``np.pad`` mode.
        coords : Tuple[float, float], optional
            Relative (row, col) position of the crop's top-left corner, each
            from 0 to 1, by default (0, 0).
        bbox : Tuple[int, int, int, int], optional
            If given, crop to (y1, y2, x1, x2) and ignore ``coords``.
        """
        self.keys_to_transform = _as_keys(keys_to_transform)
        self.output_size = output_size
        self.pad_mode = pad_mode
        self.coords = coords
        self.bbox = bbox

    def _crop(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[-2:]
        new_h, new_w = self.output_size

        if new_h > h or new_w > w:
            pad_h = max(new_h - h, 0)
            pad_w = max(new_w - w, 0)
            pad_width = ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2))
            if image.ndim == 3:
                pad_width = ((0, 0),) + pad_width
            image = np.pad(image, pad_width, mode=self.pad_mode)  # type: ignore
            h, w = image.shape[-2:]

        if self.bbox is not None:
            y1, y2, x1, x2 = self.bbox
            return image[..., y1:y2, x1:x2]

        row, col = self.coords
        r = int((h - new_h) * row)
        c = int((w - new_w) * col)
        return image[..., r : r + new_h, c : c + new_w]

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return self._map_keys(data, self._crop)


class GrayScaleTransform(_Transform):
    """Convert every listed RGB image (H, W, 3) to a 3-channel grayscale."""

    def __init__(
        self,
        keys_to_transform: List[str],
        method: Literal["average", "luminosity"] = "luminosity",
    ):
        """
        Parameters
        ----------
        keys_to_transform : List[str]
            Keys of the images to convert.
        method : {'average', 'luminosity'}, optional
            'average' is (R + G + B) / 3; 'luminosity' is
            0.299R + 0.587G + 0.114B. By default 'luminosity'.
        """
        if method not in {"average", "luminosity"}:
            raise ValueError("method must be 'average' or 'luminosity'")
        self.keys_to_transform = _as_keys(keys_to_transform)
        self.method = method

    def _gray(self, image: np.ndarray) -> np.ndarray:
        assert image.ndim == 3 and image.shape[2] == 3, "Input must have shape (H, W, 3)"
        if self.method == "average":
            gray = image.mean(axis=2)
        else:
            gray = np.dot(image[..., :3], np.array([0.299, 0.587, 0.114]))
        return np.stack([gray, gray, gray], axis=-1).astype(image.dtype)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return self._map_keys(data, self._gray)


class SolarizeTransform(_Transform):
    """Invert pixel values at or above a threshold in every listed key."""

    def __init__(self, keys_to_transform: List[str], threshold: int = 128):
        """
        Parameters
        ----------
        keys_to_transform : List[str]
            Keys of the images to solarize.
        threshold : int, optional
            Intensity threshold for inversion, by default 128.
        """
        self.keys_to_transform = _as_keys(keys_to_transform)
        self.threshold = threshold

    def _solarize(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            channels = [
                np.where(c < self.threshold, c, 255 - c) for c in cv2.split(image)
            ]
            return cv2.merge(channels)
        return np.where(image < self.threshold, image, 255 - image)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return self._map_keys(data, self._solarize)


class RotationTransform(_Transform):
    """Rotate every listed key by a fixed angle with reflection padding.

    Accepts (H, W) or (H, W, C) arrays.
    """

    def __init__(
        self,
        keys_to_transform: List[str],
        degrees: float,
        mode: Optional[Sequence[str]] = None,
    ):
        """
        Parameters
        ----------
        keys_to_transform : List[str]
            Keys of the entries to rotate.
        degrees : float
            Counterclockwise angle in degrees.
        mode : Sequence[str], optional
            One interpolation mode per key ("nearest", "bilinear", "bicubic",
            "area"). Use "nearest" for labels. By default "bilinear" for all.
        """
        self.keys_to_transform = _as_keys(keys_to_transform)
        self.degrees = degrees
        self.mode = _resolve_modes(self.keys_to_transform, mode)

    def _rotate(self, image: np.ndarray, flag_mode: str) -> np.ndarray:
        if image.ndim not in (2, 3):
            raise ValueError(
                f"Unsupported image shape: {image.shape}. Expected 2D or 3D with channels last."
            )
        flag = _interpolation(flag_mode)
        h, w = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((w // 2, h // 2), self.degrees, 1.0)

        def warp(a):
            return cv2.warpAffine(
                a, matrix, (w, h), flags=flag, borderMode=cv2.BORDER_REFLECT
            )

        if image.ndim == 2:
            return warp(image)
        return np.stack([warp(image[:, :, c]) for c in range(image.shape[2])], axis=-1)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for key, m in zip(self.keys_to_transform, self.mode):
            data[key] = self._rotate(data[key], m)
        return data


class PadCropTransform(_RandomTransform):
    """Pad or crop every listed key to the target size.

    If the target is larger than the input along an axis, that axis is padded
    symmetrically (the input is centered); otherwise it is cropped from a
    random position. The crop position is drawn once per call and shared by
    all listed keys (as relative offsets, so keys of different sizes stay
    aligned). Expects (C, H, W) or (H, W) arrays.
    """

    def __init__(
        self,
        keys_to_transform: List[str],
        target_h_size: int,
        target_w_size: int,
        padding_mode: str = "reflect",
        constant_values: int = 0,
        seed: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        keys_to_transform : List[str]
            Keys of the entries to pad or crop.
        target_h_size : int
            Desired height.
        target_w_size : int
            Desired width.
        padding_mode : str, optional
            Padding mode, by default "reflect".
        constant_values : int, optional
            Padding value when ``padding_mode="constant"``, by default 0.
        seed : int, optional
            Seed for a reproducible crop position, by default None.
        """
        self._init_rng(seed)
        self.keys_to_transform = _as_keys(keys_to_transform)
        self.target_h_size = target_h_size
        self.target_w_size = target_w_size
        self.padding_mode = padding_mode
        self.constant_values = constant_values

    def _pad_crop(self, x: np.ndarray, frac_h: float, frac_w: float) -> np.ndarray:
        if x.ndim == 3:
            x = np.transpose(x, (1, 2, 0))
        h, w = x.shape[:2]

        def pad(arr, axis, amount):
            before = amount // 2
            width = [(0, 0)] * arr.ndim
            width[axis] = (before, amount - before)
            kwargs = (
                {"constant_values": self.constant_values}
                if self.padding_mode == "constant"
                else {}
            )
            return np.pad(arr, width, mode=self.padding_mode, **kwargs)  # type: ignore

        if self.target_h_size > h:
            x = pad(x, 0, self.target_h_size - h)
        elif self.target_h_size < h:
            n = h - self.target_h_size
            start = min(int(frac_h * (n + 1)), n)
            x = x[start : start + self.target_h_size]

        if self.target_w_size > w:
            x = pad(x, 1, self.target_w_size - w)
        elif self.target_w_size < w:
            n = w - self.target_w_size
            start = min(int(frac_w * (n + 1)), n)
            x = x[:, start : start + self.target_w_size]

        if x.ndim == 3:
            x = np.transpose(x, (2, 0, 1))
        return x

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        frac_h, frac_w = self._rand(), self._rand()
        return self._map_keys(data, lambda x: self._pad_crop(x, frac_h, frac_w))


class IdentityTransform(_Transform):
    """Do nothing. Useful to skip a transform in a pipeline."""

    def __init__(self, keys_to_transform: Optional[List[str]] = None):
        """
        Parameters
        ----------
        keys_to_transform : List[str], optional
            Ignored; accepted for a uniform constructor.
        """
        self.keys_to_transform = _as_keys(keys_to_transform or [])

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return data


class IndexerTransform(_Transform):
    """Extract a single channel (``x[index]``) from every listed key."""

    def __init__(self, keys_to_transform: List[str], index: int):
        """
        Parameters
        ----------
        keys_to_transform : List[str]
            Keys of the entries to index.
        index : int
            The index of the channel to extract.
        """
        self.keys_to_transform = _as_keys(keys_to_transform)
        self.index = index

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return self._map_keys(data, lambda x: x[self.index])


class RepeatTransform(_Transform):
    """Repeat every listed key along an axis."""

    def __init__(self, keys_to_transform: List[str], axis: int, n_repetitions: int):
        """
        Parameters
        ----------
        keys_to_transform : List[str]
            Keys of the entries to repeat.
        axis : int
            The axis along which to repeat.
        n_repetitions : int
            The number of repetitions.
        """
        self.keys_to_transform = _as_keys(keys_to_transform)
        self.axis = axis
        self.n_repetitions = n_repetitions

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return self._map_keys(
            data, lambda x: np.repeat(x, self.n_repetitions, axis=self.axis)
        )


class NormalizeTransform(_Transform):
    """Normalize every listed key per channel. Expects (C, H, W) data.

    List only the keys to normalize; labels are normalized only if listed.
    """

    def __init__(
        self,
        keys_to_transform: List[str],
        mean: Sequence[float],
        std: Sequence[float],
        to_rgb: bool = False,
    ):
        """
        Parameters
        ----------
        keys_to_transform : List[str]
            Keys of the entries to normalize.
        mean : Sequence[float]
            Mean of each channel.
        std : Sequence[float]
            Standard deviation of each channel.
        to_rgb : bool, optional
            Convert single-channel data to 3-channel RGB, by default False.
        """
        assert len(mean) == len(
            std
        ), "Means and standard deviations must have the same length."
        self.keys_to_transform = _as_keys(keys_to_transform)
        self.mean = mean
        self.std = std
        self.to_rgb = to_rgb

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        if not np.issubdtype(x.dtype, np.floating):
            x = x.astype(np.float32)
        else:
            x = x.copy()
        if self.to_rgb and x.shape[0] == 1:
            x = np.repeat(x, 3, axis=0)
        assert x.shape[0] == len(
            self.mean
        ), f"Number of channels in data does not match the number of provided mean/std. {x.shape}"
        for i in range(len(self.mean)):
            x[i, :, :] = (x[i, :, :] - self.mean[i]) / self.std[i]
        return x

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return self._map_keys(data, self._normalize)


class AmplitudeScalingTransform(_Transform):
    """Multiply each listed signal by a fixed amplitude factor.

    This supports deterministic gain adjustment while preserving the signal's
    temporal pattern.

    List only continuous signal/data keys. Leave categorical labels and masks
    unlisted because multiplying their class IDs changes their meaning.
    """

    def __init__(self, keys_to_transform: List[str], scale_factor: float):
        """Initialize fixed amplitude scaling.

        Parameters
        ----------
        keys_to_transform : List[str]
            Signal keys to scale. Unlisted entries, including masks, pass through.
        scale_factor : float
            Scalar multiplier applied to every listed signal.
        """
        if not math.isfinite(scale_factor) or scale_factor < 0:
            raise ValueError("scale_factor must be finite and non-negative.")
        self.keys_to_transform = _as_keys(keys_to_transform)
        self.scale_factor = scale_factor

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return self._map_keys(data, lambda x: x * self.scale_factor)


class ReshapeTransform(_Transform):
    """Reshape every listed key."""

    def __init__(self, keys_to_transform: List[str], shape: Tuple[int, ...]):
        """
        Parameters
        ----------
        keys_to_transform : List[str]
            Keys of the entries to reshape.
        shape : Tuple[int, ...]
            The target shape.
        """
        self.keys_to_transform = _as_keys(keys_to_transform)
        self.shape = shape

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return self._map_keys(data, lambda x: x.reshape(self.shape))


class ResizeTransform(_Transform):
    """Resize every listed key to the target size. Expects (H, W[, C]) arrays."""

    def __init__(
        self,
        keys_to_transform: List[str],
        target_h_size: int,
        target_w_size: int,
        keep_aspect_ratio: bool = False,
        mode: Optional[Sequence[str]] = None,
    ):
        """
        Parameters
        ----------
        keys_to_transform : List[str]
            Keys of the entries to resize.
        target_h_size : int
            Target height in pixels.
        target_w_size : int
            Target width in pixels.
        keep_aspect_ratio : bool, optional
            If True, scale by the smaller scaling factor so the result fits
            within the target without distortion, by default False.
        mode : Sequence[str], optional
            One interpolation mode per key ("nearest", "bilinear", "bicubic",
            "area"). Use "nearest" for labels and masks. By default
            "bilinear" for all keys.
        """
        self.keys_to_transform = _as_keys(keys_to_transform)
        self.target_h_size = target_h_size
        self.target_w_size = target_w_size
        self.keep_aspect_ratio = keep_aspect_ratio
        self.mode = _resolve_modes(self.keys_to_transform, mode)

    def _resize(self, x: np.ndarray, mode: str) -> np.ndarray:
        original_height, original_width = x.shape[:2]
        width, height = self.target_w_size, self.target_h_size
        if self.keep_aspect_ratio:
            scale = min(width / original_width, height / original_height)
            # Round down to match MMsegmentation's behavior.
            width = int(original_width * scale)
            height = int(original_height * scale)
        return cv2.resize(x, (width, height), interpolation=_interpolation(mode))

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for key, m in zip(self.keys_to_transform, self.mode):
            data[key] = self._resize(data[key], m)
        return data


class VerticalPatchSwapTransform(_RandomTransform):
    """Shuffle horizontal stripes with one shared permutation per call.

    Expects (C, H, W) or (H, W) arrays. Every listed key must have matching
    spatial dimensions so the same stripe permutation preserves alignment.
    """

    def __init__(
        self,
        keys_to_transform: List[str],
        patch_height: int = 50,
        seed: Optional[int] = None,
    ):
        """Initialize the vertical patch shuffle.

        Parameters
        ----------
        keys_to_transform : List[str]
            Keys to shuffle together. Unlisted entries pass through unchanged.
        patch_height : int, optional
            Height of each horizontal stripe, by default 50.
        seed : int, optional
            Seed for reproducible draws, by default None (torch's global RNG).
        """
        if patch_height <= 0:
            raise ValueError("patch_height must be greater than 0.")
        self._init_rng(seed)
        self.keys_to_transform = _as_keys(keys_to_transform)
        self.patch_height = patch_height

    @staticmethod
    def _apply_permutation(
        x: np.ndarray, patch_height: int, permutation: np.ndarray
    ) -> np.ndarray:
        h = x.shape[-2]
        patches = [
            x[..., start : min(start + patch_height, h), :]
            for start in range(0, h, patch_height)
        ]
        return np.concatenate([patches[int(i)] for i in permutation], axis=-2)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.keys_to_transform:
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
            data[key] = self._apply_permutation(
                value, self.patch_height, permutation
            )
        return data


class GaussianBlurTransform(_RandomTransform):
    """Gaussian-blur every listed key with one random sigma per call.

    Expects (C, H, W) or (H, W) arrays. The decision to blur and the sigma
    are drawn once per call and shared by all listed keys.
    """

    def __init__(
        self,
        keys_to_transform: List[str],
        kernel_size: int,
        sigma: Tuple[float, float] = (0.1, 2.0),
        prob: float = 1.0,
        seed: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        keys_to_transform : List[str]
            Keys of the images to blur.
        kernel_size : int
            Odd kernel size in pixels.
        sigma : Tuple[float, float], optional
            Range the standard deviation is drawn uniformly from, by default (0.1, 2.0).
        prob : float, optional
            Probability of blurring, by default 1.0.
        seed : int, optional
            Seed for reproducible blurs, by default None.
        """
        self._init_rng(seed)
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if not 0.0 <= sigma[0] <= sigma[1]:
            raise ValueError("sigma must be a non-negative increasing range")
        if not 0.0 <= prob <= 1.0:
            raise ValueError("prob must be between 0 and 1")
        self.keys_to_transform = _as_keys(keys_to_transform)
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.prob = prob

    def _blur(self, x: np.ndarray, sigma: float) -> np.ndarray:
        k = (self.kernel_size, self.kernel_size)
        if x.ndim == 2:
            return cv2.GaussianBlur(x, k, sigma)
        return np.stack([cv2.GaussianBlur(c, k, sigma) for c in x], axis=0)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        do_blur = self._rand() < self.prob
        sigma = self._uniform(*self.sigma)
        if do_blur:
            self._map_keys(data, lambda x: self._blur(x, sigma))
        return data


class MultiViewTransform(_Transform):
    """Create several views of the listed keys, one sub-pipeline per view.

    Each view runs its sub-pipeline on its own deep copy of the keys in
    ``keys_to_transform``. Within a view the keys share random parameters;
    across views every view draws fresh parameters, even if two views use the
    same pipeline. Output keys are named ``{view}_{key}``; the source keys are
    removed; unlisted keys stay at the top level.
    """

    def __init__(
        self,
        keys_to_transform: List[str],
        views: Dict[str, _Transform],
    ):
        """
        Parameters
        ----------
        keys_to_transform : List[str]
            Keys copied into each view.
        views : Dict[str, _Transform]
            View name to the (sub-)pipeline that builds it. Different
            pipelines per view give asymmetric augmentation.
        """
        self.keys_to_transform = _as_keys(keys_to_transform)
        self.views = views

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        outputs: Dict[str, Any] = {}
        for view, pipeline in self.views.items():
            sub = {k: copy.deepcopy(data[k]) for k in self.keys_to_transform}
            sub = pipeline(sub)
            for key, value in sub.items():
                outputs[f"{view}_{key}"] = value
        for key in self.keys_to_transform:
            del data[key]
        data.update(outputs)
        return data
