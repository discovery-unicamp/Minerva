"""Minerva 1.0 transforms.

Every transform takes a flat ``dict[str, Any]`` and returns a flat dict. A
transform reads and writes only the keys listed in ``keys_to_transform`` (plus
any new keys it documents as outputs); every other key passes through
unchanged. Random transforms draw their parameters once per call (with torch's
RNG, which ``DataLoader`` seeds per worker) and share them across all listed
keys. Independent randomness across views comes only from
:class:`MultiViewTransform`.

Spatial transforms use NumPy's (and PIL's/OpenCV's) layout: images are
``(H, W)`` or ``(H, W, C)``, channels last, which is what the readers return,
so no transposition is needed before transforming. Use ``TransposeTransform``
at the end of a pipeline if a model wants channels first.

A listed key that is missing from the dict raises ``KeyError``. This is not
settled in the 1.0 standard (open question 3); do not depend on it.
"""

import contextlib
import copy
import math
import os
import random
import warnings
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


# Large prime stride so (seed, worker_id, view) never collide for realistic values.
_VIEW_SALT_STRIDE = 1_000_003


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
        self.salt = 0  # set by MultiViewTransform so views never share a stream
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
            self._generator.manual_seed(
                int(self.seed) + worker_id + _VIEW_SALT_STRIDE * self.salt
            )
            self._owner = owner
        return self._generator

    def __getstate__(self):
        return {
            "seed": self.seed,
            "salt": self.salt,
            "_generator": None,
            "_owner": None,
        }


def _cv2_safe(x: np.ndarray, fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Apply an OpenCV function to ``x`` and keep ``x``'s dtype.

    OpenCV does not support int64, bool or float16 (it either raises or
    silently narrows int64 to int32, which breaks ``CrossEntropyLoss``
    targets). Those dtypes are computed in a supported dtype and cast back.
    """
    dtype = x.dtype
    if dtype == np.bool_:
        work = x.astype(np.uint8)
    elif dtype in (np.int64, np.uint64):
        work = x.astype(np.float64)
    elif dtype == np.float16:
        work = x.astype(np.float32)
    else:
        return fn(x)
    out = fn(work)
    if dtype == np.bool_:
        return out > 0
    if np.issubdtype(dtype, np.integer):
        out = np.rint(out)
    return out.astype(dtype)


def _resize_spatial(x: np.ndarray, height: int, width: int, flag: int) -> np.ndarray:
    """Resize the first two axes of an (H, W) or (H, W, C) array."""
    height, width = max(1, int(height)), max(1, int(width))

    def resize(a: np.ndarray) -> np.ndarray:
        return cv2.resize(a, (width, height), interpolation=flag)

    if x.ndim == 2:
        return _cv2_safe(x, resize)
    return np.stack([_cv2_safe(x[..., c], resize) for c in range(x.shape[-1])], axis=-1)


def _offset(slack: int, fraction: float) -> int:
    """Map a fraction in [0, 1) (or 1.0) to an integer offset in [0, slack].

    Every legal position is reachable, including the last one (``slack``).
    """
    return min(int(fraction * (slack + 1)), slack)


def _per_key(keys: Sequence[str], value: Any, name: str) -> List[Any]:
    """Expand a scalar to one value per key, or validate a per-key sequence."""
    if isinstance(value, (list, tuple)):
        values = list(value)
        if len(values) != len(keys):
            raise ValueError(f"`{name}` must be a scalar or have one entry per key.")
        return values
    return [value] * len(keys)


def _pad_to_size(
    x: np.ndarray,
    target_h: int,
    target_w: int,
    mode: str = "reflect",
    value: Union[int, float] = 0,
    centered: bool = True,
) -> np.ndarray:
    """Pad the first two axes of an (H, W[, C]) array up to the target size.

    Axes that are already large enough are left alone. ``centered=True`` splits
    the padding on both sides; otherwise it is added at the bottom/right.
    """
    h, w = x.shape[:2]
    pad_h, pad_w = max(0, target_h - h), max(0, target_w - w)
    if pad_h == 0 and pad_w == 0:
        return x
    before_h = pad_h // 2 if centered else 0
    before_w = pad_w // 2 if centered else 0
    pad_width = ((before_h, pad_h - before_h), (before_w, pad_w - before_w))
    if x.ndim > 2:
        pad_width += ((0, 0),) * (x.ndim - 2)
    kwargs = {"constant_values": value} if mode == "constant" else {}
    return np.pad(x, pad_width, mode=mode, **kwargs)  # type: ignore


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
        return int(
            torch.randint(low, high, (1,), generator=self._rng.generator()).item()
        )

    @contextlib.contextmanager
    def _seed_scope(self):
        """Make code that uses the *global* RNGs reproducible for this call.

        For transforms built on torchvision, ``random`` or ``numpy.random``,
        which cannot take a generator. With ``seed=None`` this does nothing.
        Otherwise a per-call seed is drawn from this transform's own
        generator, torch/python/numpy are seeded with it inside the block,
        and their previous states are restored afterwards.
        """
        if self.seed is None:
            yield
            return
        call_seed = self._randint(0, 2**31 - 1)
        py_state, np_state = random.getstate(), np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[]):
                # CPU generator only: torch.manual_seed would also reseed CUDA
                torch.default_generator.manual_seed(call_seed)
                random.seed(call_seed)
                np.random.seed(call_seed)
                yield
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)


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
    """Reorder the axes of every listed key.

    Works on NumPy arrays and PyTorch tensors. To exchange just two axes give
    the full permutation, e.g. ``axes=(1, 0, 2)`` swaps the first two axes of
    a 3D array. A common use is ``(2, 0, 1)`` at the end of an image pipeline
    to turn (H, W, C) into the (C, H, W) layout most models expect.
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
        self.axes = tuple(axes)

    def _transpose(self, x: Any) -> Any:
        if isinstance(x, torch.Tensor):
            return x.permute(*self.axes)
        if not isinstance(x, np.ndarray):
            raise TypeError(f"Expected a numpy array or a torch tensor, got {type(x)}.")
        return np.transpose(x, self.axes)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return self._map_keys(data, self._transpose)


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
    """Pad every listed key at the bottom/right up to the target size.

    Expects (H, W) or (H, W, C) arrays; the rank is preserved. Axes that are
    already large enough are left alone. To pad *and* crop to an exact size
    (centered padding), use :class:`CropTransform`.
    """

    def __init__(
        self,
        keys_to_transform: List[str],
        target_h_size: int,
        target_w_size: int,
        padding_mode: str = "reflect",
        padding_value: Union[int, float, Sequence[Union[int, float]]] = 0,
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
        self.padding_value = _per_key(
            self.keys_to_transform, padding_value, "padding_value"
        )

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in zip(self.keys_to_transform, self.padding_value):
            data[key] = _pad_to_size(
                data[key],
                self.target_h_size,
                self.target_w_size,
                self.padding_mode,
                value,
                centered=False,
            )
        return data


class GradientTransform(_Transform):
    """Append a 0-to-1 gradient channel to every listed key.

    Expects (H, W) or (H, W, C) arrays and returns (H, W, C + 1).
    """

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
        shape = x.shape[:2]
        gradient = np.expand_dims(self.generate_gradient(shape), axis=-1)
        x_expanded = np.expand_dims(x, axis=-1) if x.ndim == 2 else x
        output = np.concatenate([x_expanded, gradient], axis=-1)
        assert output.shape == (shape[0], shape[1], x_expanded.shape[-1] + 1)
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
    """Crop every listed key to the same region, padding if it is too small.

    Expects (H, W) or (H, W, C) arrays; the first two axes are spatial. If the
    output size is larger than the input, the input is first padded
    (centered), so this also covers "pad to a size". For a random position use
    :class:`RandomCropTransform`.
    """

    def __init__(
        self,
        keys_to_transform: List[str],
        output_size: Tuple[int, int],
        pad_mode: str = "reflect",
        coords: Tuple[float, float] = (0, 0),
        bbox: Optional[Tuple[int, int, int, int]] = None,
        pad_value: Union[int, float, Sequence[Union[int, float]]] = 0,
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
        pad_value : int | Sequence[int], optional
            Fill value when ``pad_mode="constant"``: one value for all keys or
            one per key (e.g. an ignore index for the label), by default 0.
        """
        self.keys_to_transform = _as_keys(keys_to_transform)
        self.output_size = output_size
        self.pad_mode = pad_mode
        self.coords = coords
        self.bbox = bbox
        self.pad_value = _per_key(self.keys_to_transform, pad_value, "pad_value")

    def _crop(self, image: np.ndarray, pad_value: Union[int, float] = 0) -> np.ndarray:
        new_h, new_w = self.output_size
        image = _pad_to_size(image, new_h, new_w, self.pad_mode, pad_value)
        h, w = image.shape[:2]

        if self.bbox is not None:
            y1, y2, x1, x2 = self.bbox
            return image[y1:y2, x1:x2]

        row, col = self.coords
        r = _offset(h - new_h, row)
        c = _offset(w - new_w, col)
        return image[r : r + new_h, c : c + new_w]

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in zip(self.keys_to_transform, self.pad_value):
            data[key] = self._crop(data[key], value)
        return data


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
        assert (
            image.ndim == 3 and image.shape[2] == 3
        ), "Input must have shape (H, W, 3)"
        if self.method == "average":
            gray = image.mean(axis=2)
        else:
            gray = np.dot(image[..., :3], np.array([0.299, 0.587, 0.114]))
        return np.stack([gray, gray, gray], axis=-1).astype(image.dtype)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return self._map_keys(data, self._gray)


class SolarizeTransform(_Transform):
    """Invert pixel values at or above a threshold in every listed key.

    The value range comes from the dtype: integer arrays use the dtype's
    maximum (255 for uint8), float arrays are assumed to be in [0, 1]. Pass
    ``max_value`` for anything else, e.g. ``max_value=255`` for float images
    in [0, 255]. Works on (H, W) and (H, W, C) arrays.
    """

    def __init__(
        self,
        keys_to_transform: List[str],
        threshold: Optional[float] = None,
        max_value: Optional[float] = None,
    ):
        """
        Parameters
        ----------
        keys_to_transform : List[str]
            Keys of the images to solarize.
        threshold : float, optional
            Intensity at or above which values are inverted. By default half
            of the value range (128 for uint8, 0.5 for float images in [0, 1]).
        max_value : float, optional
            Top of the value range, used to invert (``max_value - x``). By
            default 255 for uint8 (the dtype maximum for other integer
            types) and 1.0 for float images.
        """
        self.keys_to_transform = _as_keys(keys_to_transform)
        self.threshold = threshold
        self.max_value = max_value

    def _range(self, image: np.ndarray) -> Tuple[float, float]:
        if self.max_value is not None:
            max_value = self.max_value
        elif np.issubdtype(image.dtype, np.integer):
            max_value = int(np.iinfo(image.dtype).max)
        else:
            max_value = 1.0
        if self.threshold is not None:
            return self.threshold, max_value
        # 128 for uint8 (the historical default), half the range otherwise
        half = (
            128 if image.dtype == np.uint8 and self.max_value is None else max_value / 2
        )
        return half, max_value

    def _solarize(self, image: np.ndarray) -> np.ndarray:
        threshold, max_value = self._range(image)
        return np.where(image < threshold, image, max_value - image).astype(image.dtype)

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
            return _cv2_safe(
                a,
                lambda b: cv2.warpAffine(
                    b, matrix, (w, h), flags=flag, borderMode=cv2.BORDER_REFLECT
                ),
            )

        if image.ndim == 2:
            return warp(image)
        return np.stack([warp(image[:, :, c]) for c in range(image.shape[2])], axis=-1)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for key, m in zip(self.keys_to_transform, self.mode):
            data[key] = self._rotate(data[key], m)
        return data


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
    """Normalize every listed key per channel. Expects (H, W, C) data.

    Channels are the last axis. An (H, W) array counts as a single channel.
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
        if x.ndim == 2:
            x = x[..., None]
        if self.to_rgb and x.shape[-1] == 1:
            x = np.repeat(x, 3, axis=-1)
        assert x.shape[-1] == len(
            self.mean
        ), f"Number of channels in data does not match the number of provided mean/std. {x.shape}"
        for i in range(len(self.mean)):
            x[..., i] = (x[..., i] - self.mean[i]) / self.std[i]
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
        self.scale_factor = float(scale_factor)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return self._map_keys(data, lambda x: x * self.scale_factor)


class RandomAmplitudeScalingTransform(_RandomTransform):
    """Scale listed continuous signals by one random factor per call.

    This simulates sensor gain variation without changing signal timing or
    pattern. Leave categorical labels and masks unlisted.
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
            raise ValueError("scale_range must be finite, non-negative, and ordered.")
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
        return _resize_spatial(x, height, width, _interpolation(mode))

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for key, m in zip(self.keys_to_transform, self.mode):
            data[key] = self._resize(data[key], m)
        return data


class VerticalPatchSwapTransform(_RandomTransform):
    """Shuffle horizontal stripes with one shared permutation per call.

    Expects (H, W, C) or (H, W) arrays. Every listed key must have matching
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
        h = x.shape[0]
        patches = [
            x[start : min(start + patch_height, h), ...]
            for start in range(0, h, patch_height)
        ]
        return np.concatenate([patches[int(i)] for i in permutation], axis=0)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.keys_to_transform:
            return data
        first = data[self.keys_to_transform[0]]
        h, w = first.shape[:2]
        permutation = torch.randperm(
            (h + self.patch_height - 1) // self.patch_height,
            generator=self._rng.generator(),
        ).numpy()
        for key in self.keys_to_transform:
            value = data[key]
            if value.shape[:2] != (h, w):
                raise ValueError(
                    "All transformed keys must have matching spatial dimensions."
                )
            data[key] = self._apply_permutation(
                value, self.patch_height, permutation
            )
        return data


class RandomVerticalPatchSwapTransform(_RandomTransform):
    """Shuffle stripes with one shared order when a random decision succeeds."""

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
        if patch_height <= 0:
            raise ValueError("patch_height must be greater than 0.")
        if not 0.0 <= prob <= 1.0:
            raise ValueError("prob must be between 0 and 1.")
        self._init_rng(seed)
        self.keys_to_transform = _as_keys(keys_to_transform)
        self.patch_height = patch_height
        self.prob = prob

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self._rand() >= self.prob or not self.keys_to_transform:
            return data
        first = data[self.keys_to_transform[0]]
        height, width = first.shape[:2]
        permutation = torch.randperm(
            (height + self.patch_height - 1) // self.patch_height,
            generator=self._rng.generator(),
        ).numpy()
        for key in self.keys_to_transform:
            value = data[key]
            if value.shape[:2] != (height, width):
                raise ValueError(
                    "All transformed keys must have matching spatial dimensions."
                )
            data[key] = VerticalPatchSwapTransform._apply_permutation(
                value, self.patch_height, permutation
            )
        return data


class GaussianBlurTransform(_Transform):
    """Apply a Gaussian blur with a fixed sigma to every listed image key.

    Leave masks and categorical labels unlisted. For a random decision or
    random sigma, use :class:`RandomGaussianBlurTransform`.
    """

    def __init__(
        self,
        keys_to_transform: List[str],
        kernel_size: int,
        sigma: float = 0.0,
    ):
        """
        Parameters
        ----------
        keys_to_transform : List[str]
            Keys of the images to blur.
        kernel_size : int
            Odd kernel size in pixels.
        sigma : float, optional
            Fixed standard deviation. OpenCV derives it from the kernel when
            zero, by default 0.0.
        """
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if not math.isfinite(sigma) or sigma < 0:
            raise ValueError("sigma must be finite and non-negative")
        self.keys_to_transform = _as_keys(keys_to_transform)
        self.kernel_size = kernel_size
        self.sigma = sigma

    def _blur(self, x: np.ndarray) -> np.ndarray:
        k = (self.kernel_size, self.kernel_size)

        def blur(a: np.ndarray) -> np.ndarray:
            return _cv2_safe(
                a, lambda b: cv2.GaussianBlur(b, k, self.sigma)
            )

        if x.ndim == 2:
            return blur(x)
        return np.stack([blur(x[..., c]) for c in range(x.shape[-1])], axis=-1)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return self._map_keys(data, self._blur)


class RandomGaussianBlurTransform(_RandomTransform):
    """Optionally blur listed image keys with one random sigma per call.

    Leave masks and categorical labels unlisted.
    """

    def __init__(
        self,
        keys_to_transform: List[str],
        kernel_size: int,
        sigma_range: Tuple[float, float] = (0.1, 2.0),
        prob: float = 1.0,
        seed: Optional[int] = None,
    ):
        """Initialize random Gaussian blur.

        Parameters
        ----------
        keys_to_transform : List[str]
            Image keys to blur. All listed keys share one decision and sigma.
        kernel_size : int
            Positive odd kernel size in pixels.
        sigma_range : Tuple[float, float], optional
            Bounds for the uniformly sampled sigma, by default (0.1, 2.0).
        prob : float, optional
            Probability of applying blur, by default 1.0.
        seed : int, optional
            Seed for reproducible draws, by default None (torch's global RNG).
        """
        low, high = sigma_range
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if not math.isfinite(low) or not math.isfinite(high) or low < 0 or low > high:
            raise ValueError("sigma_range must be finite, non-negative, and ordered")
        if not 0.0 <= prob <= 1.0:
            raise ValueError("prob must be between 0 and 1")
        self._init_rng(seed)
        self.keys_to_transform = _as_keys(keys_to_transform)
        self.kernel_size = kernel_size
        self.sigma_range = sigma_range
        self.prob = prob

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self._rand() < self.prob:
            sigma = self._uniform(*self.sigma_range)
            GaussianBlurTransform(
                self.keys_to_transform, self.kernel_size, sigma
            )(data)
        return data


class MultiViewTransform(_Transform):
    """Create several views of the listed keys, one sub-pipeline per view.

    Each view runs its sub-pipeline on its own deep copy of the keys in
    ``keys_to_transform``. Within a view the keys share random parameters;
    across views every view draws fresh parameters, even if two views use the
    same pipeline. Output keys are named ``{view}_{key}``; the source keys are
    removed; unlisted keys stay at the top level.

    Seeded random transforms in different views are given different streams
    automatically, so two views built with the same ``seed`` still differ
    (and the result stays reproducible).
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
        # Work on private copies, so wrapping a transform never touches its
        # owner's random stream. One deepcopy keeps objects that appear in
        # several views shared (one stream that advances between views).
        try:
            self.views = copy.deepcopy(views)
            self._decorrelate_views()
        except Exception as error:  # e.g. a transform that cannot be copied
            warnings.warn(
                f"MultiViewTransform could not copy its views ({error!r}); using "
                "them as given, so same-seed transforms in different views are "
                "not decorrelated.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.views = views

    def _decorrelate_views(self) -> None:
        """Give each view's seeded random transforms a distinct stream.

        Two *separate* transform objects built with the same ``seed`` would
        otherwise draw identical parameters, so the "independent" views would
        be identical. Each seeded transform is salted with its full path
        through the (possibly nested) views; the first view keeps its plain
        seed. An object that appears in several views is left alone: it is
        one generator that advances between views, so its views differ.
        """
        found: List[Tuple[Tuple[int, ...], _RandomTransform]] = []

        def walk(node: Any, path: Tuple[int, ...]) -> None:
            if isinstance(node, _TransformsPipeline):
                for child in node.transforms:
                    walk(child, path)
            elif isinstance(node, MultiViewTransform):
                for index, child in enumerate(node.views.values()):
                    walk(child, path + (index,))
            elif isinstance(node, _RandomTransform) and node.seed is not None:
                found.append((path, node))

        for index, pipeline in enumerate(self.views.values()):
            walk(pipeline, (index,))
        paths: Dict[int, set] = {}
        for path, node in found:
            paths.setdefault(id(node), set()).add(path)
        for path, node in found:
            if len(paths[id(node)]) == 1:
                salt = 0
                for index in path:
                    salt = salt * 1000 + index
                node._rng.salt = salt
                node._rng._generator = None  # rebuild with the new salt

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


class SplitTransform(_Transform):
    """Split every listed key into equal parts along a dimension (1 key to many).

    Output keys are ``{key}_0``, ``{key}_1``, ... ``{key}_{num_splits - 1}``.
    The source keys are removed; unlisted keys pass through unchanged.
    """

    def __init__(
        self,
        keys_to_transform: List[str],
        num_splits: int = 2,
        split_dimension: int = 0,
    ):
        """
        Parameters
        ----------
        keys_to_transform : List[str]
            Keys of the entries to split.
        num_splits : int, optional
            Number of equal parts, by default 2.
        split_dimension : int, optional
            Dimension to split along, by default 0.
        """
        super().__init__()
        if num_splits <= 0:
            raise ValueError(
                f"Expected input 'num_splits' to be a positive integer greater than 0, but received {num_splits}."
            )
        if split_dimension < 0:
            raise ValueError(
                f"Expected input 'split_dimension' to be a positive integer greater than or equal to 0, but received {split_dimension}."
            )
        self.keys_to_transform = _as_keys(keys_to_transform)
        self.num_splits = num_splits
        self.split_dimension = split_dimension

    def _split(self, x: Union[np.ndarray, torch.Tensor]):
        if not isinstance(x, (np.ndarray, torch.Tensor)):
            raise TypeError(
                f"Expected input 'x' to be either a numpy array or a Pytorch tensor, but received an object of type {type(x)}."
            )
        if self.split_dimension >= len(x.shape):
            raise ValueError(
                f"Invalid split dimension: expected the split dimension to be less than {len(x.shape)}, but received {self.split_dimension}."
            )
        if x.shape[self.split_dimension] % self.num_splits != 0:
            raise ValueError(
                f"Invalid split: expected {self.num_splits} to divide equally the dimension {x.shape[self.split_dimension]}."
            )
        if isinstance(x, np.ndarray):
            return np.split(x, self.num_splits, axis=self.split_dimension)
        return torch.split(
            x,
            x.shape[self.split_dimension] // self.num_splits,
            dim=self.split_dimension,
        )

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        outputs = {}
        for key in self.keys_to_transform:
            for i, part in enumerate(self._split(data[key])):
                outputs[f"{key}_{i}"] = part
        for key in self.keys_to_transform:
            del data[key]
        data.update(outputs)
        return data


class ExternalTransform(_Transform):
    """Run an external single-input callable (e.g. a torchvision transform) on
    every listed key (inward compat wrapper).

    The callable is applied to each key separately, so a random external
    transform draws independent parameters per key. Do not use it to keep
    image and label in sync; use the keyed transforms for that.
    """

    def __init__(self, keys_to_transform: List[str], transform: Callable[[Any], Any]):
        """
        Parameters
        ----------
        keys_to_transform : List[str]
            Keys of the entries to pass through ``transform``.
        transform : Callable
            Any callable taking one object and returning one object, e.g.
            ``torchvision.transforms.Compose``.
        """
        self.keys_to_transform = _as_keys(keys_to_transform)
        self.transform = transform

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return self._map_keys(data, self.transform)


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
            Axis (or axes) that may be flipped, by default 1 (the width of an
            (H, W) or (H, W, C) array). With a list,
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


class RandomCropTransform(_RandomTransform):
    """Crop every listed key at the same random position.

    Expects (H, W) or (H, W, C) arrays. If the crop is larger than the input,
    the input is padded first (see :class:`CropTransform`). Optionally the
    position can be chosen so that no single class dominates the cropped
    label: set ``label_key`` and the position is redrawn (up to
    ``max_attempts`` times) while one class covers more than ``max_ratio`` of
    the valid pixels. The accepted position is applied to every listed key.
    """

    def __init__(
        self,
        keys_to_transform: List[str],
        crop_size: Tuple[int, int],
        pad_mode: str = "reflect",
        seed: Optional[int] = None,
        pad_value: Union[int, float, Sequence[Union[int, float]]] = 0,
        label_key: Optional[str] = None,
        max_ratio: float = 0.75,
        max_attempts: int = 10,
        ignore_index: int = -1,
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
        pad_value : int | Sequence[int], optional
            Fill value when ``pad_mode="constant"``, one for all keys or one
            per key, by default 0.
        label_key : str, optional
            Key of the (H, W) label whose class ratios decide the position
            (it need not be listed in ``keys_to_transform``). By default
            None: the position is purely random.
        max_ratio : float, optional
            With ``label_key``: the largest fraction of valid pixels one class
            may occupy in the crop, by default 0.75.
        max_attempts : int, optional
            With ``label_key``: attempts before the last position is accepted,
            by default 10.
        ignore_index : int, optional
            With ``label_key``: label value left out of the ratio computation,
            by default -1.
        """
        self._init_rng(seed)
        self.keys_to_transform = _as_keys(keys_to_transform)
        self.crop_size = crop_size
        self.pad_mode = pad_mode
        self.pad_value = _per_key(self.keys_to_transform, pad_value, "pad_value")
        self.label_key = label_key
        self.max_ratio = max_ratio
        self.max_attempts = max_attempts
        self.ignore_index = ignore_index

    def _accepts(self, coords: Tuple[float, float], label: np.ndarray) -> bool:
        if self.label_key in self.keys_to_transform:
            # same padding as the label actually gets in the output
            mode = self.pad_mode
            value = self.pad_value[self.keys_to_transform.index(self.label_key)]
        else:
            # padded pixels are not real labels: mark them as ignored
            mode, value = "constant", self.ignore_index
        cropped = CropTransform(
            ["label"], self.crop_size, mode, coords, pad_value=value
        )._crop(label.astype(np.int64), value)
        valid = cropped[cropped != self.ignore_index]
        if valid.size == 0:
            return False
        _, counts = np.unique(valid, return_counts=True)
        return bool(np.max(counts / valid.size) <= self.max_ratio)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        coords = (self._rand(), self._rand())
        if self.label_key is not None:
            label = data[self.label_key]
            for _ in range(self.max_attempts - 1):
                if self._accepts(coords, label):
                    break
                coords = (self._rand(), self._rand())
        return CropTransform(
            self.keys_to_transform,
            self.crop_size,
            self.pad_mode,
            coords,
            pad_value=self.pad_value,
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
        threshold: Optional[float] = None,
        prob: float = 1.0,
        seed: Optional[int] = None,
        max_value: Optional[float] = None,
    ):
        """
        Parameters
        ----------
        keys_to_transform : List[str]
            Keys of the images to solarize. They share one random draw.
        threshold : float, optional
            Intensity threshold for inversion, by default half the value
            range (128 for uint8, 0.5 for float images in [0, 1]).
        prob : float, optional
            Probability of solarizing, by default 1.0.
        seed : int, optional
            Seed for reproducible draws, by default None (torch's global RNG).
        max_value : float, optional
            Top of the value range; by default the dtype maximum for integer
            images and 1.0 for float images (see ``SolarizeTransform``).
        """
        self._init_rng(seed)
        self.keys_to_transform = _as_keys(keys_to_transform)
        self.threshold = threshold
        self.max_value = max_value
        self.prob = prob

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self._rand() < self.prob:
            SolarizeTransform(self.keys_to_transform, self.threshold, self.max_value)(
                data
            )
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
        return ResizeTransform(self.keys_to_transform, new_h, new_w, mode=self.mode)(
            data
        )


class RandomResizedCropTransform(_RandomTransform):
    """Crop a random area and aspect ratio, then resize, identically for every listed key.

    Expects (H, W) or (H, W, C) arrays; the first two axes are spatial.
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
        h, w = first.shape[:2]
        top, left, ch, cw = self._draw_box(h, w)
        for key, m in zip(self.keys_to_transform, self.mode):
            x = data[key]
            xh, xw = x.shape[:2]
            t, l = int(round(top * xh)), int(round(left * xw))
            bh, bw = max(1, int(round(ch * xh))), max(1, int(round(cw * xw)))
            patch = x[t : t + bh, l : l + bw]
            data[key] = _resize_spatial(
                np.ascontiguousarray(patch),
                self.size[0],
                self.size[1],
                _interpolation(m),
            )
        return data


# ---------------------------------------------------------------------------
# Deprecated transforms
#
# Kept so existing code keeps working. Each one is a special case of a
# transform above and WILL BE REMOVED IN THE NEXT MAJOR RELEASE.
# ---------------------------------------------------------------------------


def _warn_deprecated(old: str, new: str) -> None:
    warnings.warn(
        f"{old} is deprecated and will be removed in the next major release. "
        f"Use {new} instead.",
        FutureWarning,
        stacklevel=3,
    )


class PadCropTransform(CropTransform):
    """Deprecated: use :class:`CropTransform`.

    Deterministic: pads (centered) when the target is larger than the input
    and otherwise takes the centered crop, the same for every call. For a
    random position use :class:`RandomCropTransform`. Will be removed in the
    next major release.
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
        ``seed`` is accepted for backwards compatibility and ignored: this
        transform no longer draws random numbers.
        """
        _warn_deprecated(
            "PadCropTransform",
            "CropTransform(keys, (height, width), coords=(0.5, 0.5), pad_mode=..., pad_value=...)",
        )
        super().__init__(
            keys_to_transform,
            (target_h_size, target_w_size),
            pad_mode=padding_mode,
            coords=(0.5, 0.5),
            pad_value=constant_values,
        )


class ClassRatioCropTransform(RandomCropTransform):
    """Deprecated: use :class:`RandomCropTransform` with ``label_key``.

    Will be removed in the next major release.
    """

    def __init__(
        self,
        keys_to_transform: List[str],
        label_key: str,
        target_size: Tuple[int, int],
        max_ratio: float = 0.75,
        max_attempts: int = 10,
        ignore_index: int = -1,
        seed: Optional[int] = None,
    ) -> None:
        _warn_deprecated(
            "ClassRatioCropTransform",
            "RandomCropTransform(keys, size, label_key=..., max_ratio=...)",
        )
        super().__init__(
            keys_to_transform,
            tuple(target_size),
            seed=seed,
            label_key=label_key,
            max_ratio=max_ratio,
            max_attempts=max_attempts,
            ignore_index=ignore_index,
        )


class SwapAxesTransform(_Transform):
    """Deprecated: use :class:`TransposeTransform` with the full permutation.

    ``SwapAxesTransform(keys, 0, 1)`` on a 2D array equals
    ``TransposeTransform(keys, (1, 0))``. Will be removed in the next major
    release.
    """

    def __init__(self, keys_to_transform: List[str], dim0: int, dim1: int):
        _warn_deprecated("SwapAxesTransform", "TransposeTransform(keys, axes)")
        for dim in (dim0, dim1):
            if dim < 0:
                raise ValueError(
                    f"Dimension {dim} must be a positive integer greater than or equal to 0."
                )
        self.keys_to_transform = _as_keys(keys_to_transform)
        self.dim0 = dim0
        self.dim1 = dim1

    def _swap(self, x: Union[np.ndarray, torch.Tensor]):
        if not isinstance(x, (np.ndarray, torch.Tensor)):
            raise TypeError(
                f"Input type {type(x)} must be a numpy array or a Pytorch tensor."
            )
        if self.dim0 >= len(x.shape) or self.dim1 >= len(x.shape):
            raise ValueError(
                f"Dimensions {self.dim0} and {self.dim1} must be less than {len(x.shape)}."
            )
        if isinstance(x, np.ndarray):
            dims = list(range(x.ndim))
            dims[self.dim0], dims[self.dim1] = dims[self.dim1], dims[self.dim0]
            return np.transpose(x, dims)
        return x.transpose(self.dim0, self.dim1)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return self._map_keys(data, self._swap)
