from typing import List, Literal, Optional, Tuple, Union

import numpy as np

from minerva.transforms.transform import (
    Crop,
    Flip,
    GrayScale,
    Identity,
    Resize,
    Rotation,
    Solarize,
    _Transform,
)


class _RandomSyncedTransform(_Transform):
    """Orchestrate the application of a type of random transform to a list of data, ensuring that the same random state is used for all of them."""

    def __init__(self, num_samples: int = 1, seed: Optional[int] = None):
        """Orchestrate the application of a type of random transform to a list of data, ensuring that the same random state is used for all of them.

        Parameters
        ----------
        transform : _Transform
            A transform that will be applied to the input data.
        num_samples : int
            The number of samples that will be transformed.
        seed : Optional[int], optional
            The seed that will be used to generate the random state, by default None.
        """
        assert num_samples > 0, "num_samples must be greater than 0"
        self.num_samples = num_samples
        self.transformations_executed = 0
        self.rng = np.random.default_rng(seed)
        self.transform = Identity()

    def __call__(self, data):
        if self.transformations_executed == 0:
            self.transform = self.select_transform()

        self.transformations_executed = (
            self.transformations_executed + 1
        ) % self.num_samples

        return self.transform(data)

    def select_transform(self):
        raise NotImplementedError(
            "This method should be implemented by the child class."
        )


class RandomFlip(_RandomSyncedTransform):

    def __init__(
        self,
        num_samples: int = 1,
        possible_axis: Union[int, List[int]] = 1,
        prob: float = 0.5,
        seed: Optional[int] = None,
    ):
        """A transform that flips the input data along a random axis.

        Parameters
        ----------
        num_samples : int
            The number of samples that will be transformed.
        possible_axis : Union[int, List[int]], optional
            Possible axis to be transformed, will be chosen at random, by default 0
        prob : float, optional
            Probability of applying the transform, by default 0.5
        seed : Optional[int], optional
            A seed to ensure deterministic run, by default None
        """
        super().__init__(num_samples, seed)
        self.possible_axis = possible_axis
        assert 0.0 <= prob <= 1.0, "prob must be between 0 and 1"
        self.prob = prob

    def select_transform(self):
        """selects the transform to be applied to the data."""

        if isinstance(self.possible_axis, int):
            flip_axis = self.rng.choice([True, False], p=[self.prob, 1 - self.prob])
            if flip_axis:
                return Flip(axis=self.possible_axis)

        else:
            flip_axis = [
                self.rng.choice([True, False], p=[self.prob, 1 - self.prob])
                for _ in range(len(self.possible_axis))
            ]
            if True in flip_axis:
                chosen_axis = [
                    axis for axis, flip in zip(self.possible_axis, flip_axis) if flip
                ]
                return Flip(axis=chosen_axis)

        return Identity()

    def __str__(self) -> str:
        return f"RandomFlip(num_samples={self.num_samples}, possible_axis={self.possible_axis}, prob={self.prob})"


class RandomCrop(_RandomSyncedTransform):
    def __init__(
        self,
        crop_size: Tuple[int, int],
        num_samples: int = 1,
        seed: Optional[int] = None,
        pad_mode: str = "reflect",
    ):
        """
        A random cropping transform that applies the same random crop to multiple data items.

        This transform randomly selects crop coordinates and applies the same crop operation
        to multiple data items that need to be processed together, ensuring consistency across
        related data (e.g., image-mask pairs, multiple images with shared labels, etc.).
        If the crop extends beyond image boundaries, padding is applied using the specified mode.
        The random crop coordinates are generated once and reused across all data items in the group.

        Parameters
        ----------
        crop_size : Tuple[int, int]
            The desired output size for the crop as (height, width) in pixels.
        num_samples : int, default=1
            The number of times the same random crop transformation will be applied when called.
            Set to 2 for data-label pairs, 3 or more for datasets with multiple related items,
            or 1 for single data items or contrastive learning scenarios.
        seed : Optional[int], default=None
            Random seed for reproducible crop selection. If None, uses system randomness.
        pad_mode : str, default="reflect"
            Padding mode to use when crop extends beyond image boundaries. Common modes
            include "reflect", "constant", "edge", "wrap".
        """
        super().__init__(num_samples, seed)
        self.crop_size = crop_size
        self.pad_mode = pad_mode

    def select_transform(self):
        X = self.rng.random()
        Y = self.rng.random()
        return Crop(output_size=self.crop_size, pad_mode=self.pad_mode, coords=(X, Y))

    def __str__(self) -> str:
        return f"RandomCrop(num_samples={self.num_samples}, crop_size={self.crop_size}, pad_mode={self.pad_mode})"


class RandomGrayScale(_RandomSyncedTransform):

    def __init__(
        self,
        num_samples: int = 1,
        seed: Optional[int] = None,
        prob: float = 0.1,
        method: Literal["average", "luminosity"] = "luminosity",
    ):
        """
        A random grayscale conversion transform that applies the same grayscale operation to multiple data items.

        This transform randomly decides whether to convert images to grayscale based on a specified probability.
        When activated, it applies the same grayscale conversion to all related data items, ensuring consistency.
        The transform uses different methods for grayscale conversion and can be controlled by probability to
        create data augmentation effects.

        Parameters
        ----------
        num_samples : int, default=1
            The number of times the same random grayscale transformation will be applied when called.
            Set to 2 for data-label pairs, 3 or more for datasets with multiple related items,
            or 1 for single data items or contrastive learning scenarios.
        seed : Optional[int], default=None
            Random seed for reproducible grayscale selection. If None, uses system randomness.
        prob : float, default=0.1
            Probability of applying the grayscale transformation. Must be between 0.0 and 1.0.
            Higher values make grayscale conversion more likely.
        method : Literal["average", "luminosity"], default="luminosity"
            Method for grayscale conversion. "average" computes simple mean of RGB channels,
            "luminosity" uses weighted average based on human perception of brightness.
        """
        super().__init__(num_samples, seed)
        self.method = method
        self.prob = prob

    def select_transform(self):

        if self.rng.random() < self.prob:
            return GrayScale(method=self.method)

        else:
            return Identity()

    def __str__(self) -> str:
        return f"RandomGrayScale(num_samples={self.num_samples}, method={self.method}, prob={self.prob})"


class RandomSolarize(_RandomSyncedTransform):
    def __init__(
        self,
        num_samples: int = 1,
        seed: Optional[int] = None,
        threshold: int = 128,
        prob: float = 1.0,
    ):
        """
        A random solarization transform that applies the same solarize operation to multiple data items.

        This transform randomly decides whether to apply solarization based on a specified probability.
        Solarization inverts pixel values above a given threshold, creating a photographic negative effect
        for bright regions while keeping darker regions unchanged. When activated, it applies the same
        solarization parameters to all related data items, ensuring consistency.

        Parameters
        ----------
        num_samples : int, default=1
            The number of times the same random solarize transformation will be applied when called.
            Set to 2 for data-label pairs, 3 or more for datasets with multiple related items,
            or 1 for single data items or contrastive learning scenarios.
        seed : Optional[int], default=None
            Random seed for reproducible solarization selection. If None, uses system randomness.
        threshold : int, default=128
            Pixel intensity threshold for solarization. Pixels with values above this threshold
            will be inverted. Valid range is typically 0-255 for 8-bit images.
        prob : float, default=1.0
            Probability of applying the solarization transformation. Must be between 0.0 and 1.0.
            Higher values make solarization more likely.
        """

        super().__init__(num_samples, seed)
        self.threshold = threshold
        self.prob = prob

    def select_transform(self):

        if self.rng.random() < self.prob:
            return Solarize(self.threshold)

        else:
            return Identity()

    def __str__(self) -> str:
        return f"RandomSolarize(num_samples={self.num_samples}, threshold={self.threshold}, prob={self.prob})"


class RandomRotation(_RandomSyncedTransform):
    def __init__(
        self,
        degrees: float,
        prob: float,
        num_samples: int = 1,
        seed: Optional[int] = None,
    ):
        """
        A random rotation transform that applies the same rotation to multiple data items.

        This transform randomly decides whether to apply rotation based on a specified probability.
        When activated, it samples a rotation angle uniformly from the specified range and applies
        the same rotation to all related data items, ensuring consistency across image-mask pairs
        or other related data.

        Parameters
        ----------
        degrees : float
            Maximum absolute value of the rotation angle in degrees. The angle is sampled
            uniformly from [-degrees, +degrees].
        prob : float
            Probability of applying the rotation transformation. Must be between 0.0 and 1.0.
            Higher values make rotation more likely.
        num_samples : int, default=1
            The number of times the same random rotation transformation will be applied when called.
            Set to 2 for data-label pairs, 3 or more for datasets with multiple related items,
            or 1 for single data items or contrastive learning scenarios.
        seed : Optional[int], default=None
            Random seed for reproducible rotation selection and angle generation. If None, uses system randomness.
        """
        super().__init__(num_samples=num_samples, seed=seed)
        self.degrees = degrees
        self.prob = prob

    def select_transform(self):
        if self.rng.random() < self.prob:
            angle = self.rng.uniform(-self.degrees, self.degrees)
            return Rotation(degrees=angle)
        else:
            return Identity()

    def __str__(self) -> str:
        return f"RandomRotation(num_samples={self.num_samples}, degrees={self.degrees}, prob={self.prob})"


class RandomResize(_RandomSyncedTransform):

    def __init__(
        self,
        target_scale: Tuple[int, int],
        ratio_range: Tuple[float, float],
        num_samples: int,
        seed: Optional[int] = None,
    ):
        """
        A random resize transform that applies the same scale-based resize to multiple data items.

        This transform randomly samples a scaling factor within a specified range and applies
        it to a base target scale to determine the final resize dimensions. The same scaling
        factor and resulting dimensions are applied to all related data items, ensuring
        consistency across image-mask pairs or other related data. This is useful for
        multi-scale training and data augmentation.

        Parameters
        ----------
        target_scale : Tuple[int, int]
            Base target dimensions as (height, width) in pixels that will be scaled
            by the random factor.
        ratio_range : Tuple[float, float]
            Range of scaling factors as (min_ratio, max_ratio). The scaling factor
            is sampled uniformly from this range and applied to both dimensions.
        num_samples : int
            The number of times the same random resize transformation will be applied when called.
            Set to 2 for data-label pairs, 3 or more for datasets with multiple related items,
            or 1 for single data items or contrastive learning scenarios.
        seed : Optional[int], default=None
            Random seed for reproducible scaling factor selection. If None, uses system randomness.
        """
        super().__init__(num_samples, seed)
        self.target_scale = target_scale
        self.ratio_range = ratio_range
        self.resize: Optional[_Transform] = None

    def select_transform(self):

        # Apply a random scaling factor within the ratio range
        scale_factor = self.rng.uniform(*self.ratio_range)
        new_width = int(self.target_scale[1] * scale_factor)
        new_height = int(self.target_scale[0] * scale_factor)

        return Resize(new_width, new_height)

    def __str__(self) -> str:
        return f"RandomResize(num_samples={self.num_samples}, target_scale={self.target_scale}, ratio_range={self.ratio_range})"
