import random
from typing import List, Optional, Tuple, Union

import numpy as np

from minerva.transforms.transform import Flip, Resize, _Transform


class EmptyTransform(_Transform):
    """A transform that does nothing to the input data."""

    def __call__(self, data):
        return data


class _RandomSyncedTransform(_Transform):
    """Orchestrate the application of a type of random transform to a list of data, ensuring that the same random state is used for all of them."""

    def __init__(self, num_samples: int, seed: Optional[int] = None):
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
        self.num_samples = num_samples
        self.transformations_executed = 0
        self.rng = np.random.default_rng(seed)
        self.transform = EmptyTransform()

    def __call__(self, data):
        if self.transformations_executed == 0:
            self.transform = self.select_transform(data)
            self.transformations_executed += 1
            return self.transform(data)
        else:
            if self.transformations_executed == self.num_samples - 1:
                self.transformations_executed = 0
            else:
                self.transformations_executed += 1
            return self.transform(data)

    def select_transform(self, data) -> _Transform:
        raise NotImplementedError(
            "This method should be implemented by the child class."
        )


class RandomFlip(_RandomSyncedTransform):

    def __init__(
        self,
        num_samples: int,
        possible_axis: Union[int, List[int]] = 0,
        seed: Optional[int] = None,
    ):
        """A transform that flips the input data along a random axis.

        Parameters
        ----------
        num_samples : int
            The number of samples that will be transformed.
        possible_axis : Union[int, List[int]], optional
            Possible axis to be transformed, will be chosen at random, by default 0
        seed : Optional[int], optional
            A seed to ensure deterministic run, by default None
        """
        super().__init__(num_samples, seed)
        self.possible_axis = possible_axis

    def select_transform(self, data):
        """selects the transform to be applied to the data."""

        if isinstance(self.possible_axis, int):
            flip_axis = self.rng.choice([True, False])
            if flip_axis:
                return Flip(axis=self.possible_axis)

        else:
            flip_axis = [
                bool(self.rng.choice([True, False]))
                for _ in range(len(self.possible_axis))
            ]
            if True in flip_axis:
                chosen_axis = [
                    axis for axis, flip in zip(self.possible_axis, flip_axis) if flip
                ]
                return Flip(axis=chosen_axis)

        return EmptyTransform()


class RandomResize(_RandomSyncedTransform):

    def __init__(
        self,
        target_scale: Tuple[int, int],
        ratio_range: Tuple[float, float],
        num_samples: int,
        seed: Optional[int] = None,
    ):
        super().__init__(num_samples, seed)
        self.target_scale = target_scale
        self.ratio_range = ratio_range
        self.resize: Optional[_Transform] = None

    def select_transform(self, data):
        orig_height, orig_width = data.shape[:2]

        # Apply a random scaling factor within the ratio range
        scale_factor = self.rng.uniform(*self.ratio_range)
        new_width = int(self.target_scale[1] * scale_factor)
        new_height = int(self.target_scale[0] * scale_factor)

        return Resize(new_width, new_height)
