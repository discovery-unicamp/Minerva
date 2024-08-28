import random
from typing import List, Optional, Union

import numpy as np

from minerva.transforms.transform import Flip, _Transform


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
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def __call__(self, data):
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
        super().__init__(num_samples, seed)
        self.possible_axis = possible_axis
        self.flip: Optional[_Transform] = None

    def __call__(self, data):
        if self.transformations_executed == 0:
            self.transformations_executed += 1
            if isinstance(self.possible_axis, int):
                flip_axis = bool(random.getrandbits(1))
                if flip_axis:
                    self.flip = Flip(axis=self.possible_axis)
                    return self.flip(data)
                else:
                    self.flip = EmptyTransform()
                    return self.flip(data)
            else:
                flip_axis = [
                    bool(random.getrandbits(1)) for _ in range(len(self.possible_axis))
                ]
                if True in flip_axis:
                    chosen_axis = [
                        axis
                        for axis, flip in zip(self.possible_axis, flip_axis)
                        if flip
                    ]
                    self.flip = Flip(axis=chosen_axis)
                    return self.flip(data)
                else:
                    self.flip = EmptyTransform()
                    return self.flip(data)

        else:
            if self.transformations_executed == self.num_samples - 1:
                self.transformations_executed = 0
            else:
                self.transformations_executed += 1
            return self.flip(data)
