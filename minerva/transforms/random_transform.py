from typing import List, Optional, Union, Tuple

import numpy as np

from minerva.transforms.transform import Flip, _Transform, Crop, GrayScale, Solarize, Rotation


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
            self.transform = self.select_transform()
            self.transformations_executed += 1
            return self.transform(data)
        else:
            if self.transformations_executed == self.num_samples - 1:
                self.transformations_executed = 0
            else:
                self.transformations_executed += 1
            return self.transform(data)

    def select_transform(self):
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

    def select_transform(self):
        """selects the transform to be applied to the data."""

        if isinstance(self.possible_axis, int):
            flip_axis = self.rng.choice([True, False])
            print(flip_axis)
            if flip_axis:
                return Flip(axis=self.possible_axis)

        else:
            flip_axis = [
                bool(self.rng.choice([True, False]))
                for _ in range(len(self.possible_axis))
            ]
            print(flip_axis)
            if True in flip_axis:
                chosen_axis = [
                    axis for axis, flip in zip(self.possible_axis, flip_axis) if flip
                ]
                return Flip(axis=chosen_axis)

        return EmptyTransform() 
    

# # TODO
# class RandomCrop(_RandomSyncedTransform):
#     def __init__(
#         self,
#         num_samples: int,
#         crop_size: Tuple[int, int],
#         seed: Optional[int] = None,
#         pad_mode: str = 'reflect'
#     ):
#         super().__init__(num_samples, seed)
#         self.crop_size = crop_size
#         self.pad_mode = pad_mode

#     def select_transform(self):
        
#         # h, w = image.shape[:2]
#         crop_h, crop_w = self.crop_size

#         # Select random top-left coordinates for the crop
#         max_x = h - crop_h
#         max_y = w - crop_w
#         x = self.rng.integers(0, max_x + 1)
#         y = self.rng.integers(0, max_y + 1)

#         return Crop(output_size=self.crop_size, pad_mode=self.pad_mode, coords=(x, y))
    
    
class RandomGrayScale(_RandomSyncedTransform):
    def __init__(
        self,
        num_samples: int,
        seed: Optional[int] = None,
        prob: float = 0.1,
        gray: float = 1.0,
        ):  
        
        super().__init__(num_samples, seed)
        self.gray = gray
        self.prob = prob
        
    def select_transform(self):
        
        if self.rng.random() < self.prob:
            return GrayScale(gray=self.gray)

        else:
            return EmptyTransform()
        

class RandomSolarize(_RandomSyncedTransform):
    def __init__(
        self,
        num_samples: int,
        seed: Optional[int] = None,
        threshold: int = 128,
        prob: float = 1.0,
    ):
        
        super().__init__(num_samples, seed)
        self.threshold = threshold
        self.prob = prob
        
    def select_transform(self):
        
        if self.rng.random() < self.prob:
            return Solarize(self.threshold)
        
        else:
            return EmptyTransform()
    

class RandomRotation(_RandomSyncedTransform):
    def __init__(
        self,
        degrees: float,
        prob: float,
        num_samples: int,
        seed: Optional[int] = None,
    ):
    
        super().__init__(num_samples, seed)
        self.prob = prob
        self.degrees = degrees
    
    def select_transform(self):
        
        if self.rng.random() < self.prob:
            degrees = self.rng.uniform(-self.degrees, self.degrees)
            return Rotation(degrees = degrees)
        else:
            return EmptyTransform()
