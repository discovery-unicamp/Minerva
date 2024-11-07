from typing import List, Optional, Union

import numpy as np
import cv2
from minerva.transforms.transform import Flip, _Transform
from typing import Tuple, Optional

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
    
    
class RandomCrop(_RandomSyncedTransform):

    def __init__(
        self,
        num_samples: int,
        output_size: Tuple[int, int],
        pad_mode: str = 'reflect',
        seed: Optional[int] = None
    ):
        """
        Initializes the RandomSyncedCrop transform.

        Parameters
        ----------
        num_samples : int
            The number of samples that will be transformed.
        output_size : Tuple[int, int]
            The desired output size (height, width) of the crop.
        pad_mode : str, optional
            Padding mode if padding is needed, by default 'reflect'.
        seed : Optional[int], optional
            A seed to ensure deterministic behavior, by default None.
        """
        super().__init__(num_samples, seed)
        self.output_size = output_size
        self.pad_mode = pad_mode

    def select_transform(self):
        """Selects a crop transform with random coordinates for the crop."""
        # Generate crop parameters once and use them for all samples
        def crop_transform(image: np.ndarray) -> np.ndarray:
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

            # Generate random start positions for cropping
            start_x = self.rng.integers(0, w - new_w + 1)
            start_y = self.rng.integers(0, h - new_h + 1)

            # Return cropped image
            return image[start_y:start_y + new_h, start_x:start_x + new_w]

        return crop_transform
    

class RandomRotation(_RandomSyncedTransform):
    """Synchronously applies a random rotation to each sample in a batch."""

    def __init__(self, num_samples: int, degrees: float, seed: Optional[int] = None):
        """
        Parameters
        ----------
        num_samples : int
            Number of samples in the batch.
        degrees : float
            Maximum degree for random rotation, chosen uniformly in [-degrees, degrees].
        seed : Optional[int]
            Seed for reproducible randomness, defaults to None.
        """
        super().__init__(num_samples, seed)
        self.degrees = degrees

    def select_transform(self):
        """Randomly selects a rotation angle and applies it to each sample."""
        angle = self.rng.uniform(-self.degrees, self.degrees)

        def rotate(image: np.ndarray) -> np.ndarray:
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(image, rotation_matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

        return rotate


class RandomGrayscale(_RandomSyncedTransform):

    def __init__(self, num_samples: int, prob: float = 0.1, seed: Optional[int] = None):
        """
        Parameters
        ----------
        num_samples : int
            Number of samples in the batch.
        prob : float
            Probability of converting an image to grayscale.
        seed : Optional[int]
            Seed for reproducible randomness, defaults to None.
        """
        super().__init__(num_samples, seed)
        self.prob = prob

    def select_transform(self):
        """Randomly determines if grayscale conversion will be applied."""
        convert_to_grayscale = self.rng.random() < self.prob

        def grayscale(image: np.ndarray) -> np.ndarray:
            if convert_to_grayscale:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                return np.stack([gray] * 3, axis=-1)
            return image

        return grayscale
    
    
class RandomSyncedSolarize(_RandomSyncedTransform):

    def __init__(self, num_samples: int, prob: float = 0.5, threshold_range: tuple = (64, 192), seed: Optional[int] = None):
        """
        Parameters
        ----------
        num_samples : int
            Number of samples in the batch.
        prob : float
            Probability of applying solarization.
        threshold_range : tuple
            Range from which the solarization threshold is randomly chosen.
        seed : Optional[int]
            Seed for reproducible randomness, defaults to None.
        """
        super().__init__(num_samples, seed)
        self.prob = prob
        self.threshold_range = threshold_range

    def select_transform(self):
        """Randomly determines if solarization will be applied and chooses a random threshold if it is."""
        apply_solarize = self.rng.random() < self.prob
        threshold = self.rng.integers(*self.threshold_range) if apply_solarize else None

        def solarize(image: np.ndarray) -> np.ndarray:
            if threshold is not None:
                return np.where(image < threshold, image, 255 - image)
            return image

        return solarize
    
