from typing import Any

import numpy as np

from minerva.transforms.transform import _Transform


class ClassRatioCrop(_Transform):

    def __init__(
        self,
        target_h_size: int,
        target_w_size: int,
        cat_max_ratio: float = 0.75,
        max_attempts: int = 10,
    ) -> None:
        """Crop the input data to a target size, while keeping the ratio of classes in the image.

        Parameters
        ----------
        target_h_size : int
            The target height of the crop.
        target_w_size : int
            The target width of the crop.
        cat_max_ratio : float, optional
            The maximum ratio of pixels of a single class in the crop, by default 0.75
        max_attempts : int, optional
            The maximum number of attempts to crop the image, by default 10
        """
        self.target_h_size = target_h_size
        self.target_w_size = target_w_size
        self.cat_max_ratio = cat_max_ratio
        self.max_attempts = max_attempts
        self.crop_coords = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h, w = x.shape[:2]

        if self.crop_coords is None:

            if x.dtype != np.uint8:
                raise ValueError(
                    "You must provide a mask first to use this functionality. For that you enable support_context_transforms if your dataset supports it, or use a different dataset that does supports it."
                )

            for _ in range(self.max_attempts):
                # Randomly select the top-left corner for the crop
                top = np.random.randint(0, h - self.target_h_size + 1)
                left = np.random.randint(0, w - self.target_w_size + 1)

                # Extract the crop from both image and label
                cropped_image = x[
                    top : top + self.target_h_size, left : left + self.target_w_size
                ]

                # Calculate the proportion of the most frequent class in the crop
                _, counts = np.unique(cropped_image, return_counts=True)
                class_ratios = counts / (self.target_h_size * self.target_w_size)

                if np.max(class_ratios) <= self.cat_max_ratio:
                    self.crop_coords = (top, left)
                    return cropped_image

            # If no valid crop was found, return the last crop (without meeting the ratio constraint)
            self.crop_coords = (top, left)
            return cropped_image

        else:
            top, left = self.crop_coords
            self.crop_coords = None
            return x[top : top + self.target_h_size, left : left + self.target_w_size]
