from typing import Optional, Sequence, Tuple

import numpy as np
from torchvision import transforms as v2

from minerva.transforms.transform import Normalize, TransformPipeline, _Transform


class _ContextTransform(_Transform):
    """
    Base class for context transforms that operate on paired data samples.

    Context transforms are specialized transforms that work with paired data such as
    image-mask pairs, where the transformation needs to be applied consistently to both
    components. Unlike regular transforms that operate on single inputs, context transforms
    receive a tuple of related data and must maintain the relationship between the components
    during transformation. This is essential for tasks like segmentation where the spatial
    correspondence between image and mask must be preserved.

    Parameters
    ----------
    x : Tuple[np.ndarray, np.ndarray]
        A tuple containing paired data, typically (image, mask) or (data, label),
        where both components need to be transformed consistently.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The transformed pair of data with the same structure as the input but
        with transformations applied consistently to both components.
    """

    def __call__(
        self, x: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply the context transformation to paired data.

        This method should be overridden in subclasses to define the specific
        transformation logic that operates on both components of the input tuple
        simultaneously, ensuring consistency between related data.

        Parameters
        ----------
        x : Tuple[np.ndarray, np.ndarray]
            A tuple containing the paired data to be transformed.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The transformed pair of data.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement __call__ method")


class ContextTransformPipeline(TransformPipeline):
    def __init__(self, transforms: Sequence[_Transform]):
        """
        A transform pipeline that applies a sequence of transforms to image-mask pairs.

        This pipeline extends the basic TransformPipeline to handle both context transforms
        (which operate on image-mask pairs simultaneously) and regular transforms (which
        are applied separately to each component). It intelligently determines whether each
        transform in the sequence is a context transform or regular transform and applies
        it appropriately, maintaining consistency between the image and mask throughout
        the pipeline.

        Parameters
        ----------
        transforms : Sequence[_Transform]
            A sequence of transforms to be applied to the input data. Can contain a mix
            of _ContextTransform instances (applied to the pair) and regular _Transform
            instances (applied separately to image and mask).
        """
        self.transforms = transforms

    def __call__(self, x: Tuple[np.ndarray, np.ndarray]):

        for transform in self.transforms:
            if isinstance(transform, _ContextTransform):

                x = transform(x)
            else:

                x_a = transform(x[0])
                x_b = transform(x[1])
                x = (x_a, x_b)

        return x


class ClassRatioCrop(_ContextTransform):

    def __init__(
        self,
        target_size: Tuple[int, int],
        max_ratio: float = 0.75,
        max_attempts: int = 10,
        ignore_index: int = -1,
        seed: Optional[int] = None,
    ) -> None:
        """
        A context transform that crops image-mask pairs while controlling class distribution.

        This transform attempts to crop both the input image and its corresponding mask to a target size
        while ensuring that no single class dominates the cropped region beyond a specified ratio.
        It randomly selects crop locations and validates the class distribution in the mask before
        accepting the crop. This helps maintain balanced class representation in cropped samples,
        which is particularly useful for segmentation tasks where class imbalance can be problematic.

        Parameters
        ----------
        target_size : Tuple[int, int]
            The target size of the crop as (height, width) in pixels.
        max_ratio : float, default=0.75
            The maximum ratio of pixels that any single class can occupy in the cropped mask.
            Must be between 0.0 and 1.0. Lower values enforce more balanced class distribution.
        max_attempts : int, default=10
            The maximum number of random crop attempts before accepting the last available crop.
            Higher values increase the chance of finding a well-balanced crop but may slow processing.
        ignore_index : int, default=-1
            Label value in the mask to be ignored when computing class ratios.
            Pixels with this label will not contribute to the class distribution calculation.
        seed : Optional[int], default=None
            Random seed for reproducible cropping. If None, uses system randomness.
        """
        self.target_h_size = target_size[0]
        self.target_w_size = target_size[1]
        self.max_ratio = max_ratio
        self.max_attempts = max_attempts
        self.ignore_index = ignore_index
        self.rng = np.random.default_rng(seed)

    def __call__(self, x: Tuple[np.ndarray, np.ndarray]):
        img, mask = x
        h, w = mask.shape

        for i in range(self.max_attempts):
            top = (
                self.rng.integers(0, h - self.target_h_size + 1)
                if h - self.target_h_size > 0
                else 0
            )
            left = (
                self.rng.integers(0, w - self.target_w_size + 1)
                if w - self.target_w_size > 0
                else 0
            )

            cropped_mask = mask[
                top : top + self.target_h_size, left : left + self.target_w_size
            ]

            # Exclude ignore_index from ratio calculation
            valid_pixels = cropped_mask[cropped_mask != self.ignore_index]
            if valid_pixels.size > 0:
                _, counts = np.unique(valid_pixels, return_counts=True)
                class_ratios = counts / valid_pixels.size
                if np.max(class_ratios) <= self.max_ratio or i == self.max_attempts - 1:
                    cropped_img = img[
                        top : top + self.target_h_size, left : left + self.target_w_size
                    ]
                    return cropped_img, cropped_mask
            elif i == self.max_attempts - 1:
                # fallback: accept crop with only ignore_index
                cropped_img = img[
                    top : top + self.target_h_size, left : left + self.target_w_size
                ]
                return cropped_img, cropped_mask


class DataOnlyTransform(_ContextTransform):
    def __init__(self, transform: _Transform) -> None:
        """
        A context transform that applies a regular transform only to the data component.

        This transform wrapper allows regular transforms to be used in context transform
        pipelines by applying them only to the first element of the input tuple (typically
        the image/data) while leaving the second element (typically the mask/label) unchanged.
        This is useful when you want to apply transforms like normalization, tensor conversion,
        or other data preprocessing steps that should only affect the input data and not the labels.

        Parameters
        ----------
        transform : _Transform
            The regular transform to apply to the data component only. This can be any
            transform that operates on single inputs (normalization, tensor conversion, etc.).
        """
        self.transform = transform

    def __call__(
        self, x: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        data, label = x
        transformed_data = self.transform(data)
        return transformed_data, label
