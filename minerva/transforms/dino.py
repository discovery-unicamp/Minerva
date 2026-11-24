# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.
#
# References:
#   https://github.com/facebookresearch/dinov3/blob/main/dinov3/data/transforms.py
#   https://github.com/facebookresearch/dinov3/blob/main/dinov3/data/augmentations.py

import math
import random
from typing import Any, Dict, List, Sequence, Tuple
from torchvision.transforms import v2

import numpy as np
import torch
from torch import nn

from .transform import _as_keys, _Transform


class GaussianBlur(v2.RandomApply):
    """
    Apply Gaussian Blur to the PIL image.

    Parameters
    ----------
    p : float, default 0.5
        Probability of applying blur.
    radius_min : float, default 0.1
        Minimum Gaussian kernel sigma.
    radius_max : float, default 2.0
        Maximum Gaussian kernel sigma.
    """

    def __init__(
        self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0
    ):
        # NOTE: torchvision is applying 1 - probability to return the original image
        keep_p = 1 - p
        transform = v2.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=keep_p)


# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> v2.Normalize:
    """
    Create a torchvision v2 Normalize transform with specified mean and std.

    Parameters
    ----------
    mean : Sequence[float], default (0.485, 0.456, 0.406)
        Per-channel sequence of mean values.
    std : Sequence[float], default (0.229, 0.224, 0.225)
        Per-channel sequence of standard deviations.

    Returns
    -------
    v2.Normalize
        Configured normalization transform.
    """
    return v2.Normalize(mean=mean, std=std)


# TODO Change this to use new Minerva transforms API once it is available
class DINOTransform(_Transform):
    """
    Multi-crop data augmentation pipeline for DINO self-supervised learning.

    Called on a dict, each listed key ``k`` is replaced by flat keys holding
    stacked crops: ``{k}_global_crops`` (n_global, C, H, W),
    ``{k}_global_crops_teacher`` (n_global, C, H, W), ``{k}_local_crops``
    (n_local, C, h, w), ``{k}_gram_teacher_crops`` (if enabled) and
    ``{k}_offsets``. The nested dict of lists consumed by legacy code is
    returned by :meth:`augment`.

    Parameters
    ----------
    keys_to_transform : List[str]
        Keys of the raw images to augment.
    global_crops_scale : list of float, default [0.32, 1.0]
        Scale range for global crops.
    local_crops_scale : list of float, default [0.05, 0.32]
        Scale range for local crops.
    local_crops_number : int, default 8
        Number of small local crops to extract.
    global_crops_size : int, default 224
        Resolution (height and width) of global crops.
    local_crops_size : int, default 96
        Resolution (height and width) of local crops.
    gram_teacher_crops_size : int, optional
        Resolution for Gram teacher crops, if enabled.
    gram_teacher_no_distortions : bool, default False
        If True, Gram teacher crops will not undergo color jittering or distortions.
    teacher_no_color_jitter : bool, default False
        If True, teacher views will not receive color distortions.
    local_crops_subset_of_global_crops : bool, default False
        If True, extracts local crops as subsets from within global crops.
    patch_size : int, default 16
        Patch size used when local crops are extracted as subsets of global crops.
    share_color_jitter : bool, default False
        Whether to share identical color jittering across crops.
    horizontal_flips : bool, default True
        Whether to apply random horizontal flips.
    mean : tuple of float, default IMAGENET_DEFAULT_MEAN
        Normalization RGB mean values.
    std : tuple of float, default IMAGENET_DEFAULT_STD
        Normalization RGB standard deviations.
    """

    def __init__(
        self,
        keys_to_transform: List[str],
        global_crops_scale=[0.32, 1.0],
        local_crops_scale=[0.05, 0.32],
        local_crops_number=8,
        global_crops_size=224,
        local_crops_size=96,
        gram_teacher_crops_size=None,
        gram_teacher_no_distortions=False,
        teacher_no_color_jitter=False,
        local_crops_subset_of_global_crops=False,
        patch_size=16,
        share_color_jitter=False,
        horizontal_flips=True,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    ):
        self.keys_to_transform = _as_keys(keys_to_transform)
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.gram_teacher_crops_size = gram_teacher_crops_size
        self.gram_teacher_no_distortions = gram_teacher_no_distortions
        self.teacher_no_color_jitter = teacher_no_color_jitter
        self.local_crops_subset_of_global_crops = local_crops_subset_of_global_crops
        self.patch_size = patch_size
        self.share_color_jitter = share_color_jitter
        self.mean = mean
        self.std = std

        # logger.info("###################################")
        # logger.info("Using data augmentation parameters:")
        # logger.info(f"global_crops_scale: {global_crops_scale}")
        # logger.info(f"local_crops_scale: {local_crops_scale}")
        # logger.info(f"local_crops_number: {local_crops_number}")
        # logger.info(f"global_crops_size: {global_crops_size}")
        # logger.info(f"local_crops_size: {local_crops_size}")
        # logger.info(f"gram_crops_size: {gram_teacher_crops_size}")
        # logger.info(f"gram_teacher_no_distortions: {gram_teacher_no_distortions}")
        # logger.info(f"teacher_no_color_jitter: {teacher_no_color_jitter}")
        # logger.info(f"local_crops_subset_of_global_crops: {local_crops_subset_of_global_crops}")
        # logger.info(f"patch_size if local_crops_subset_of_global_crops: {patch_size}")
        # logger.info(f"share_color_jitter: {share_color_jitter}")
        # logger.info(f"horizontal flips: {horizontal_flips}")
        # logger.info("###################################")

        # Global crops and gram teacher crops can have different sizes. We first take a crop of the maximum size
        # and then resize it to the desired size for global and gram teacher crops.
        global_crop_max_size = max(
            global_crops_size, gram_teacher_crops_size if gram_teacher_crops_size else 0
        )

        # random resized crop and flip
        self.geometric_augmentation_global = v2.Compose(
            [
                v2.RandomResizedCrop(
                    global_crop_max_size,
                    scale=global_crops_scale,
                    interpolation=v2.InterpolationMode.BICUBIC,
                ),
                v2.RandomHorizontalFlip(p=0.5 if horizontal_flips else 0.0),
            ]
        )

        resize_global = (
            nn.Identity()
        )  # Resize transform applied to global crops after random crop
        self.resize_global_post_transf = (
            nn.Identity()
        )  # Resize transform applied to global crops after all other transforms
        self.resize_gram_teacher = (
            None  # Resize transform applied to crops for gram teacher
        )
        if gram_teacher_crops_size is not None:
            # All resize transforms will do nothing if the crop size is already the desired size.
            if gram_teacher_no_distortions:
                # When there a no distortions for the gram teacher crop, we can resize before the distortions.
                # This is the preferred order, because it keeps the image size for the augmentations consistent,
                # which matters e.g. for GaussianBlur.
                resize_global = v2.Resize(
                    global_crops_size,
                    interpolation=v2.InterpolationMode.BICUBIC,
                )
            else:
                # When there a no distortions for the gram teacher crop, we need to resize after the distortions,
                # because the distortions are shared between global and gram teacher crops.
                self.resize_global_post_transf = v2.Resize(
                    global_crops_size,
                    interpolation=v2.InterpolationMode.BICUBIC,
                )

            self.resize_gram_teacher = v2.Resize(
                gram_teacher_crops_size,
                interpolation=v2.InterpolationMode.BICUBIC,
            )

        self.geometric_augmentation_local = v2.Compose(
            [
                v2.RandomResizedCrop(
                    local_crops_size,
                    scale=local_crops_scale,
                    interpolation=v2.InterpolationMode.BICUBIC,
                ),
                v2.RandomHorizontalFlip(p=0.5 if horizontal_flips else 0.0),
            ]
        )

        # color distortions / blurring
        color_jittering = v2.Compose(
            [
                v2.RandomApply(
                    [
                        v2.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                v2.RandomGrayscale(p=0.2),
            ]
        )

        global_transfo1_extra = GaussianBlur(p=1.0)

        global_transfo2_extra = v2.Compose(
            [
                GaussianBlur(p=0.1),
                v2.RandomSolarize(threshold=0.5, p=0.2),
            ]
        )

        local_transfo_extra = GaussianBlur(p=0.5)

        # normalization
        self.normalize = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                make_normalize_transform(mean=mean, std=std),
            ]
        )

        if self.share_color_jitter:
            self.color_jittering = color_jittering
            self.global_transfo1 = v2.Compose(
                [resize_global, global_transfo1_extra, self.normalize]
            )
            self.global_transfo2 = v2.Compose(
                [resize_global, global_transfo2_extra, self.normalize]
            )
            self.local_transfo = v2.Compose([local_transfo_extra, self.normalize])
        else:
            self.global_transfo1 = v2.Compose(
                [resize_global, color_jittering, global_transfo1_extra, self.normalize]
            )
            self.global_transfo2 = v2.Compose(
                [resize_global, color_jittering, global_transfo2_extra, self.normalize]
            )
            self.local_transfo = v2.Compose(
                [color_jittering, local_transfo_extra, self.normalize]
            )

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply multi-crop augmentations to every listed key (see class docstring)."""
        for key in self.keys_to_transform:
            crops = self.augment(data.pop(key))
            for name in (
                "global_crops",
                "global_crops_teacher",
                "local_crops",
                "gram_teacher_crops",
            ):
                if name in crops:
                    data[f"{key}_{name}"] = torch.stack(crops[name])
            data[f"{key}_offsets"] = crops["offsets"]
        return data

    def augment(self, image):
        """
        Apply multi-crop augmentations to an input image.

        Parameters
        ----------
        image : PIL.Image.Image or torch.Tensor
            Input raw image.

        Returns
        -------
        dict
            Dictionary containing transformed views:
            - 'global_crops': list of global crop tensors.
            - 'global_crops_teacher': list of global crop tensors for the teacher.
            - 'local_crops': list of local crop tensors.
            - 'gram_teacher_crops': list of Gram teacher crops (if enabled).
            - 'offsets': crop offset coordinates.
        """
        output = {}
        output["weak_flag"] = True  # some residual from mugs

        if self.share_color_jitter:
            image = self.color_jittering(image)

        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1_transf = self.global_transfo1(im1_base)
        global_crop_1 = self.resize_global_post_transf(global_crop_1_transf)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2_transf = self.global_transfo2(im2_base)
        global_crop_2 = self.resize_global_post_transf(global_crop_2_transf)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        if self.teacher_no_color_jitter:
            output["global_crops_teacher"] = [
                self.normalize(im1_base),
                self.normalize(im2_base),
            ]
        else:
            output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        if self.gram_teacher_crops_size is not None:
            # crops for gram teacher:
            if self.gram_teacher_no_distortions:
                gram_crop_1 = self.normalize(self.resize_gram_teacher(im1_base))
                gram_crop_2 = self.normalize(self.resize_gram_teacher(im2_base))
            else:
                gram_crop_1 = self.resize_gram_teacher(global_crop_1_transf)
                gram_crop_2 = self.resize_gram_teacher(global_crop_2_transf)
            output["gram_teacher_crops"] = [gram_crop_1, gram_crop_2]

        # local crops:
        if self.local_crops_subset_of_global_crops:
            _local_crops = [
                self.local_transfo(im1_base)
                for _ in range(self.local_crops_number // 2)
            ] + [
                self.local_transfo(im2_base)
                for _ in range(self.local_crops_number // 2)
            ]

            local_crops = []
            offsets = []
            gs = self.global_crops_size
            ls = self.local_crops_size
            for img in _local_crops:
                rx, ry = (
                    torch.randint(0, (gs - ls) // self.patch_size, (2,)).tolist()
                )
                rx, ry = rx * self.patch_size, ry * self.patch_size
                local_crops.append(img[:, rx : rx + ls, ry : ry + ls])
                offsets.append((rx, ry))

            output["local_crops"] = local_crops
            output["offsets"] = offsets
        else:
            local_crops = [
                self.local_transfo(self.geometric_augmentation_local(image))
                for _ in range(self.local_crops_number)
            ]
            output["local_crops"] = local_crops
            output["offsets"] = ()

        return output


class MaskingGenerator:
    """
    Block-wise patch masking generator for masked image modeling (e.g. iBOT in DINO).

    Parameters
    ----------
    input_size : int or tuple of int
        Grid dimensions (height, width) in terms of patch count.
    num_masking_patches : int, optional
        Target number of patches to mask.
    min_num_patches : int, default 4
        Minimum patch area for a single rectangular mask block.
    max_num_patches : int, optional
        Maximum patch area for a single rectangular mask block.
    min_aspect : float, default 0.3
        Minimum aspect ratio of generated mask blocks.
    max_aspect : float, optional
        Maximum aspect ratio of generated mask blocks.
    """

    def __init__(
        self,
        input_size,
        num_masking_patches=None,
        min_num_patches=4,
        max_num_patches=None,
        min_aspect=0.3,
        max_aspect=None,
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = (
            max_num_patches
            if max_num_patches is not None
            else (
                num_masking_patches
                if num_masking_patches is not None
                else self.num_patches
            )
        )

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %s, %.3f ~ %.3f)" % (
            self.height,
            self.width,
            self.min_num_patches,
            self.max_num_patches if self.max_num_patches is not None else 0,
            str(self.num_masking_patches),
            self.log_aspect_ratio[0],
            self.log_aspect_ratio[1],
        )
        return repr_str

    def get_shape(self) -> Tuple[int, int]:
        """Return the (height, width) grid shape."""
        return self.height, self.width

    def _mask(self, mask: np.ndarray, max_mask_patches: int) -> int:
        """
        Attempt to mask a random rectangular block in-place on the mask grid.

        Parameters
        ----------
        mask : np.ndarray
            Boolean mask array.
        max_mask_patches : int
            Maximum number of patches allowed to be masked in this step.

        Returns
        -------
        int
            Number of newly masked patches.
        """
        delta = 0
        for _ in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top : top + h, left : left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self, num_masking_patches: int = 0) -> np.ndarray:
        """
        Generate a boolean mask array with the specified number of masked patches.

        Parameters
        ----------
        num_masking_patches : int, default 0
            Number of patches to mask.

        Returns
        -------
        np.ndarray
            2D boolean mask array of shape (height, width) where True denotes masked.
        """
        mask = np.zeros(shape=self.get_shape(), dtype=bool)
        mask_count = 0
        while mask_count < num_masking_patches:
            max_mask_patches = num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        return self.complete_mask_randomly(mask, num_masking_patches)

    def complete_mask_randomly(
        self, mask: np.ndarray, num_masking_patches: int
    ) -> np.ndarray:
        """
        Fill any remaining unmasked patches randomly to reach exact patch target.

        Parameters
        ----------
        mask : np.ndarray
            Current boolean mask.
        num_masking_patches : int
            Total target masked patch count.

        Returns
        -------
        np.ndarray
            Completed boolean mask array of shape (height, width).
        """
        shape = mask.shape
        m2 = mask.flatten()
        to_add = np.random.choice(
            np.where(~m2)[0], size=num_masking_patches - m2.sum(), replace=False
        )
        m2[to_add] = True
        return m2.reshape(shape)
