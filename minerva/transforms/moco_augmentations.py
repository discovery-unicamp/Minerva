# minerva/transforms/moco_augmentations.py

import torch
from torchvision import transforms
from PIL import Image
from typing import Tuple


class MoCoRandomAugmentations:
    """
    Default MoCo augmentations.

    Based on the paper's technical details:
    - 224×224-pixel crop from a randomly resized image
    - Random color jittering
    - Random horizontal flip
    - Random grayscale conversion
    """

    def __init__(self):
        """Initialize MoCo's default augmentation pipeline."""
        self.augment = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __call__(self, image: Image.Image) -> Tuple[torch.Tensor, None]:
        """
        Apply MoCo augmentations to a PIL Image.

        Parameters
        ----------
        image : PIL.Image.Image
            Input PIL image

        Returns
        -------
        Tuple[torch.Tensor, None]
            Augmented tensor and None (for compatibility with other augmentation classes)
        """
        return self.augment(image), None
