from typing import Any, Dict, List

from torchvision import transforms

from .transform import _as_keys, _Transform


class MoCoAugmentationTransform(_Transform):
    """Default MoCo augmentations applied to every listed PIL image.

    - 224x224 crop from a randomly resized image
    - Random color jittering
    - Random grayscale conversion
    - Random horizontal flip
    - ``ToTensor`` and ImageNet normalization

    Each listed key is replaced by its augmented tensor. Keys draw their
    random parameters independently (the augmentation is torchvision's), so
    list one image key and use ``MultiViewTransform`` for the two MoCo views.
    """

    def __init__(self, keys_to_transform: List[str]):
        """
        Parameters
        ----------
        keys_to_transform : List[str]
            Keys of the PIL images to augment.
        """
        self.keys_to_transform = _as_keys(keys_to_transform)
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

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return self._map_keys(data, self.augment)
