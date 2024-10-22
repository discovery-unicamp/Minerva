import torchvision.transforms as T
from torchvision.transforms import GaussianBlur
from .transform import _Transform
from typing import Union, Tuple
import numpy as np
import torch


class BYOLTransform(_Transform):
    def __init__(self,
                input_size = 224,
                degrees=5,
                r_prob=0.5,
                h_prob=0.5,
                v_prob=0.5,
                collor_jitter_prob=0.5,
                grayscale_prob=0.2,
                gaussian_blur_prob=0.5,
                solarize_prob=0.0
                ):
        self.transformV1 = T.Compose([
            T.RandomCrop(size=input_size, pad_if_needed=True, padding_mode='reflect'),
            T.RandomApply([T.RandomRotation(degrees=degrees)], p=r_prob),
            T.RandomHorizontalFlip(p=h_prob),
            T.RandomVerticalFlip(p=v_prob),
            T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=collor_jitter_prob),
            T.RandomGrayscale(p=grayscale_prob),
            T.RandomApply([GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=gaussian_blur_prob),
            T.RandomSolarize(solarize_prob),
        ])
        self.transformV2 = T.Compose([
            T.RandomCrop(size=input_size, pad_if_needed=True, padding_mode='reflect'),
            T.RandomApply([T.RandomRotation(degrees=degrees)], p=r_prob),
            T.RandomHorizontalFlip(p=h_prob),
            T.RandomVerticalFlip(p=v_prob),
            T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=collor_jitter_prob),
            T.RandomGrayscale(p=grayscale_prob),
            T.RandomApply([GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=gaussian_blur_prob),
            T.RandomSolarize(solarize_prob),
        ])
        
    def __call__(self, x:Union[np.array, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).type(torch.FloatTensor)        
        v1 = self.transformV1(x)
        v2 = self.transformV2(x)
        return v1, v2
