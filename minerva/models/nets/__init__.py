from .base import SimpleSupervisedModel
from .deeplabv3 import DeepLabV3Model
from .setr import SETR_PUP
from .unet import UNet
from .wisenet import WiseNet

__all__ = [
    "SimpleSupervisedModel",
    "DeepLabV3Model",
    "SETR_PUP",
    "UNet",
    "WiseNet",
]
