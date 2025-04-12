from .base import SimpleSupervisedModel
from .image.deeplabv3 import DeepLabV3
from .image.setr import SETR_PUP
from .image.unet import UNet
from .image.wisenet import WiseNet
from .mlp import MLP


__all__ = ["SimpleSupervisedModel", "DeepLabV3", "SETR_PUP", "UNet", "WiseNet", "MLP"]
