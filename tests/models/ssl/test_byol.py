import torch
from torch.nn import Sequential, Conv2d, CrossEntropyLoss
from torchvision.transforms import Resize
import numpy as np

from minerva.models.ssl.lfr import RepeatedModuleList, LearnFromRandomnessModel
from minerva.models.nets.image.deeplabv3 import DeepLabV3Backbone
from minerva.models.ssl.byol import BYOL

def test_byol(): 

    backbone = DeepLabV3Backbone()
    
    # Testing model instantiation
    
    model = BYOL(backbone=backbone)
    
    assert model is not None

    x = np.random.rand(2, 3, 256, 256)
    x_tensor = torch.tensor(x, dtype=torch.float32)
    
    # Testing both forward methods
    
    x_forward = model.forward(x_tensor)
    x_forward_momentum = model.forward_momentum(x_tensor)
    
    assert x_forward.shape == x_forward_momentum.shape