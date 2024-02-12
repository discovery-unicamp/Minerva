import torch
from sslt.models.nets.unet import UNet

def test_unet():
    # Test the class instantiation
    model = UNet()
    assert model is not None

    # Generate a random input tensor (B, C, H, W)
    input_shape = (2, 1, 500, 500)
    x = torch.rand(*input_shape)
    
    # Test the forward method
    output = model(x)
    assert output.shape == input_shape