import lightning as L
import torch
import torchmetrics

from minerva.utils.data import RandomDataModule

from minerva.models.nets.deeplabv3 import DeepLabV3Model

def test_deeplabv3_forward():

    # Test the class instantiation
    model = DeepLabV3Model()
    assert model is not None

    # Generate a random input tensor (B, C, H, W) and the random mask of the
    # same shape
    input_shape = (2, 3, 701, 255)
    x = torch.rand(*input_shape)
    mask = torch.rand(*input_shape)

    # Test the forward method
    output = model(x)
    assert (
        # Batch = 1
        # 6 output classes (default)
        # Width = 701
        # Height = 255
        output.shape[0] == 2 and 
         output.shape[1] == 6 and
          output.shape[2] == 701 and 
           output.shape[3] == 255
    ), f"Expected output shape {input_shape}, but got {output.shape}"
