import torch
from sslt.models.nets.unet import UNet


def test_unet():
    # Test the class instantiation
    model = UNet()
    assert model is not None

    # Generate a random input tensor (B, C, H, W) and the random mask of the
    # same shape
    input_shape = (2, 1, 500, 500)
    x = torch.rand(*input_shape)
    mask = torch.rand(*input_shape)

    # Test the forward method
    output = model(x)
    assert (
        output.shape == input_shape
    ), f"Expected output shape {input_shape}, but got {output.shape}"

    # Test the training step
    loss = model.training_step((x, mask), 0).item()
    assert loss is not None
    assert loss >= 0, f"Expected non-negative loss, but got {loss}"