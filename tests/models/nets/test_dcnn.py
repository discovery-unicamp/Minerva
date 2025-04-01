import torch

from minerva.models.nets.dcnn import DCNN


def test_dcnn_model():
    # Test the class instantiation
    model = DCNN()
    assert model is not None

    # Test the forward method
    input_shape = (32, 1, 36, 68)
    expected_output_size = torch.Size([32, 6])
    x = torch.rand(*input_shape)
    output = model(x)
    
    assert (
        output.shape == expected_output_size
    ), f"Expected output shape {input_shape}, but got {output.shape}"