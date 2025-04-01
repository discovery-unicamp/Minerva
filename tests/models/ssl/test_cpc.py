import torch
from minerva.models.ssl.cpc import CPC
from minerva.models.nets.cpc_networks import (
    CNN,
    HARCPCAutoregressive,
    PredictionNetwork,
)


def test_cpc_model():
    # Instantiate encoder, autoregressive, and prediction head
    g_enc = CNN()
    g_ar = HARCPCAutoregressive()

    # Test the class instantiation
    model = CPC(
        g_enc=g_enc,
        g_ar=g_ar,
        prediction_head_in_channels=256,
        prediction_head_out_channels=128,
    )
    assert model is not None, "Model instantiation failed."

    # Generate a random input tensor (B, C, T) and the expected output shape
    input_shape = (1, 6, 60)
    expected_output_shape_z = (1, 60, 128)

    # Create random input data
    x = torch.rand(*input_shape)

    # Test the forward method
    # Output forward method: return z,y
    output = model.forward(x)
    if isinstance(output, tuple):
        output = output[0]

    assert (
        output.shape == expected_output_shape_z
    ), f"Expected output shape {expected_output_shape_z}, but got {output.shape}"
