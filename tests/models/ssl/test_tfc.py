import torch
from minerva.models.ssl.tfc import TFC
    
def test_tfc_model():

    # Instantitate encoder, density_estimator, and auto_regressor
    time_encoder = None # TODO: instantiate a valid encoder
    time_projector = None # TODO: instantiate a valid projector
    frequency_encoder = None # TODO: instantiate a valid encoder
    frequency_projector = None # TODO: instantiate a valid projector

    # Test the class instantiation
    model = TFC(time_encoder=time_encoder, time_projector=time_projector,
                frequency_encoder=frequency_encoder, frequency_projector=frequency_projector)
    assert model is not None

    # Generate a random input tensor (B, C, T) and the random mask of the
    # same shape
    input_shape = (2, 6, 100) # TODO: fix the input shape
    expected_output_shape = (2, 256) # TODO: fix the expected output shape
    x = torch.rand(*input_shape)

    # Test the forward method
    output = model.forward(x)
    assert (
        output.shape == expected_output_shape
    ), f"Expected output shape {expected_output_shape}, but got {output.shape}"

    # Test the training step
    loss = model.training_step(x, 0).item()
    assert loss is not None
    assert loss >= 0, f"Expected non-negative loss, but got {loss}"
