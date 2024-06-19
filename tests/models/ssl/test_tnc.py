from torch import nn, rand
from minerva.models.ssl.tnc import TNC
    
def test_tnc_model():

    # Instantitate encoder, density_estimator, and auto_regressor
    encoder = None # TODO
    discriminator = None # TODO

    # Test the class instantiation
    model = TNC(encoder=encoder, discriminator=discriminator)
    assert model is not None

    # Generate a random input tensor (B, C, T) and the random mask of the
    # same shape
    input_shape = (2, 6, 100) # TODO: fix the input shape
    expected_output_shape = (2, 256) # TODO: fix the expected output shape
    x = rand(*input_shape)

    # Test the forward method
    output = model.forward(x)
    assert (
        output.shape == expected_output_shape
    ), f"Expected output shape {expected_output_shape}, but got {output.shape}"

    # Test the training step
    loss = model.training_step(x, 0).item()
    assert loss is not None
    assert loss >= 0, f"Expected non-negative loss, but got {loss}"
