import torch
from minerva.models.adapters import MaxPoolingTransposingSqueezingAdapter


def test_MaxPoolingTransposingSqueezingAdapter():

    input_tensor = torch.randn(10, 128, 64)  # Example input tensor with shape (batch_size, time_steps, features)
    
    adapter = MaxPoolingTransposingSqueezingAdapter(kernel_size=128)
    
    output_tensor = adapter(input_tensor)

    expected_output_shape = torch.Size([10, 64]) #batch_size x features
    
    assert (
        output_tensor.shape == expected_output_shape
    ), f"Expected output shape {expected_output_shape}, but got {output_tensor.shape}"

    


    