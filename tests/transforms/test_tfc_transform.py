from minerva.transforms.tfc import TFC_Transforms
import numpy as np
import torch


def test_tfc_transform_numpy():
    tranform = TFC_Transforms()
    # Create a dummy input
    x_original = np.random.rand(10, 20)

    # Apply the transform pipeline
    returns = tranform(x_original)
    assert len(returns) == 4
    x, y1, freq, y2 = returns

    # Check if the transformed data has the same shape as the input
    assert x.shape == y1.shape == freq.shape == y2.shape == x_original.shape


def test_tfc_transform_tensor():
    tranform = TFC_Transforms()
    # Create a dummy input tensor
    x_original = torch.rand(10, 20)

    # Apply the transform pipeline
    returns = tranform(x_original)
    assert len(returns) == 4
    x, y1, freq, y2 = returns

    # Check if the transformed data has the same shape as the input
    assert x.shape == y1.shape == freq.shape == y2.shape == x_original.shape


test_tfc_transform_numpy()
test_tfc_transform_tensor()
