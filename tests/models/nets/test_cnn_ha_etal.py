import torch

from minerva.models.nets.cnn import CNN_HaEtAl_1D, CNN_HaEtAl_2D


def test_cnn_ha_etal_1d_forward():
    input_shape = (1, 6, 60)
    model = CNN_HaEtAl_1D(input_shape=input_shape)
    assert model is not None

    x = torch.rand(1, *input_shape)
    y = model(x)
    assert y is not None


def test_cnn_ha_etal_2d_forward():
    input_shape = (1, 6, 60)
    model = CNN_HaEtAl_2D(input_shape=input_shape)
    assert model is not None

    x = torch.rand(1, *input_shape)
    y = model(x)
    assert y is not None
