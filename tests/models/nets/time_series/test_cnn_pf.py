import torch

from minerva.models.nets.time_series.cnns import CNN_PF_2D, CNN_PFF_2D


def test_cnn_pf_forward():
    input_shape = (1, 6, 60)
    model = CNN_PF_2D(input_shape=input_shape, pad_at=3)
    assert model is not None

    x = torch.rand(1, *input_shape)
    y = model(x)
    assert y is not None


def test_cnn_ha_pff_forward():
    input_shape = (1, 6, 60)
    model = CNN_PFF_2D(input_shape=input_shape, pad_at=3)
    assert model is not None

    x = torch.rand(1, *input_shape)
    y = model(x)
    assert y is not None
