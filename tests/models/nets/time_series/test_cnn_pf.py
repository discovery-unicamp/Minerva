import torch

from minerva.models.nets.time_series.cnns import CNN_PF_2D, CNN_PFF_2D, CNN_PF_Backbone


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


def test_cnn_pf_forward_flatten():
    input_shape = (1, 6, 60)
    model = CNN_PF_Backbone(in_channels=1, pad_at=3, flatten=True)
    assert model is not None

    x = torch.rand(1, *input_shape)
    y = model(x)

    assert y is not None
    assert len(y.shape) == 2
    assert y.shape[0] == 1
    assert y.shape[1] > 0


def test_cnn_pf_forward_no_flatten():
    input_shape = (1, 6, 60)
    model = CNN_PF_Backbone(in_channels=1, pad_at=3, flatten=False)
    assert model is not None

    x = torch.rand(1, *input_shape)
    y = model(x)

    assert y is not None
    assert len(y.shape) > 2
    assert y.shape[0] == 1
