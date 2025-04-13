import torch

from minerva.models.nets.time_series.inception_time import InceptionTime


def test_inception_time_forward():
    input_shape = (6, 60)
    model = InceptionTime(input_shape=input_shape)
    assert model is not None

    x = torch.rand(1, *input_shape)
    y = model(x)
    assert y is not None


def test_inception_time_creation(simple_torchmetrics):
    input_shape = (6, 60)
    # with metrics
    model = InceptionTime(input_shape=input_shape, **simple_torchmetrics)
