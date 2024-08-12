import torch

from minerva.models.nets.time_series.resnet import (
    ResNet1D_8,
    ResNetSE1D_5,
    ResNetSE1D_8,
)


def test_resnet_1d_8_forward():
    input_shape = (6, 60)
    model = ResNet1D_8(
        input_shape=input_shape,
        num_classes=6,
        learning_rate=1e-3,
    )
    assert model is not None

    x = torch.rand(1, *input_shape)
    y = model(x)
    assert y is not None


def test_resnet_se_1d_8_forward():
    input_shape = (6, 60)
    model = ResNetSE1D_8(
        input_shape=input_shape,
        num_classes=6,
        learning_rate=1e-3,
    )
    assert model is not None

    x = torch.rand(1, *input_shape)
    y = model(x)
    assert y is not None


def test_resnet_se_1d_5_forward():
    input_shape = (6, 60)
    model = ResNetSE1D_5(
        input_shape=input_shape,
        num_classes=6,
        learning_rate=1e-3,
    )
    assert model is not None

    x = torch.rand(1, *input_shape)
    y = model(x)
    assert y is not None
