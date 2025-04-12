import torch

from minerva.models.nets.time_series.imu_transformer import (
    IMUCNN,
    IMUTransformerEncoder,
)


def test_imu_transformer_forward():
    input_shape = (6, 60)
    model = IMUTransformerEncoder(input_shape=input_shape)
    assert model is not None

    x = torch.rand(1, *input_shape)
    y = model(x)
    assert y is not None


def test_imu_transformer_creation(simple_torchmetrics):
    input_shape = (6, 60)
    # with metrics
    model = IMUTransformerEncoder(input_shape=input_shape, **simple_torchmetrics)


def test_imu_cnn_forward():
    input_shape = (6, 60)
    model = IMUCNN(input_shape=input_shape)
    assert model is not None

    x = torch.rand(1, *input_shape)
    y = model(x)
    assert y is not None


def test_imu_cnn_creation(simple_torchmetrics):
    input_shape = (6, 60)
    # with metrics
    model = IMUCNN(input_shape=input_shape, **simple_torchmetrics)
