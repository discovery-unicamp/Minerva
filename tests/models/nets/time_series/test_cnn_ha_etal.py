import torch
import torchmetrics

from minerva.models.nets.time_series.cnns import (
    CNN_HaEtAl_1D,
    CNN_HaEtAl_2D,
    CNN_PF_2D,
    CNN_PFF_2D,
)


def test_cnn_ha_etal_1d_forward():
    input_shape = (1, 6, 60)
    model = CNN_HaEtAl_1D(input_shape=input_shape)
    assert model is not None

    x = torch.rand(1, *input_shape)
    y = model(x)
    assert y is not None


def test_cnn_ha_etal_1d_creation(simple_torchmetrics):
    input_shape = (1, 6, 60)
    # with metrics
    model = CNN_HaEtAl_1D(input_shape=input_shape, **simple_torchmetrics)


def test_cnn_ha_etal_2d_forward():
    input_shape = (1, 6, 60)
    model = CNN_HaEtAl_2D(input_shape=input_shape)
    assert model is not None

    x = torch.rand(1, *input_shape)
    y = model(x)
    assert y is not None


def test_cnn_ha_etal_2d_creation(simple_torchmetrics):
    input_shape = (1, 6, 60)
    # with metrics
    model = CNN_HaEtAl_2D(input_shape=input_shape, **simple_torchmetrics)


def test_cnn_pf_2d_forward():
    input_shape = (1, 6, 60)
    model = CNN_PF_2D(input_shape=input_shape, pad_at=3)
    assert model is not None

    x = torch.rand(1, *input_shape)
    y = model(x)
    assert y is not None


def test_cnn_pf_2d_creation(simple_torchmetrics):
    input_shape = (1, 6, 60)
    # with metrics
    model = CNN_PF_2D(input_shape=input_shape, pad_at=3, **simple_torchmetrics)
    
    
def test_cnn_pff_2d_forward():
    input_shape = (1, 6, 60)
    model = CNN_PFF_2D(input_shape=input_shape, pad_at=3)
    assert model is not None

    x = torch.rand(1, *input_shape)
    y = model(x)
    assert y is not None
    
def test_cnn_pff_2d_creation(simple_torchmetrics):
    input_shape = (1, 6, 60)
    # with metrics
    model = CNN_PFF_2D(input_shape=input_shape, pad_at=3, **simple_torchmetrics)