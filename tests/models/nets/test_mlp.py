import pytest
import torch
import torch.nn as nn

from minerva.models.nets.mlp import MLP


def test_mlp_basic_forward():
    mlp = MLP([10, 20, 30])
    x = torch.randn(5, 10)
    out = mlp(x)
    assert out.shape == (5, 30)


def test_mlp_with_full_batchnorm():
    mlp = MLP(
        layer_sizes=[16, 32, 64],
        activation_cls=nn.ReLU,
        intermediate_ops=[nn.BatchNorm1d(32), nn.BatchNorm1d(64)],
    )
    x = torch.randn(8, 16)
    out = mlp(x)
    assert out.shape == (8, 64)


def test_mlp_with_partial_intermediate_ops():
    mlp = MLP(
        layer_sizes=[16, 32, 64, 128],
        activation_cls=nn.ReLU,
        intermediate_ops=[nn.BatchNorm1d(32), None, nn.Dropout(p=0.1)],
    )
    x = torch.randn(4, 16)
    out = mlp(x)
    assert out.shape == (4, 128)


def test_mlp_with_final_op():
    final = nn.Sigmoid()
    mlp = MLP(layer_sizes=[32, 64, 128], final_op=final)
    x = torch.randn(4, 32)
    out = mlp(x)
    assert out.shape == (4, 128)
    assert torch.all((0 <= out) & (out <= 1))


def test_mlp_invalid_layer_sizes():
    with pytest.raises(AssertionError, match="at least 2 layers"):
        MLP([128])  # only one layer

    with pytest.raises(AssertionError, match="positive integers"):
        MLP([128, 0, 64])  # invalid layer size


def test_mlp_invalid_activation():
    with pytest.raises(
        AssertionError, match="activation_cls must inherit from torch.nn.Module"
    ):
        MLP([128, 64], activation_cls=int)  # invalid activation


def test_mlp_invalid_intermediate_ops_length():
    with pytest.raises(ValueError, match="Length of intermediate_ops"):
        MLP(
            layer_sizes=[10, 20, 30],
            intermediate_ops=[None],  # only one op for two layers
        )


def test_repr_and_structure():
    mlp = MLP([10, 20, 30], final_op=nn.Tanh())
    model_str = str(mlp)
    assert "Linear" in model_str
    assert "Tanh" in model_str
