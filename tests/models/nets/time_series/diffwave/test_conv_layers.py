import torch

from minerva.models.nets.time_series.diffwave.conv_layers import Conv, ZeroConv1d


def test_diffwave_conv_preserves_length():
    model = Conv(2, 4, kernel_size=3, dilation=2)
    x = torch.rand(2, 2, 16)

    output = model(x)

    assert output.shape == (2, 4, 16)


def test_diffwave_zero_conv_returns_zeros():
    model = ZeroConv1d(4, 2)
    x = torch.rand(2, 4, 16)

    output = model(x)

    torch.testing.assert_close(output, torch.zeros(2, 2, 16))


def test_diffwave_zero_conv_can_learn_from_zero_weights():
    model = ZeroConv1d(4, 2)
    x = torch.ones(2, 4, 16)
    loss = (model(x) - 1).square().mean()

    loss.backward()

    assert model.conv.weight.grad.abs().sum() > 0
