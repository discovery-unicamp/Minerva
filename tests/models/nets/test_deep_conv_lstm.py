import torch
from minerva.models.nets.deep_conv_lstm import DeepConvLSTM

def test_deep_conv_lstm_forward():
    input_shape = (1, 6, 60)
    model = DeepConvLSTM(input_shape=input_shape)
    assert model is not None

    x = torch.rand(1, *input_shape)
    y = model(x)
    assert y is not None

