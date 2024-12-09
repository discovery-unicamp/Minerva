from minerva.models.nets.time_series.gans import (
    GAN,
    TTSGAN_Encoder,
    TTSGAN_Discriminator,
    TTSGAN_Generator,
)
import torch


def test_ttsgan_discriminator_forward():
    input_shape = (6, 1, 60)

    model = TTSGAN_Discriminator(seq_len=60, channels=6)

    assert model is not None

    x = torch.rand(1, *input_shape)
    y = model(x)
    assert y is not None

def test_ttsgan_generator_forward():
    input_shape = (1, 100)

    model = TTSGAN_Generator(seq_len=60, channels = 6)

    assert model is not None
    
    x = torch.rand(1, *input_shape)
    y = model(x)
    
    assert y is not None