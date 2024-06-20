import torch
from minerva.models.nets.convtae_encoders import ConvTAEEncoder, ConvTAEDecoder 

def test_convtae_encoder_forward():
    model = ConvTAEEncoder(in_channels=6, time_steps=60, encoding_size=256, fc_num_layers=3, conv_num_layers=3, conv_mid_channels=12, conv_kernel=5, conv_padding=0)
    assert model is not None

    x = torch.rand(10, 6, 60)
    y = model(x)
    assert y is not None


def test_cnn_ha_pff_forward():
    model = ConvTAEDecoder(target_channels=6, target_time_steps=60, encoding_size=256, fc_num_layers=3, conv_num_layers=3, conv_mid_channels=12, conv_kernel=5, conv_padding=0)
    assert model is not None

    x = torch.rand(10, 256)
    y = model(x)
    assert y is not None