from minerva.models.ssl.autoencoder import Autoencoder
from minerva.models.nets.conv_autoencoders_encoders import (
    ConvTAEEncoder,
    ConvTAEDecoder,
)
import torch


def test_autoencoder_forward():
    encoder = ConvTAEEncoder(
        in_channels=6,
        time_steps=60,
        encoding_size=256,
        fc_num_layers=3,
        conv_num_layers=3,
        conv_mid_channels=12,
        conv_kernel=5,
        conv_padding=0,
    )
    decoder = ConvTAEDecoder(
        target_channels=6,
        target_time_steps=60,
        encoding_size=256,
        fc_num_layers=3,
        conv_num_layers=3,
        conv_mid_channels=12,
        conv_kernel=5,
        conv_padding=1,
    )
    model = Autoencoder(encoder, decoder)

    x = torch.rand(32, 6, 60)
    y = model(x)
    assert y is not None
    assert y.shape == x.shape
