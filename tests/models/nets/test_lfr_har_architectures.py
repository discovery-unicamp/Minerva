from minerva.models.nets.lfr_har_architectures import LFR_HAR_Backbone, LFR_HAR_Projector, LFR_HAR_Predictor
import torch

def test_lfr_har_backbone():
    model = LFR_HAR_Backbone(encoding_size=256, input_channel=6)
    assert model is not None

    x = torch.rand(32, 6, 60)
    y = model(x)
    assert y is not None


def test_lfr_har_projector():
    model = LFR_HAR_Projector(encoding_size=256, input_channel=6)
    assert model is not None

    x = torch.rand(32, 6, 60)
    y = model(x)
    assert y is not None


def test_lfr_har_predictor():
    model = LFR_HAR_Predictor(encoding_size=256, middle_dim=128, num_layers=3)
    assert model is not None

    x = torch.rand(32, 256)
    y = model(x)
    assert y is not None
