from minerva.models.nets.lfr_har_architectures import (
    HARSCnnEncoder,
    LFR_HAR_Projector,
    LFR_HAR_Predictor,
    LFR_HAR_Projector_List,
    LFR_HAR_Predictor_List,
)
import torch


def test_lfr_har_backbone():
    model = HARSCnnEncoder(dim=256, input_channel=6, inner_conv_output_dim=128 * 10)
    assert model is not None

    x = torch.rand(32, 6, 60)
    y = model(x)
    assert y is not None


def test_lfr_har_projector():
    model = LFR_HAR_Projector(encoding_size=256, input_channel=6, middle_dim=544)
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


def test_lfr_har_projector_list():
    projector_list = LFR_HAR_Projector_List(
        size=6, encoding_size=256, input_channel=6, middle_dim=544
    )
    assert projector_list is not None

    x = torch.rand(32, 6, 60)
    for projector in projector_list:
        y = projector(x)
        assert y is not None


def test_lfr_har_predictor_list():
    predictor_list = LFR_HAR_Predictor_List(
        size=6, encoding_size=256, middle_dim=128, num_layers=3
    )
    assert predictor_list is not None

    x = torch.rand(32, 256)
    for projector in predictor_list:
        y = projector(x)
        assert y is not None
