from minerva.models.ssl.lfr import LearnFromRandomnessModel
from minerva.models.nets.lfr_har_architectures import (
    HARSCnnEncoder,
    LFR_HAR_Projector,
    LFR_HAR_Predictor,
)
import torch
import pytest


def test_lfr_har():
    model = LearnFromRandomnessModel(
        backbone=HARSCnnEncoder(
            dim=256, input_channel=6, inner_conv_output_dim=128 * 10
        ),
        projectors=torch.nn.ModuleList(
            [LFR_HAR_Projector(encoding_size=256, input_channel=6) for _ in range(8)]
        ),
        predictors=torch.nn.ModuleList(
            [
                LFR_HAR_Predictor(encoding_size=256, middle_dim=128, num_layers=3)
                for _ in range(8)
            ]
        ),
        loss_fn=None,
        learning_rate=1e-3,
        flatten=False,
        predictor_training_epochs=7,
    )
    assert model is not None

    x = torch.rand(32, 6, 60)
    y = model(x)
    assert y is not None


def test_lfr_har_adapter():
    example_adapter = torch.nn.Linear(300, 256)
    model = LearnFromRandomnessModel(
        backbone=HARSCnnEncoder(
            dim=300, input_channel=6, inner_conv_output_dim=128 * 10
        ),
        projectors=torch.nn.ModuleList(
            [LFR_HAR_Projector(encoding_size=256, input_channel=6) for _ in range(8)]
        ),
        predictors=torch.nn.ModuleList(
            [
                LFR_HAR_Predictor(encoding_size=256, middle_dim=128, num_layers=3)
                for _ in range(8)
            ]
        ),
        loss_fn=None,
        learning_rate=1e-3,
        flatten=False,
        predictor_training_epochs=7,
        adapter=example_adapter,
    )
    assert model is not None

    x = torch.rand(32, 6, 60)
    y = model(x)
    assert y is not None


def test_lfr_erroneous_adapter_input():
    example_adapter = torch.nn.Linear(301, 256)
    model = LearnFromRandomnessModel(
        backbone=HARSCnnEncoder(
            dim=300, input_channel=6, inner_conv_output_dim=128 * 10
        ),
        projectors=torch.nn.ModuleList(
            [LFR_HAR_Projector(encoding_size=256, input_channel=6) for _ in range(8)]
        ),
        predictors=torch.nn.ModuleList(
            [
                LFR_HAR_Predictor(encoding_size=256, middle_dim=128, num_layers=3)
                for _ in range(8)
            ]
        ),
        loss_fn=None,
        learning_rate=1e-3,
        flatten=False,
        predictor_training_epochs=7,
        adapter=example_adapter,
    )
    assert model is not None

    x = torch.rand(32, 6, 60)
    with pytest.raises(RuntimeError):
        y = model(x)


def test_lfr_erroneous_adapter_output():
    example_adapter = torch.nn.Linear(300, 255)
    model = LearnFromRandomnessModel(
        backbone=HARSCnnEncoder(
            dim=300, input_channel=6, inner_conv_output_dim=128 * 10
        ),
        projectors=torch.nn.ModuleList(
            [LFR_HAR_Projector(encoding_size=256, input_channel=6) for _ in range(8)]
        ),
        predictors=torch.nn.ModuleList(
            [
                LFR_HAR_Predictor(encoding_size=256, middle_dim=128, num_layers=3)
                for _ in range(8)
            ]
        ),
        loss_fn=None,
        learning_rate=1e-3,
        flatten=False,
        predictor_training_epochs=7,
        adapter=example_adapter,
    )
    assert model is not None

    x = torch.rand(32, 6, 60)
    with pytest.raises(RuntimeError):
        y = model(x)
