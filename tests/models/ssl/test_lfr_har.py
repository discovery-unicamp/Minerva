from minerva.models.ssl.lfr_har import LearnFromRandomnessModel
from minerva.models.nets.lfr_har_architectures import LFR_HAR_Backbone, LFR_HAR_Projector, LFR_HAR_Predictor
import torch



def test_lfr_har():
    model = LearnFromRandomnessModel(
        backbone=LFR_HAR_Backbone(encoding_size=256, input_channel=6),
        projectors=torch.nn.ModuleList([LFR_HAR_Projector(encoding_size=256, input_channel=6) for _ in range(8)]),
        predictors=torch.nn.ModuleList([LFR_HAR_Predictor(encoding_size=256, middle_dim=128, num_layers=3) for _ in range(8)]),
        loss_fn=None,
        learning_rate=1e-3,
        flatten=True,
        predictor_training_epochs=7
    )
    assert model is not None

    x = torch.rand(32, 6, 60)
    y = model(x)
    assert y is not None
