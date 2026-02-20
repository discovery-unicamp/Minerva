from minerva.models.ssl.lfr import LearnFromRandomnessModel
from minerva.models.nets.lfr_har_architectures import HARSCnnEncoder
import torch
from minerva.models.nets.lfr_har_architectures import (
    HARSCnnEncoder,
    LFR_HAR_Projector_List,
    LFR_HAR_Predictor_List,
)
import torch.nn.functional as F

# These tests should ensure that the LFR implementation matches the code in https://github.com/layer6ai-labs/lfr


def test_loss():
    num_targets = 3
    random_input = torch.rand((32, 6, 60))
    backbone = HARSCnnEncoder(dim=9, input_channel=6, inner_conv_output_dim=1280)
    backbone.eval()
    projectors = LFR_HAR_Projector_List(
        size=num_targets, encoding_size=9, input_channel=6, middle_dim=544
    )
    projectors.eval()
    predictors = LFR_HAR_Predictor_List(
        size=num_targets, encoding_size=9, middle_dim=128, num_layers=1
    )
    predictors.eval()
    model = LearnFromRandomnessModel(
        backbone=backbone, projectors=projectors, predictors=predictors
    )

    ### Comparing projectors and predictors

    # Original code, adapted from
    # https://github.com/layer6ai-labs/lfr/blob/f513dbfc540ef92104dcac982ac1e2c06ed49099/ssl_models/lfr.py#L60
    z_w = backbone(random_input)
    target_reps = []
    predicted_reps = []
    for i in range(num_targets):
        target = projectors[i]
        predictor = predictors[i]
        z_a = target(random_input)  # NxC
        p_a = predictor(z_w)
        target_reps.append(z_a)
        predicted_reps.append(p_a)

    # Adapted code in Minerva
    y_pred, y_proj = model.forward(random_input)
    # All values should be the same
    for target_idx in range(num_targets):
        original_target = target_reps[target_idx].flatten()
        adapted_target = y_proj.transpose(1, 0)[target_idx].flatten()
        for original_value, adapted_value in zip(original_target, adapted_target):
            assert original_value == adapted_value
        original_predictions = predicted_reps[target_idx].flatten()
        adapted_predictions = y_pred.transpose(1, 0)[target_idx].flatten()
        for original_value, adapted_value in zip(
            original_predictions, adapted_predictions
        ):
            assert original_value == adapted_value

    ### Comparing the loss calculations

    # Original code, adapted from
    # https://github.com/layer6ai-labs/lfr/blob/f513dbfc540ef92104dcac982ac1e2c06ed49099/utils/trainers.py#L60
    def off_diagonal(x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def bt_loss_bs(p, z, lambd=0.01, normalize=False):
        # barlow twins loss but in batch dims
        c = torch.matmul(F.normalize(p), F.normalize(z).T)
        assert c.min() > -1 and c.max() < 1
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + lambd * off_diag
        if normalize:
            loss = loss / p.shape[0]
        return loss

    # Computing loss - original version
    loss = torch.tensor(0)
    for t in range(num_targets):
        p = predicted_reps[t]
        z = target_reps[t]
        loss = loss + bt_loss_bs(p, z, lambd=0.01)
    loss = loss / num_targets
    # Computing loss - adapted version
    adapted_loss = model._loss_from_targets(y_pred=y_pred, y_proj=y_proj)
    assert loss.item() == adapted_loss.item()
    assert loss == adapted_loss
