from minerva.models.nets.diet_linear import DIETLinear
import torch

def test_diet_linear():
    model = DIETLinear(in_features==256, out_features==100)
    assert model is not None

    x = torch.rand(32, 256, 100)
    y = model(x)
    assert y is not None