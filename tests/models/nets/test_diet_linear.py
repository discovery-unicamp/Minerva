from minerva.models.nets.diet_linear import DIETLinear
from minerva.models.nets.adapted_head import AdaptedHead
import torch

def test_diet_linear():
    model = DIETLinear(in_features=256, out_features=100)
    assert model is not None

    x = torch.rand(32, 256)
    y = model(x)
    assert y is not None

def test_adapted_diet_linear():
    model = DIETLinear(in_features=256, out_features=100)
    assert model is not None
    adapted_model = AdaptedHead(
        model=model,
        adapter=lambda x : x.reshape(32,-1)
    )
    assert adapted_model is not None

    x = torch.rand(32, 128, 2)
    y = adapted_model(x)
    assert y is not None