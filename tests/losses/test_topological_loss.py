import torch
from minerva.losses.topological_loss import TopologicalLoss


def test_topological_loss():
    topological_loss = TopologicalLoss()
    x = torch.rand(10, 3, 32, 32)
    x_encoded = torch.rand(10, 256)
    loss = topological_loss(x, x_encoded)
    assert loss is not None
    assert loss.item() is not None
