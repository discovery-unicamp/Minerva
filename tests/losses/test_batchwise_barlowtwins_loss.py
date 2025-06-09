import torch
from minerva.losses.batchwise_barlowtwins_loss import BatchWiseBarlowTwinLoss
from minerva.losses.batchwise_barlowtwins_loss import BarlowTwinsLoss


def test_batchwise_barlowtwins_loss():
    batchwise_barlowtwins_loss = BatchWiseBarlowTwinLoss()
    y_predicted = torch.rand(10, 256)
    y_projected = torch.rand(10, 256)
    loss = batchwise_barlowtwins_loss(y_predicted, y_projected)
    assert loss is not None
    assert loss.item() is not None


def test_barlowtwins_loss():
    barlowtwins_loss = BarlowTwinsLoss()
    y_predicted = torch.rand(10, 256)
    y_projected = torch.rand(10, 256)
    loss = barlowtwins_loss(y_predicted, y_projected)
    assert loss is not None
    assert loss.item() is not None
