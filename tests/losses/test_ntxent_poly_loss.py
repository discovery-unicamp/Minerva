import torch
from minerva.losses.ntxent_loss_poly import NTXentLoss_poly


def test_ntxentpoly_loss():
    device = torch.device('cpu')
    ntxentpoly_loss = NTXentLoss_poly(device = device, batch_size = 10, temperature = 0.2, use_cosine_similarity = True)
    x = torch.rand(10, 32)
    loss = ntxentpoly_loss(x, x)
    assert loss is not None
    assert loss.item() is not None