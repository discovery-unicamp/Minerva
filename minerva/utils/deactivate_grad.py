from torch import nn


def deactivate_requires_grad(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False
 