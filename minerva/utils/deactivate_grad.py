from torch import nn
import torch

@torch.no_grad()
def deactivate_requires_grad(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False
 