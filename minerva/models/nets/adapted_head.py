import torch
from typing import Callable


class AdaptedHead(torch.nn.Module):
    def __init__(self, model:torch.nn.Module, adapter:Callable):
        super().__init__()
        self.model = model
        self.adapter = adapter
        
    def forward(self, x):
        x = self.adapter(x)
        return self.model.forward(x)