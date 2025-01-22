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
    

class DIETLinear(torch.nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int
        ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = torch.nn.Linear(in_features, out_features)
        
    def forward(self, x):
        x = self.fc(x)
        return x