import torch
import lightning as L
from typing import Dict, Optional, Union, Callable


class DIETLinear(torch.nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            adapter: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
        ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.adapter = adapter
        self.fc = torch.nn.Linear(in_features, out_features)
        
    def forward(self, x):
        if self.adapter is not None:
            x = self.adapter(x)
        x = self.fc(x)
        return x