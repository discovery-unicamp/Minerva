import torch
import lightning as L

class DIETLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.fc = torch.nn.Linear(in_features, out_features)
        
    def forward(self, x):
        x = self.fc(x)
        return x