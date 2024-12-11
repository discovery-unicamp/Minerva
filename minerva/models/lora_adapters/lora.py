import torch
import torch.nn as nn
import math

class LoRA(nn.Module):
    """ LoRA (Low-rank Adaptation) for Linear Layers. Use this for Transformers or Linear layers. """
    def __init__(self, original_module, bias=True, alpha=1, r=4):
        super(LoRA, self).__init__()

        self.original_module = original_module
        self.matrix_A = torch.nn.Linear(original_module.in_features, r, bias=bias)
        self.matrix_B = torch.nn.Linear(r, original_module.out_features, bias=bias)
        self.scaling = alpha / r

        self.init_weights()
    
    def init_weights(self):
        torch.nn.init.kaiming_uniform_(self.matrix_A.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.matrix_B.weight)
        
    def forward(self, x):
        return self.original_module(x) + self.scaling * self.matrix_B(self.matrix_A(x))