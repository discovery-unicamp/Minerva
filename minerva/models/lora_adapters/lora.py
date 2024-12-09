import torch.nn as nn

class LoRA(nn.Module):
    """ Base class for LoRA """
    def __init__(self):
        super(LoRA, self).__init__()
    
    def init_weights(self):
        raise NotImplementedError("init_weights must be implemented in subclasses.")
        
    def forward(self, x):
        raise NotImplementedError("forward must be implemented in subclasses.")