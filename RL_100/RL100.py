import torch
from torch.nn import Module

# functions

def exists(v):
    return v is not None

# class

class RL100(Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError

    def forward(self, state):
        raise NotImplementedError
