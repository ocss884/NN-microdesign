import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F


class NN(nn.Module):
    """
    Neural network class.
    Args:
        volume_fraction (np.ndarray): Volume fraction of material.
        num_modes (int): Number of possible deformation modes.
    """
    def __init__(self, volume_fraction:np.ndarray, num_modes:int = 31):
        super().__init__()
        self.dim_list = [76, 760, 76]
        self.num_layers = len(self.dim_list)
        self.weights = nn.ParameterList([nn.Parameter(torch.empty(num_modes, self.dim_list[i], self.dim_list[i+1])) for i in range(self.num_layers-1)])
        self.bias = nn.ParameterList([nn.Parameter(torch.empty(num_modes, 1, self.dim_list[i+1])) for i in range(self.num_layers-1)])
        self.activations = [nn.Tanh()]*(self.num_layers-2) + [nn.ReLU()]
        
        self.init_weights()
        self.volume_fraction = torch.Tensor(volume_fraction)
        self.register_buffer("vol", self.volume_fraction)
        
    def init_weights(self):
        for i in range(self.num_layers-1):
            nn.init.xavier_normal_(self.weights[i])
            self.bias[i].data.zero_()
        
    def forward(self, x: torch.Tensor):
        y = x[:, None, None, :]
        for i in range(self.num_layers-1):
            y = y@self.weights[i]+self.bias[i]
            y = self.activations[i](y)
        pred_y = (y.squeeze())/(y@self.vol)

        return pred_y