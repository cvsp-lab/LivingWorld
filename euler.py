import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.linalg import inv
import numpy as np

class EulerMotion_MLP(nn.Module):
    '''
    Args:
        input:
            [num_vertices, 3]
        output:
            [num_vertices, 3]
        Misc:
            add position encoding?
    '''
    def __init__(self):
        super(EulerMotion_MLP, self).__init__()
        self.pe = PositionalEncoder(d_input=3, n_freqs=4)
        self.input_dim = 27  # add time dimension (2 * n_freqs +1) * 3
        self.hidden_dims = [128,64]
        self.output_dim = 3
        self.hidden_layers = nn.ModuleList([nn.Linear(self.input_dim, self.hidden_dims[0])])
        for i in range(1, len(self.hidden_dims)):
            self.hidden_layers.append(nn.Linear(self.hidden_dims[i-1], self.hidden_dims[i]))
        self.output_layer = nn.Linear(self.hidden_dims[-1], self.output_dim)
        self.activate_function = nn.ReLU()


    def forward(self, point):
        point = self.pe(point)
        for layer in self.hidden_layers:
            point = self.activate_function(layer(point))
        xyz_ = self.output_layer(point)
        return xyz_
    
class PositionalEncoder(nn.Module):
    def __init__(self, d_input, n_freqs,log_space = False ):
        super(PositionalEncoder, self).__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.log_space = log_space
        self.d_output = d_input * (1 + 2 * self.n_freqs)

        # Define frequencies in either linear or log scale
        if self.log_space:
            freq_bands = 2.**torch.linspace(0., self.n_freqs - 1, self.n_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**(self.n_freqs - 1), self.n_freqs)

        self.freq = freq_bands
        

    def forward(self, x):
        res = [x]
        for freq in self.freq:
            res.append(self.sin_fn(x,freq))
            res.append(self.cos_fn(x,freq))
        return torch.cat(res, dim=-1)
    
    def sin_fn(self,x, freq):
        return torch.sin(x * freq)

    def cos_fn(self,x, freq):
        return torch.cos(x * freq)
