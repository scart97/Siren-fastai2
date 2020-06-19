import torch
from torch import nn
import torch.nn.functional as F
from typing import List
import math

def siren_init(tensor, use_this_fan_in=None):
    if use_this_fan_in is not None:
        fan_in = use_this_fan_in
    else:
        fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
    bound = math.sqrt(6.0 / fan_in)
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


class SirenLayer(nn.Module):
    """
        Siren Layer - it's a modified linear layer with sine activation
        https://arxiv.org/abs/2006.09661
    """
    def __init__(self, in_features, out_features, bias=True, w0 = 1):
        """
            w0 comes from the end of section 3, it should be 30 for the first layer
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.w0 = torch.tensor(w0)
        self.reset_parameters()

    def reset_parameters(self):
        siren_init(self.weight)
        if self.bias is not None:
            fan_in = nn.init._calculate_correct_fan(self.weight, 'fan_in')
            siren_init(self.bias, use_this_fan_in=fan_in)

    def forward(self, x):
        x = F.linear(x, self.w0 * self.weight, self.bias)
        return torch.sin(x)
    
    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, w0={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.w0
        )


def siren_model(dimensions:List[int]):
    first_layer = SirenLayer(dimensions[0], dimensions[1], w0=30)
    other_layers = []
    for dim0, dim1 in zip(dimensions[1:-1], dimensions[2:]):
        other_layers.append(SirenLayer(dim0, dim1))
    return nn.Sequential(first_layer, *other_layers)        
