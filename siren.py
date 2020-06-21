import torch
from torch import nn
import torch.nn.functional as F
from typing import List
import math


def siren_init(tensor, use_this_fan_in=None):
    if use_this_fan_in is not None:
        fan_in = use_this_fan_in
    else:
        fan_in = nn.init._calculate_correct_fan(tensor, "fan_in")
    bound = math.sqrt(6.0 / fan_in)
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


def apply_siren_init(layer: nn.Module):
    siren_init(layer.weight)
    if layer.bias is not None:
        fan_in = nn.init._calculate_correct_fan(layer.weight, "fan_in")
        siren_init(layer.bias, use_this_fan_in=fan_in)


class Siren(nn.Module):
    """
        Siren activation
        https://arxiv.org/abs/2006.09661
    """

    def __init__(self, w0=1):
        """
            w0 comes from the end of section 3
            it should be 30 for the first layer
            and 1 for the rest
        """
        super().__init__()
        self.w0 = torch.tensor(w0)

    def forward(self, x):
        return torch.sin(self.w0 * x)

    def extra_repr(self):
        return "w0={}".format(self.w0)


def siren_layer(in_features, out_features, bias=True, w0=1):
    """
        Siren Layer - it's a modified linear layer with sine activation
    """
    layer = nn.Sequential(nn.Linear(in_features, out_features, bias), Siren(w0))
    apply_siren_init(layer[0])
    return layer


def siren_model(dimensions: List[int]):
    first_layer = siren_layer(dimensions[0], dimensions[1], w0=30)
    other_layers = []
    for dim0, dim1 in zip(dimensions[1:-1], dimensions[2:]):
        other_layers.append(siren_layer(dim0, dim1))
    return nn.Sequential(first_layer, *other_layers)
