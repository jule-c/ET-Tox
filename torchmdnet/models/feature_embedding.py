import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch_scatter import scatter
from .shifted_softplus import ShiftedSoftplus
from .swish import Swish
from typing import Optional
from typing import Dict, Union


class FeatureEmbedding(nn.Module):
    def __init__(
            self,
            num_features: int,
            out_features: int,
            activation: str = "swish",
            bias: bool = True,
            zero_init: bool = True,
            max_z: int = 100,
    ):

        super(FeatureEmbedding, self).__init__()
        # initialize attributes
        if activation == "ssp":
            Activation = ShiftedSoftplus
        elif activation == "swish":
            Activation = Swish
        else:
            raise ValueError(
                "Argument 'activation' may only take the "
                "values 'ssp', or 'swish' but received '" + str(activation) + "'."
            )

        self.num_features = num_features
        self.out_features = out_features
        self.bias = bias
        if not self.bias:
            self.element_bias = nn.Embedding(max_z, out_features)
        self.init_linear1 = nn.Linear(num_features, out_features, bias=True)
        self.init_linear2 = nn.Linear(num_features, out_features, bias=True)
        
        self.linear1 = nn.Linear(out_features, out_features, bias=bias)
        self.activation1 = Activation(out_features)
        self.linear2 = nn.Linear(out_features, out_features, bias=bias)
        self.activation2 = Activation(out_features)
        
        self.reset_parameters(bias, zero_init)

    def reset_parameters(self, bias: bool = True, zero_init: bool = True) -> None:
        nn.init.orthogonal_(self.linear1.weight)
        if zero_init:
            nn.init.zeros_(self.linear2.weight)
        else:
            nn.init.orthogonal_(self.linear2.weight)
        if bias:
            nn.init.zeros_(self.linear1.bias)
            nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        y = self.init_linear1(x)
        x = self.init_linear2(x)  

        #ele_b = torch.zeros(x.shape[0], 1).to(x.device)
        if not self.bias:
            ele_b = self.element_bias(z)
        x = self.linear1(x) + ele_b
        x = self.activation1(x) + y
        x = self.linear2(x) + ele_b
        x = self.activation2(x) + y
        return x

class Residual(nn.Module):
    def __init__(
        self,
        num_features: int,
        out_features: int,
        activation: str = "swish",
        bias: bool = True,
        zero_init: bool = True,
        max_z: int = 100,
    ):

        super(Residual, self).__init__()
        # initialize attributes
        if activation == "ssp":
            Activation = ShiftedSoftplus
        elif activation == "swish":
            Activation = Swish
        else:
            raise ValueError(
                "Argument 'activation' may only take the "
                "values 'ssp', or 'swish' but received '" + str(activation) + "'."
            )
        
        self.num_features = num_features
        self.out_features = out_features
        self.bias = bias
        if not self.bias:
            self.element_bias = nn.Embedding(max_z, 1)
        self.activation1 = Activation(num_features)
        self.linear1 = nn.Linear(num_features, num_features, bias=bias)
        self.activation2 = Activation(num_features)
        self.linear2 = nn.Linear(num_features, out_features, bias=bias)
        self.reset_parameters(bias, zero_init)

    def reset_parameters(self, bias: bool = True, zero_init: bool = True) -> None:
        nn.init.orthogonal_(self.linear1.weight)
        if zero_init:
            nn.init.zeros_(self.linear2.weight)
        else:
            nn.init.orthogonal_(self.linear2.weight)
        if bias:
            nn.init.zeros_(self.linear1.bias)
            nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        ele_b = 0
        if not self.bias:
            ele_b = self.element_bias(z)
        y = self.activation1(x)
        y = self.linear1(y) + ele_b
        y = self.activation2(y)
        if self.out_features != self.num_features:
            y = x + y
            y = self.linear2(y) + ele_b
            return y
        else:
            y = self.linear2(y) + ele_b
            return x + y
