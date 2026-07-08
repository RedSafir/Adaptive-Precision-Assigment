import torch
import torch.nn as nn
from .quant_function import *
import numpy as np

#----- 
# __all__ = ["Quantizer"]
__all__ = ["Quantizer",
           "QuantizerCustom"]
#-----


class Quantizer(nn.Module):
    def __init__(
        self,
        forward_number=None,
        backward_number=None,
        forward_rounding="stochastic",
        backward_rounding="stochastic",
    ):
        super(Quantizer, self).__init__()
        self.quantize = quantizer(
            forward_number, backward_number, forward_rounding, backward_rounding
        )

    def forward(self, x):
        return self.quantize(x)

#----- 
class QuantizerCustom(nn.Module):
    def __init__(self, fwd_num):
        super(QuantizerCustom, self).__init__()
        self.quantize = quantizer_custom(fwd_num)

    def forward(self, x, exp_bias_pow, allow_inf):
        return self.quantize(x, exp_bias_pow, allow_inf)
#-----
