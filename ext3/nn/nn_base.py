from ext3.typing import *
from ext3.core   import EModl, EModlClsMgr2

import torch
import torch.nn.functional as torch_F
import numpy as np

__all__: list = []

#======#
# modl #
#======#
class _Add(torch.nn.Module):
    def forward(self, x1, x2):
        return x1 + x2

class _Cat(torch.nn.Module):
    dim: int
    def __init__(self, dim: int):
        super(_Cat, self).__init__()
        self.dim = dim
    def forward(self, *tensors):
        return torch.cat(tensors, dim=self.dim)

class _Identity(torch.nn.Module):
    def forward(self, x):
        return x

class _Mean(torch.nn.Module):
    dim: List[int]
    def __init__(self, dim: List[int]):
        super(_Mean, self).__init__()
        self.dim = dim
    def forward(self, x):
        return torch.mean(x, dim=self.dim)

class _Mul(torch.nn.Module):
    def forward(self, x1, x2):
        return x1 * x2

class _SplitHalf(torch.nn.Module):
    dim  : int
    index: int
    def __init__(self, dim: int, index: int):
        super(_SplitHalf, self).__init__()
        # assert: only first or second half.
        assert(index in (0,1))
        self.dim   = dim
        self.index = index
    def forward(self, x):
        # return x[:, ..., :, :c, :, ..., :] if self.index=0,
        #        x[:, ..., :, c:, :, ..., :] if self.index=1,
        # where c = int(x.size(self.dim) * 0.5) and :c (or c:) is at position dim.

        # set: slices.
        slices = [slice(None, None, None) for _ in range(x.dim())]
        c = int(x.shape[self.dim] * 0.5)
        if self.index == 0: slices[self.dim] = slice(None, c, None)
        else              : slices[self.dim] = slice(c, None, None)
        
        # set: res.
        res = torch.Tensor.__getitem__(x, slices)
        if self.index == 0: return res.clone() # to make res.data_ptr() different from x.data_ptr().
        else              : return res

#======#
# func #
#======#
class _LogSoftmaxFunc(EModlClsMgr2.Function):
    @staticmethod
    def forward_user(x: TS, emodl: EModl) -> TS:
        # y = logsoftmax(x, dim=dim).
        # require: emodl has type torch.nn.LogSoftmax.
        dim: int = emodl.dim # type: ignore
        y = torch_F.log_softmax(x, dim=dim)
        return y

    @staticmethod
    def backward_user(gy: TS, y: TS, emodl: EModl) -> TS:
        # gx_i = gy_i - exp(y_i) * (\sum_j gy_j).
        # require: emodl has type torch.nn.LogSoftmax.
        dim: int = emodl.dim # type: ignore
        gy_sum = gy.sum(dim=dim).unsqueeze(dim=dim)
        gx = gy - y.exp() * gy_sum
        return gx

class _SigmoidFunc(EModlClsMgr2.Function):
    @staticmethod
    def forward_user(x: TS, emodl: EModl) -> TS:
        # y = sigmoid(x).
        y = torch.sigmoid(x)
        return y

    @staticmethod
    def backward_user(gy: TS, y: TS, emodl: EModl) -> TS:
        # gx_i = gy_i * (1-y_i) * y_i.
        gx = gy * (1-y) * y
        return gx
