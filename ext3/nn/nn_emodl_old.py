from ext3.typing import *
from ext3.core   import EModl, EFuncMgr, EModlClsMgr1, EModlClsMgr2
from .nn_base    import _Add, _Cat, _Identity, _Mean, _Mul, _SplitHalf, _LogSoftmaxFunc, _SigmoidFunc

import torch

__all__ = [
    'flatten', 'permute', 'reshape', 'view', # 'getitem',
    'Add', 'Cat', 'Input', 'Mean', 'Mul', 'SplitHalf', # 'Output',
    'AdaptiveAvgPool2d', 'AvgPool2d', 'BatchNorm2d', 'Conv2d',
    'Linear', 'MaxPool2d', 'NLLLoss', 'ReLU',
    'LogSoftmax', 'Sigmoid',
    'Module', 'Sequential', 'CrossEntropyLoss',
]

# func(op with 1 input, 1 output, no fp op): wrap with efunc mgr.
# getitem: Callable = EFuncMgr.gen(torch.Tensor.__getitem__) # shfv2.
flatten: Callable = EFuncMgr.gen(torch.flatten)            # sqz.
permute: Callable = EFuncMgr.gen(torch.Tensor.permute)     # shfv2.
reshape: Callable = EFuncMgr.gen(torch.Tensor.reshape)     # shfv2.
view   : Callable = EFuncMgr.gen(torch.Tensor.view)        # shfv2, mblv2, resnet.

# modl(atomic op, new): wrap with emodl mgr1.
# Output           : Type[EModl] = EModlClsMgr1.gen(_Identity)
Add              : Type[EModl] = EModlClsMgr1.gen(_Add)           # shfv2, mblv2.
Cat              : Type[EModl] = EModlClsMgr1.gen(_Cat)           # sqz, shfv2.
Input            : Type[EModl] = EModlClsMgr1.gen(_Identity)      # ALL.
Mean             : Type[EModl] = EModlClsMgr1.gen(_Mean)          # shfv2 (imgnet).
Mul              : Type[EModl] = EModlClsMgr1.gen(_Mul)           # regnet.
SplitHalf        : Type[EModl] = EModlClsMgr1.gen(_SplitHalf)     # shfv2.

# modl(atomic op, bwd does not use y): wrap with emodl mgr1.
AdaptiveAvgPool2d: Type[EModl] = EModlClsMgr1.gen(torch.nn.AdaptiveAvgPool2d)
AvgPool2d        : Type[EModl] = EModlClsMgr1.gen(torch.nn.AvgPool2d)
BatchNorm2d      : Type[EModl] = EModlClsMgr1.gen(torch.nn.BatchNorm2d)
Conv2d           : Type[EModl] = EModlClsMgr1.gen(torch.nn.Conv2d)
Linear           : Type[EModl] = EModlClsMgr1.gen(torch.nn.Linear)
MaxPool2d        : Type[EModl] = EModlClsMgr1.gen(torch.nn.MaxPool2d)
NLLLoss          : Type[EModl] = EModlClsMgr1.gen(torch.nn.NLLLoss)
ReLU             : Type[EModl] = EModlClsMgr1.gen(torch.nn.ReLU)

# modl(atomic op, bwd does use y): wrap with emodl mgr2.
LogSoftmax       : Type[EModl] = EModlClsMgr2.gen(torch.nn.LogSoftmax, _LogSoftmaxFunc)
Sigmoid          : Type[EModl] = EModlClsMgr2.gen(torch.nn.Sigmoid,    _SigmoidFunc)

# modl(composite op): do not wrap.
Module     = torch.nn.Module
Sequential = torch.nn.Sequential
class CrossEntropyLoss(torch.nn.CrossEntropyLoss):
    def __init__(self, *args, **kwargs):
        super(CrossEntropyLoss, self).__init__(*args, **kwargs)
        self.logsoftmax = LogSoftmax(dim=-1)       # use wrapped op.
        self.nllloss    = NLLLoss(*args, **kwargs) # use wrapped op.

    def forward(self, input, target):
        assert(input.dim() == 2 and target.dim() == 1)
        y = self.logsoftmax(input)
        y = self.nllloss(y, target)
        return y
