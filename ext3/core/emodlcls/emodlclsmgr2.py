from ext3.typing       import *
from ext3.core.include import EModl, _hook_fwd_main, _hook_bwd_main
from .emodlclsmgr1     import EModlClsMgr1, _set_cls_info

import torch, collections

__all__ = [ 'EModlClsMgr2', '_set_cls_info' ]

#==============#
# EModlClsMgr2 #
#==============#
class EModlClsMgr2():
    #-----#
    # gen #
    #-----#
    @staticmethod
    def gen(ModlClsBase: Type[torch.nn.Module], FuncClsNew: 'Type[EModlClsMgr2.Function]') -> Type[EModl]:
        """ wrap a torch modl class whose bwd func DOES use the output of its fwd func. """

        # set: ModlClsNew using FuncClsNew.
        ModlClsNewPre: Any = EModlClsMgr1.gen(ModlClsBase)
        class ModlClsNew(ModlClsNewPre):
            def __init__(self, *args, **kwargs):
                super(ModlClsNew, self).__init__(*args, **kwargs)
                # remove {fwd,bwd}_hook(). contain them directly in forward().
                self._forward_hooks  = collections.OrderedDict()
                self._backward_hooks = collections.OrderedDict()

            def forward(self, x: TS) -> TS:
                # use forward() (and its implicit backward()) that contain {fwd,bwd}_hook().
                return FuncClsNew.apply(x, self)

        # set: class info of ModlClsNew. [see EModlClsMgr1.gen().]
        _set_cls_info(ModlClsNew, ModlClsBase)
        return ModlClsNew

    #----------#
    # Function #
    #----------#
    #
    # SPEC of EModlClsMgr2.Function.
    #
    # - a function with one input x and one output y. bwd uses gy and y only.
    # - {forward, backward}_user: needs to be implemented.
    #
    class Function(torch.autograd.Function):
        #------#
        # main #
        #------#
        @staticmethod
        def forward_user(x: TS, emodl: EModl) -> TS:
            raise NotImplemented

        @staticmethod
        def backward_user(gy: TS, y: TS, emodl: EModl) -> TS:
            raise NotImplemented

        #-------#
        # hooks # (for X, Y, GX, GY)
        #-------#
        # contain fwd_hook() directly in forward().
        @classmethod
        def forward(cls, ctx, x: TS, emodl: EModl) -> TS: # type: ignore
            """ compute y, round y, and set info_* of fwd tensors. """

            # calc: y.
            y = cls.forward_user(x, emodl)

            # do: forward hook. [see EModl._hook_fwd().]
            xs: Seq[TS]; ys: Seq[TS]; ps: Seq[TS]
            xs, ys, ps = [x], [y], []
            ys = _hook_fwd_main(xs, ys, ps, emodl)
            y = ys[0]

            # save: ctx.
            ctx.save_for_backward(y)
            ctx.fwd_args = (emodl,)

            # ret.
            return y
        
        # contain bwd_hook() directly in backward().
        @classmethod
        def backward(cls, ctx, gy: TS) -> Seq[Opt[TS]]: # type: ignore
            """ compute gxs, round gxs, and set info_* of bwd tensors. """

            # load: ctx.
            y: TS; emodl: EModl
            (y,) = ctx.saved_tensors
            (emodl,) = ctx.fwd_args

            # calc: gx.
            gx = cls.backward_user(gy, y, emodl)

            # do: backward hook. [see EModl._hook_bwd().]
            gxs: Seq[Opt[TS]]; gys: Seq[Opt[TS]]; ps: Seq[TS]
            gxs, gys, ps = [gx], [gy], []
            gxs = _hook_bwd_main(gxs, gys, ps, emodl)

            # ret.
            return (gxs[0], None) #= (gx, gmodl)
