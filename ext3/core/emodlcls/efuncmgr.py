from ext3.typing       import *
from ext3.core.include import TsDictMgr

__all__ = [ 'EFuncMgr' ]

#==========#
# EFuncMgr #
#==========#
class EFuncMgr():
    #-----#
    # gen #
    #-----#
    @classmethod
    def gen(cls, func: Callable) -> Callable:
        """ wrap a torch func that takes one input and one output, and does not perform any fp op. """

        # set: new_func.
        def new_func(x: TS, *args, **kwargs) -> TS:
            # set: y=func(x). no rnd.
            y = func(x, *args, **kwargs)

            # set: TsDictMgr.
            TsDictMgr.update(y.data_ptr(), TsDictMgr.get(x.data_ptr()))
            return y

        # ret.
        return new_func
