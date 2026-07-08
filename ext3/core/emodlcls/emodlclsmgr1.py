from ext3.typing       import *
from ext3.core.include import EModl

import torch

__all__ = [ 'EModlClsMgr1' ]

#==============#
# EModlClsMgr1 #
#==============#
class EModlClsMgr1():
    #-----#
    # gen #
    #-----#
    @staticmethod
    def gen(ModlClsBase: Type[torch.nn.Module]) -> Type[EModl]:
        """ wrap a torch modl class whose bwd func does NOT use the output of its fwd func. """

        # set: ModlClsNew from scratch.
        class ModlClsNew(ModlClsBase, EModl): # type: ignore
            def __init__(self, *args, **kwargs) -> None:
                super(ModlClsNew, self).__init__(*args, **kwargs)

        # set: class info of ModlClsNew.
        _set_cls_info(ModlClsNew, ModlClsBase)
        return ModlClsNew

#========#
# helper #
#========#
def _set_cls_info(ModlClsNew: Type[EModl], ModlClsBase: Type[torch.nn.Module]) -> None:
    """ set class info of ModlClsNew. """

    # # set: class vars.
    # ModlClsNew.info_basecls = ModlClsBase

    # set: class name. (https://stackoverflow.com/questions/5352781/how-to-set-class-names-dynamically#comment118557822_5353609)
    ModlClsNew.__qualname__ = f'E_{ModlClsBase.__qualname__}'
    ModlClsNew.__name__     = f'E_{ModlClsBase.__name__}'
