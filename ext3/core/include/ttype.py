from ext3.typing import *

import enum

__all__ = [ 'Ttype' ]

#=======#
# Ttype # <--- type of tensor (in a modl) = input | param | output | grad of input | ...
#=======#
class Ttype(enum.Enum):
    X  = 0 # input.
    P  = 1 # param.
    Y  = 2 # output.
    GX = 3 # grad of input.
    GP = 4 # grad of param.
    GY = 5 # grad of output.

    @classmethod
    def get_all(cls) -> Seq['Ttype']:
        return (cls.X,  cls.P,  cls.Y,
                cls.GX, cls.GP, cls.GY)

    @classmethod
    def has_same_shape(cls, ttype1: 'Ttype', ttype2: 'Ttype') -> bool:
        """ return True if ttype1 and ttype2 must have the same shape. """
        valid_pairs = [(cls.X, cls.GX),
                       (cls.P, cls.GP),
                       (cls.Y, cls.GY)]
        for valid_pair in valid_pairs:
            if ttype1 in valid_pair:
                return ttype2 in valid_pair
        raise ValueError
