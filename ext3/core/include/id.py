from ext3.typing import *

import torch

__all__ = [ 'Id' ]

#====#
# Id # <--- id of entity = (knd, val).
#====#
class Id():
    #--------------#
    # obj var/func #
    #--------------#
    # - knd: kind of id.
    #   - see `ext3/core/emodlobj/emodlobjmgr.py` to know which `knd`s are actually used.
    # - val: val of id.
    #   - requires val \in [0, maxval[knd]] U [-(maxval[knd]+1), -1].
    knd: Any
    val: int

    def __init__(self, knd: Any, val: int) -> None:
        # assert(isinstance(val, int))
        self.knd = knd
        self.val = val
        self._normalize()

    def _normalize(self) -> None:
        """handle val < 0 in the same way as Python handles arr[val]."""
        if self.knd not in self.maxval:
            # NOTE: This branch is needed to use unallocated hypothetical Id,
            #       which is to be compared with an allocated real Id.
            self.val = -999
        else:
            totnum_knd = self.maxval[self.knd] + 1
            if -totnum_knd <= self.val < totnum_knd:
                self.val = self.val % totnum_knd
            else:
                raise ValueError

    def __repr__(self) -> str:
        return f'Id({self.knd}, {self.val})'

    def __eq__(self, other) -> bool:
        self ._normalize()
        other._normalize()
        return self.knd == other.knd and self.val == other.val

    def __hash__(self) -> int:
        return hash((self.knd, self.val))

    def __lt__(self, other) -> bool:
        self ._normalize()
        other._normalize()
        if isinstance(other, Id) and self.knd == other.knd:
            return self.val < other.val
        raise ValueError
        
    #----------------#
    # class var/func #
    #----------------#
    # - maxval[knd] = max {allocated `val`s for `knd`}.
    maxval: Dict[Any, int] = {}
    
    @classmethod
    def init(cls) -> None:
        """init Id class."""
        cls.maxval = {}
        
    @classmethod
    def get_curid(cls, knd: Any) -> 'Id':
        """return the current largest Id of kind `knd`."""
        return Id(knd, cls.maxval[knd])

    @classmethod
    def get_newid(cls, knd: Any) -> 'Id':
        """return new Id, starting from Id(knd, 0)."""
        if knd not in cls.maxval: 
            cls.maxval[knd] = -1
        cls.maxval[knd] += 1
        return Id(knd, cls.maxval[knd])
