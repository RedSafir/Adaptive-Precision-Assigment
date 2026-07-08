from ext3.typing import *

__all__ = [ 'TsDictMgr' ]

#===========#
# TsDictMgr #
#===========#
class TsDictMgr():
    #------#
    # data #
    #------#
    _record_on: bool = False
    data: Dict[int, Any] = {}

    @classmethod
    def record_on(cls, v: bool) -> None:
        if v is True:
            cls._record_on = True; cls.data = {}
        elif v is False:
            cls._record_on = False
        else:
            raise NotImplemented

    @classmethod
    def get(cls, tsptr: int) -> Opt[Any]:
        return cls.data.get(tsptr, None)

    @classmethod
    def update(cls, tsptr: int, v: Any) -> None:
        """ To add data, cls._record_on must be True. """
        if cls._record_on is True:
            cls.data[tsptr] = v
