from ext3.typing import *
from ext3.util   import list_flatten, list_exist_in
from .dtype      import Dtype
from .ttype      import Ttype
from .id         import Id
from .emodl      import EModl

import torch, copy

__all__ = [ 'Pasn', 'DtypePlan' ]

#======#
# Pasn # <--- EModl -> DtypePlan, where DtypePlan = (Ttype |-> Opt[Dtype]) Dict.
#======#      
#
# SPEC of Pasn.
#
# - data:
#   - data(emodl)(tt): dtype for tt at emodl.
#   - data(emodl)(tt)=dt   means that you use dt.
#   - data(emodl)(tt)=None means that you should infer it based on data(emodl'), where emodl' is from adj layers.
#     - E.g., if data(emodl)(X)=None, then we use data(emodl of prev layer)(Y).
#
DtypePlan = Dict[Ttype, Opt[Dtype]]
class Pasn():
    #-----#
    # var #
    #-----#
    data: Dict[EModl, DtypePlan]
    
    #--------#
    # create #
    #--------#
    def __init__(self, emodls: Union[Itr[EModl], Seq[EModl]], dtype_fwd: Dtype, dtype_bwd: Opt[Dtype]=None) -> None:
        """ 
        Y = P = GY = GP = dtype /\ X = GX = None.
        """
        if dtype_bwd is None: dtype_bwd = dtype_fwd
        dtplan = {Ttype.X : None,      Ttype.GX: None, 
                  Ttype.P : dtype_fwd, Ttype.GP: dtype_bwd,
                  Ttype.Y : dtype_fwd, Ttype.GY: dtype_bwd}
        self.data = {emodl: dict(dtplan) for emodl in emodls}

    def clone(self) -> 'Pasn':
        return copy.deepcopy(self)

    #-----#
    # str #
    #-----#
    def __repr__(self) -> str:
        res_list = []
        for emodl in self.data.keys():
            # set: res.
            res  = f'{type(emodl).__qualname__:15}: '
            res += f'dtype=['
            for ttype in (Ttype.Y, Ttype.GY, Ttype.P, Ttype.GP, Ttype.X, Ttype.GX):
                res += f'{self.data[emodl][ttype]}, '
            res += f']'
            res  = res.replace(', ]', ']')
            # set: res_list.
            res_list.append(res)
        return '\n'.join(res_list)
            
    #--------#
    # update #
    #--------#
    def update(self, upd_dtplan: DtypePlan, upd_flag: Callable[[EModl], bool]) -> None:
        """
        data[emodl]
        = data[emodl].update({upd_dtplan}), if {upd_flag}(emodl) is True;
        = data[emodl],                      otherwise.
        """
        for emodl in self.data:
            if upd_flag(emodl) is True:
                self.data[emodl].update(upd_dtplan)
        
    #-------------#
    # update_by_* #
    #-------------#
    def update_by_id(self, id_pos: str, id_mat: str, upd_dtplan: DtypePlan, upd_ids: Seq[Id]) -> None:
        """
        {id_mat} = 'id'  ==> use {upd_dtplan} and {upd_flag}(emodl) := ({id_pos}_emodl's id     in {upd_ids}    ).
        {id_mat} = 'knd' ==> use {upd_dtplan} and {upd_flag}(emodl) := ({id_pos}_emodl's id.knd in {upd_ids}.knd).
        """

        assert(id_pos in ('cur', 'prv', 'nxt'))
        assert(id_mat in ('id', 'knd'))
        
        def upd_flag(cur_emodl: EModl) -> bool:
            # set: emodls.
            if   id_pos == 'cur': emodls = [cur_emodl]
            elif id_pos == 'prv': emodls = list_flatten(cur_emodl.info_mdprv.mdref)
            elif id_pos == 'nxt': emodls = list_flatten(cur_emodl.info_mdnxt.mdref)
            # set: res.
            res: List[bool] = []
            for emodl in emodls:
                ids: Seq[Id] = list(emodl.info_mdcur.id.values())
                if   id_mat == 'id':
                    res.append(list_exist_in(ids, upd_ids))
                elif id_mat == 'knd':
                    res.append(list_exist_in([id.knd for id in ids], [id.knd for id in upd_ids]))
            # ret.
            if   id_pos == 'cur': return res[0]
            elif id_pos == 'prv': return all(res) and res != []
            elif id_pos == 'nxt': return any(res) and res != []
            # NOTE: res != [] is needed to correctly handle the first and last emodl.
            raise ValueError
            
        return self.update(upd_dtplan, upd_flag)

    def update_by_id_idv_all(self, id_pos: str, id_mat: str, upd_dtplan: DtypePlan, upd_idvals: Seq[int]) -> None:
        upd_ids = [Id(('idv_all', None ), idval) for idval in upd_idvals]
        self.update_by_id(id_pos, id_mat, upd_dtplan, upd_ids)

    def update_by_id_idv_mdl(self, id_pos: str, id_mat: str, upd_dtplan: DtypePlan, upd_idvals: Seq[int], upd_emdtps: Seq[Type[EModl]]) -> None:
        upd_ids = [Id(('idv_mdl', emdtp), idval) for idval in upd_idvals for emdtp in upd_emdtps]
        self.update_by_id(id_pos, id_mat, upd_dtplan, upd_ids)

    def update_by_id_grp_all(self, id_pos: str, id_mat: str, upd_dtplan: DtypePlan, upd_idvals: Seq[int]) -> None:
        upd_ids = [Id(('grp_all', None ), idval) for idval in upd_idvals]
        self.update_by_id(id_pos, id_mat, upd_dtplan, upd_ids)
