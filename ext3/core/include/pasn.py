<<<<<<< HEAD
=======
"""
pasn.py — Provides both legacy `Pasn` (Dtype planning) and `NativePasn` (runtime
native-precision assignment) for backward compatibility.

This module merges the original Pasn API used by older components (EModl-based
dtype planning) and the newer NativePasn used by native FP8/FP16 management.
"""

>>>>>>> 6373ff3dbda849e7adc086df40be0f1cde6e023d
from ext3.typing import *
from ext3.util   import list_flatten, list_exist_in
from .dtype      import Dtype
from .ttype      import Ttype
from .id         import Id
from .emodl      import EModl
<<<<<<< HEAD
=======

import torch, copy, random
import torch.nn as nn
>>>>>>> 6373ff3dbda849e7adc086df40be0f1cde6e023d

import torch, copy

<<<<<<< HEAD
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
=======
__all__ = [ 'Pasn', 'DtypePlan', 'NativePasn' ]

# ----- Legacy Pasn (Dtype planning used by emodlobjmgr) -----
>>>>>>> 6373ff3dbda849e7adc086df40be0f1cde6e023d
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
<<<<<<< HEAD
        Y = P = GY = GP = dtype /\ X = GX = None.
=======
        Initialize Pasn mapping: Y = P = GY = GP = dtype_fwd / X = GX = None.
>>>>>>> 6373ff3dbda849e7adc086df40be0f1cde6e023d
        """
        if dtype_bwd is None: dtype_bwd = dtype_fwd
        dtplan = {Ttype.X : None,      Ttype.GX: None, 
                  Ttype.P : dtype_fwd, Ttype.GP: dtype_bwd,
                  Ttype.Y : dtype_fwd, Ttype.GY: dtype_bwd}
        self.data = {emodl: dict(dtplan) for emodl in emodls}
<<<<<<< HEAD

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
=======

    def clone(self) -> 'Pasn':
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        res_list = []
        for emodl in self.data.keys():
            res  = f'{type(emodl).__qualname__:15}: '
            res += f'dtype=['
            for ttype in (Ttype.Y, Ttype.GY, Ttype.P, Ttype.GP, Ttype.X, Ttype.GX):
                res += f'{self.data[emodl][ttype]}, '
            res += f']'
            res  = res.replace(', ]', ']')
            res_list.append(res)
        return '\n'.join(res_list)

    def update(self, upd_dtplan: DtypePlan, upd_flag: Callable[[EModl], bool]) -> None:
        for emodl in self.data:
            if upd_flag(emodl) is True:
                self.data[emodl].update(upd_dtplan)
        
    def update_by_id(self, id_pos: str, id_mat: str, upd_dtplan: DtypePlan, upd_ids: Seq[Id]) -> None:
        assert(id_pos in ('cur', 'prv', 'nxt'))
        assert(id_mat in ('id', 'knd'))
        
        def upd_flag(cur_emodl: EModl) -> bool:
            if   id_pos == 'cur': emodls = [cur_emodl]
            elif id_pos == 'prv': emodls = list_flatten(cur_emodl.info_mdprv.mdref)
            elif id_pos == 'nxt': emodls = list_flatten(cur_emodl.info_mdnxt.mdref)
            res: List[bool] = []
            for emodl in emodls:
                ids: Seq[Id] = list(emodl.info_mdcur.id.values())
                if   id_mat == 'id':
                    res.append(list_exist_in(ids, upd_ids))
                elif id_mat == 'knd':
                    res.append(list_exist_in([id.knd for id in ids], [id.knd for id in upd_ids]))
            if   id_pos == 'cur': return res[0]
            elif id_pos == 'prv': return all(res) and res != []
            elif id_pos == 'nxt': return any(res) and res != []
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


# ----- NativePasn: runtime native-precision manager used by newer code -----
class NativePasn:
    """
    Manajer yang mengatur Siklus Grouping -> Demotion -> Promotion untuk native precision.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.layers: List[nn.Module] = []
        self.group_sizes: List[int] = []
        self._group_layers()

    def _group_layers(self) -> None:
        self.layers = []
        self.group_sizes = []
        from ext3.nn.nn_native import NativeConv2d, NativeLinear
        for name, module in self.model.named_modules():
            if isinstance(module, (NativeConv2d, NativeLinear)):
                self.layers.append(module)
                numel = sum(p.numel() for p in module.parameters() if p.requires_grad)
                self.group_sizes.append(numel if numel > 0 else 1)

    def apply_demotion(self, scheme: str, r_min: float, r_max: float, base_mode: NativePrecisionMode, low_mode: NativePrecisionMode, protect_ends: bool = True, seed: int = -1) -> Dict[str, str]:
        for layer in self.layers:
            layer.set_native_precision(base_mode)
        if not self.layers:
            return {}
        total_size = sum(self.group_sizes)
        rel_sizes = [(i, size, size / total_size) for i, size in enumerate(self.group_sizes)]
        if scheme == 'rand':
            if seed >= 0:
                random.seed(seed)
            random.shuffle(rel_sizes)
        elif scheme == 'topr_dec':
            rel_sizes.sort(key=lambda x: x[1], reverse=True)
        elif scheme == 'topr_inc':
            rel_sizes.sort(key=lambda x: x[1], reverse=False)
        else:
            print(f"[NativePasn] Warning: Unknown scheme '{scheme}', fallback to base_mode only.")
            return {}
        demote_indices = set()
        current_r = 0.0
        for idx, size, ratio in rel_sizes:
            if current_r >= r_min:
                break
            demote_indices.add(idx)
            current_r += ratio
        demoted_count = 0
        for i, layer in enumerate(self.layers):
            if protect_ends and (i == 0 or i == len(self.layers) - 1):
                layer.set_native_precision(base_mode)
                continue
            if i in demote_indices:
                layer.set_native_precision(low_mode)
                demoted_count += 1
        return {
            'total_layers': len(self.layers),
            'demoted_layers': demoted_count,
            'base_mode': str(base_mode),
            'low_mode': str(low_mode),
            'scheme': scheme,
            'target_ratio': r_min,
            'actual_ratio': current_r
        }

    def check_and_promote(self, base_mode: NativePrecisionMode) -> int:
        from ext3.nn.nn_native import get_fp8_manager
        manager = get_fp8_manager()
        promoted_count = 0
        for layer in self.layers:
            if layer.get_native_precision() == NativePrecisionMode.LOW_FP8:
                layer_uid = layer._ensure_layer_uid()
                fwd_uid = f"{layer_uid}.fwd_input"
                wt_uid = f"{layer_uid}.fwd_weight"
                if manager.has_spike(fwd_uid) or manager.has_spike(wt_uid):
                    layer.set_native_precision(base_mode)
                    manager.clear_spike(fwd_uid)
                    manager.clear_spike(wt_uid)
                    promoted_count += 1
                    print(f"*** [NativePasn] PROMOTION: Layer {layer_uid} mengalami AMAX spike. Promoted to {base_mode}.")
        return promoted_count

# Backwards-compatible alias: allow `from ext3.core.include import Pasn` to work
# where code expects the legacy Pasn API.

>>>>>>> 6373ff3dbda849e7adc086df40be0f1cde6e023d
