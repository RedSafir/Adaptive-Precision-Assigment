import torch
import torch.nn as nn
import numpy as np
from torch.cuda.amp import autocast

from ext3.core.emodlobj import EModlObjMgr
from ext3.core.include.dtype import Dtype, FP32, FP16
from ext3.core.include.pasn import Pasn
from ext3.core.include.ttype import Ttype
from ext3.nn.nn_native import NativeConv2d, NativeLinear

def new_id_grp_all_factory():
    """
    Factory function to define block-based grouping logic for layers.
    - NativeLinear -> always new group, reset tracking
    - NativeConv2d -> new group ONLY if in_channels changes
    - Others (BN, ReLU, Pool) -> inherit previous group
    """
    last_in_channels = None
    def new_id_grp_all(emodl):
        nonlocal last_in_channels
        if isinstance(emodl, NativeLinear):
            last_in_channels = None
            return True
        elif isinstance(emodl, NativeConv2d):
            cur = emodl.in_channels
            ret = (last_in_channels != cur)
            last_in_channels = cur
            return ret
        else:
            return False
    return new_id_grp_all

def assign_precision(model: nn.Module, config: dict, base_dtype=None) -> Pasn:
    """
    Performs precision assignment via EModlObjMgr and Pasn.
    - Registers modules
    - Performs dummy forward pass to construct the EModl graph (under autocast if using FP16)
    - Demotes both activations and parameters to FP8 based on target ratio
    - Prints the grouping & demotion map
    """
    # 1. Detect base dtype
    if base_dtype is None:
        if 'grad_scaler_init_scale' in config:
            base_dtype = FP16
        else:
            base_dtype = FP32

    print("\n" + "=" * 80)
    if base_dtype == FP16:
        print("PRECISION ASSIGNMENT (Original APA Architecture - FP16 Base)")
    else:
        print("PRECISION ASSIGNMENT (Original APA Architecture)")
    print("=" * 80)
    
    # Register modules
    EModlObjMgr.unregister_all()
    EModlObjMgr.register(model)
    
    # Block-Based Tensor Grouping
    new_id_grp_all = new_id_grp_all_factory()
    EModlObjMgr.set_info_mdcur_id(new_id_grp_all)
    EModlObjMgr.reset_info(True)
    EModlObjMgr.set_param_forward_pre()
    
    # Dummy forward pass
    with torch.no_grad():
        if base_dtype == FP16:
            with autocast(dtype=torch.float16):
                dummy_input = torch.randn(2, 3, 32, 32).to(config['device'])
                model(dummy_input)
        else:
            dummy_input = torch.randn(2, 3, 32, 32).to(config['device'])
            model(dummy_input)
            
    EModlObjMgr.set_param_backward_pos(1.0)
    EModlObjMgr.reset_info(False)
    
    # Initialize effective tensor numels
    EModlObjMgr.set_info_ts_numel(2, config['batch_size'])
    
    # Create Pasn & Dtypes
    FP8 = Dtype(4, 3, 0)
    pasn = Pasn(EModlObjMgr.get_emodls(), dtype_fwd=base_dtype)
    
    # Demotion logic
    target_ratio = config['pa_upd_rmin']
    ids, r = EModlObjMgr.get_ids_chosen('grp_all', config['pa_upd_schm'], r_min=target_ratio)
    
    # Protect first and last layers
    if len(ids) > 0:
        sorted_emodls = EModlObjMgr.get_emodls_sort()
        first_id = sorted_emodls[0].info_mdcur.id['grp_all']
        last_id = sorted_emodls[-1].info_mdcur.id['grp_all']
        ids = [i for i in ids if i != first_id and i != last_id]
        
    upd_idvals = [i.val for i in ids]
    
    # Phase 1: Demote activations (cur node -> Y, GY)
    upd_dtplan_cur = {Ttype.Y: FP8, Ttype.GY: FP8}
    pasn.update_by_id_grp_all('cur', 'id', upd_dtplan_cur, upd_idvals)
    
    # Phase 2: Demote parameters (prv node -> P, GP)
    upd_dtplan_prv = {Ttype.P: FP8, Ttype.GP: FP8}
    pasn.update_by_id_grp_all('prv', 'id', upd_dtplan_prv, upd_idvals)
    
    # Apply to graph
    EModlObjMgr.set_info_ts_dtype(pasn)
    EModlObjMgr.set_info_ts_rndmd()
    
    print(f"  Target Demotion Ratio : {target_ratio:.3f}")
    print(f"  Total Layers          : {len(EModlObjMgr.get_emodls_sort())}")
    print(f"  Demoted Layers        : {len(ids)}")
    
    print("\n=== Grouping & Demotion Map ===")
    for emodl in EModlObjMgr.get_emodls_sort():
        uid = emodl.info_mdcur.id['grp_all'].val
        dtype_y = emodl.info_ts[Ttype.Y].dtype[0] if hasattr(emodl.info_ts[Ttype.Y], 'dtype') and len(emodl.info_ts[Ttype.Y].dtype) > 0 else base_dtype
        dtype_p = emodl.info_ts[Ttype.P].dtype[0] if hasattr(emodl.info_ts[Ttype.P], 'dtype') and len(emodl.info_ts[Ttype.P].dtype) > 0 else base_dtype
        mode_y = dtype_y.to_native().mode_name
        mode_p = dtype_p.to_native().mode_name
        markers = []
        if mode_y == "low_fp8": markers.append("Y=FP8")
        if mode_p == "low_fp8": markers.append("P=FP8")
        marker_str = f" [DEMOTED: {', '.join(markers)}]" if markers else ""
        print(f"Group {uid:3d} | {type(emodl).__qualname__:30s} | Y={mode_y:10s} P={mode_p:10s}{marker_str}")
        
    print("=" * 80)
    return pasn

def check_and_promote_overflow(ovr_thrs: float = 0.0, target_precision=FP32):
    """
    Checks if active/parameter tensor overflow ratio exceeds threshold,
    and promotes the layers back to high precision (FP32/FP16) accordingly.
    """
    undovrs = EModlObjMgr.get_undovrs()
    flag = np.concatenate([
        undovrs[Ttype.P][:, 1] > ovr_thrs,
        undovrs[Ttype.Y][:, 1] > ovr_thrs,
    ])
    EModlObjMgr.inc_ts_prec(flag, {Ttype.Y: target_precision, Ttype.P: target_precision})
