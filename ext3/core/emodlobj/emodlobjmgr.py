from ext3.typing        import *
from ext3.util          import dict_add_, dict_ratio, list_sort, logg_full
from ext3.core.include  import Dtype, INT, Ttype, Id, TsDictMgr, EModl, Pasn
from ext3.core.emodlcls import EFuncMgr

import torch, random
import numpy as np

__all__ = [ 'EModlObjMgr' ]

#=============#
# EModlObjMgr #
#=============#
class EModlObjMgr():
    #-----------#
    # modls_reg #
    #-----------#
    modls_reg: List[torch.nn.Module] = []
    inc_ts_prec_flag: List[float] = []

    @classmethod
    def register(cls, *modls: torch.nn.Module) -> None:
        for modl in modls:
            cls.modls_reg.append(modl)

    @classmethod
    def unregister_all(cls) -> None:
        cls.modls_reg = []

    @classmethod
    def get_emodls(cls) -> Itr[EModl]:
        """ return emodls as an iterator. """
        for modl in cls.modls_reg:
            for modl_sub in modl.modules():
                if isinstance(modl_sub, EModl): # and modl_sub.info_mdcur.time != -1:
                    yield modl_sub

    @classmethod
    def get_emodls_sort(cls) -> Seq[EModl]:
        """ return emodls sorted by emodl.info_mdcur.time. """
        emodl_time_list = [(emodl, emodl.info_mdcur.time) for emodl in cls.get_emodls()]
        emodl_time_list = list_sort(emodl_time_list, [(1,False)]) # sort by time (inc).
        emodls = [emodl for emodl, time in emodl_time_list]
        return emodls

    # @classmethod
    # def _get_basecls(cls, m: torch.nn.Module) -> Opt[Type[torch.nn.Module]]:
    #     if isinstance(m, EModl):
    #         return m.info_basecls
    #     return None

    #------#
    # call # (each emodl processed separately)
    #------#
    @classmethod
    def _call_in_emodls(cls, func_name: str, *args, **kwargs) -> None:
        """ call m.func_name(*args, **kwargs) for all emodls in cls.modls_reg. """
        for emodl in cls.get_emodls():
            getattr(emodl, func_name)(*args, **kwargs)

    @classmethod
    def _call_in_emodls_sort(cls, func_name: str, *args, **kwargs) -> None:
        """ call m.func_name(*args, **kwargs) for all emodls in cls.modls_reg. """
        for emodl in cls.get_emodls_sort():
            getattr(emodl, func_name)(*args, **kwargs)

    @classmethod
    def print_info_dtype(cls, out) -> None:
        cls._call_in_emodls_sort('print_info_dtype', out)

    @classmethod
    def print_info_mdadj(cls, out) -> None:
        cls._call_in_emodls_sort('print_info_mdadj', out)
        
    @classmethod
    def set_param_forward_pre(cls) -> None:
        cls._call_in_emodls('set_param_forward_pre')

    @classmethod
    def set_param_backward_pos(cls, loss_scale: float) -> None:
        cls._call_in_emodls('set_param_backward_pos', loss_scale)

    @classmethod
    def set_param_backup(cls) -> None:
        cls._call_in_emodls('set_param_backup')

    @classmethod
    def set_param_restore(cls) -> None:
        cls._call_in_emodls('set_param_restore')

    @classmethod
    def set_dtype_rndmd_backup(cls) -> None:
        cls._call_in_emodls('set_dtype_rndmd_backup')

    @classmethod
    def set_dtype_rndmd_restore(cls) -> None:
        cls._call_in_emodls('set_dtype_rndmd_restore')

    @classmethod
    def reset_info(cls, v: bool) -> None:
        assert(isinstance(v, bool))
        # do/don't reset emodl's info.
        cls._call_in_emodls('reset_info', v)
        # do/don't record data on TsDictMgr (in emodl's fwd/bwd hooks).
        TsDictMgr.record_on(v)

    #--------------#
    # set_info_md* #
    #--------------#
    @classmethod
    def set_info_mdcur_id(cls, new_id_grp_all: Callable[[EModl,], bool]) -> None:
        """
        set emodl.info_mdcur.id.
        - ASSUME: emodl.info_mdcur.time.
        """

        # init: Id.
        Id.init()

        # set: emodl.info_mdcur.id.
        for emodl in cls.get_emodls_sort():
            id_idv_all = Id.get_newid(('idv_all', None))
            id_idv_mdl = Id.get_newid(('idv_mdl', type(emodl)))
            if new_id_grp_all(emodl) is True:
                id_grp_all = Id.get_newid(('grp_all', None)) # alloc new id.
            else:
                id_grp_all = Id.get_curid(('grp_all', None)) # use cur id.
            emodl.info_mdcur.id = {'idv_all': id_idv_all,
                                   'idv_mdl': id_idv_mdl,
                                   'grp_all': id_grp_all}

    #--------------#
    # set_info_ts* #
    #--------------#
    @classmethod
    def set_info_ts_numel(cls, bs_cur: int, bs_fut: int) -> None:
        """
        set emodl.info_ts[*].numel['eff'].
        - ASSUME: emodl.info_ts[*].{tsptr, numel['raw']}.
        - set numel['eff'] for bs=bs_fut, based on numel['raw'] for bs=bs_cur.
        """

        # helper.
        def _get_numel_eff(emodl: EModl, ttype: Ttype, tsind: int) -> int:
            # set: numel_raw.
            numel_raw: int
            numel_raw = emodl.info_ts[ttype].numel['raw'][tsind]

            # set: ceff.
            ceff: float
            if   ttype in (Ttype.Y, Ttype.GY, Ttype.X, Ttype.GX): ceff = float(bs_fut) / float(bs_cur)
            elif ttype in (Ttype.P, Ttype.GP):                    ceff = 1.
            else: raise NotImplemented

            # ret.
            return int(numel_raw * ceff)

        # main.
        for emodl in cls.get_emodls():
            for ttype in Ttype.get_all():
                emodl.info_ts[ttype].numel['eff'] =\
                    [_get_numel_eff(emodl, ttype, tsind)
                     for tsind, _ in enumerate(emodl.info_ts[ttype].tsptr)]
                                    
    @classmethod
    def set_info_ts_dtype(cls, pasn: Pasn) -> None:
        """
        set emodl.info_ts[*].dtype.
        - ASSUME: emodl.{info_ts[*].tsptr, info_mdprv}.
        """

        # helper.
        def _get_dtype(emodl: EModl, ttype: Ttype, tsind: int) -> Dtype:
            # set: dtype_opt.
            dtype_opt: Opt[Dtype]
            dtype_opt = pasn.data[emodl][ttype]

            # set: ref_{emodls, ttype, tsinds}.
            ref_emodls: Seq[EModl]; ref_ttype: Ttype; ref_tsinds: Seq[int]
            if isinstance(dtype_opt, Dtype):
                ref_emodls = [emodl]
                ref_ttype  = ttype
                ref_tsinds = [tsind]
            elif dtype_opt is None:
                if   ttype == Ttype.X:  # use prv's Y.
                    ref_emodls = emodl.info_mdprv.mdref[tsind]
                    ref_ttype  = Ttype.Y
                    ref_tsinds = emodl.info_mdprv.tsind[tsind]
                elif ttype == Ttype.GX: # use prv's GY.
                    ref_emodls = emodl.info_mdprv.mdref[tsind]
                    ref_ttype  = Ttype.GY 
                    ref_tsinds = emodl.info_mdprv.tsind[tsind]
                else:
                    raise ValueError
            else:
                raise ValueError

            # set: ref_dtypes.
            ref_dtypes: Seq[Opt[Dtype]]
            ref_dtypes = [pasn.data[ref_emodl][ref_ttype] for ref_emodl in ref_emodls]

            # assert: ref_dtypes.
            assert(len(ref_dtypes) <= 1)
            assert(all(isinstance(ref_dtype, Dtype) for ref_dtype in ref_dtypes))

            # set: dtype.
            dtype: Dtype
            if   len(ref_dtypes) == 0: dtype = INT
            elif len(ref_dtypes) == 1: dtype = ref_dtypes[0] # type: ignore
            else: raise ValueError

            # ret.
            return dtype
        
        # main.
        for emodl in cls.get_emodls():
            for ttype in Ttype.get_all():
                emodl.info_ts[ttype].dtype =\
                    [_get_dtype(emodl, ttype, tsind).clone() # NOTE: use .clone() to init dtype.cur_exp_bias.
                     for tsind, _ in enumerate(emodl.info_ts[ttype].tsptr)]
                    
    @classmethod
    def set_info_ts_rndmd(cls) -> None:
        """
        set emodl.info_ts[*].rndmd.
        - ASSUME: emodl.info_ts[*].dtype.
        """

        # main.
        for emodl in cls.get_emodls():
            for ttype in [Ttype.Y]: # NOTE: set info_ts[*].rndmd for only Y; don't need for GX as it will use round func (not modl).

                # set: emodl.info_ts[*].rndmd.
                emodl.info_ts[ttype].rndmd =\
                    [dtype.get_rndmd()
                     for dtype in emodl.info_ts[ttype].dtype]

                # add: rnmd to emodl._modules[*].
                for i, rndmd in enumerate(emodl.info_ts[ttype].rndmd):
                    emodl.add_module(f'info_ts[{ttype}].rndmd[{i}]'.replace('.', ':'), rndmd)
                    
    #-------#
    # get_* #
    #-------#
    @classmethod
    def get_numbit(cls, dtype_use: Opt[Dict[Ttype,Dtype]]) -> int:
        """
        returns total number of bits for all tensors.
        - ASSUME: emodl.info_ts[*].{tsptr, numel['eff'], dtype}.
        - dtype_use != None ==> use dtype_use to compute numbit.
        - dtype_use == None ==> use cur dtype to compute numbit.
        """

        # helper.
        def _get_numbit(emodl: EModl, ttype: Ttype, tsind: int) -> int:
            numel = emodl.info_ts[ttype].numel['eff'][tsind]
            dtype = emodl.info_ts[ttype].dtype       [tsind]
            if dtype_use is not None:
                dtype = dtype_use[ttype] # use dtype_use, instead of currently set dtype.
            return numel * dtype.get_numbit()
            
        # main.
        res = 0
        for emodl in cls.get_emodls():
            for ttype in (Ttype.P, Ttype.Y, Ttype.GY, Ttype.GP): # exclude X and GX to avoid double counting.
                res += sum(
                    [_get_numbit(emodl, ttype, tsind)
                     for tsind, _ in enumerate(emodl.info_ts[ttype].tsptr)])
        return res
    
    @classmethod
    def get_numels_by_id(cls, id_type: str) -> Dict[Id, int]:
        """
        returns numel['eff'] by id[id_type]. 
        - ASSUME: emodl.{info_ts[*].numel['eff'], info_mdcur, info_mdprv}.
        - ASSUME: id(emodl_cur.Y) = id(emodl_cur); id(emodl_cur.P) = id(emodl_prv).
        """

        # helper.
        def _get_numel(emodl: EModl, ttype: Ttype) -> int:
            return sum(emodl.info_ts[ttype].numel['eff'])

        def _get_id(emodl: EModl, pos: str) -> Id:
            if   pos == 'cur': pass
            elif pos == 'prv': emodl = emodl.info_mdprv.mdref[0][0]
            else             : raise ValueError
            return emodl.info_mdcur.id[id_type]
        
        # main.
        res: Dict[Id, int]
        res = {}
        for emodl in cls.get_emodls():
            for ttype in (Ttype.P, Ttype.Y, Ttype.GY, Ttype.GP): # exclude X and GX to avoid double counting.
                numel = _get_numel(emodl, ttype)
                if numel > 0:
                    if   ttype in (Ttype.Y, Ttype.GY): id = _get_id(emodl, 'cur')
                    elif ttype in (Ttype.P, Ttype.GP): id = _get_id(emodl, 'prv')
                    else                             : raise ValueError
                    dict_add_(res, id, numel, 0)
        return res
    
    @classmethod
    def get_ids_chosen(cls, id_type: str, scheme: str, r_min: Opt[float]=None, k: Opt[int]=None, rand_seed: int=-1) -> Tuple[Seq[Id], float]:
        """
        return (res_ids, res_r), where res_ids = list of id[`id_dtype`] selected according to `scheme`, res_r = r(res_ids).
        - rand  : pick some ids in a random order          , until res_r >= `r_min`.
        - topr_*: pick some ids in a sorted order (dec/inc), until res_r >= `r_min`.
        - topk_*: pick `k`  ids in a sorted order (dec/inc).
        """
        # check: scheme.
        assert(scheme in ('rand', 'topr_dec', 'topr_inc', 'topk_dec', 'topk_inc'))
        
        # get: emodl's numels (proportional to args.bs_tr) by id[id_type].
        numels_abs: Dict[      Id, int   ]
        numels_rel: List[Tuple[Id, float]]
        numels_abs = cls.get_numels_by_id(id_type)
        numels_rel = list(dict_ratio(numels_abs).items())

        # reorder: numels_rel.
        if scheme == 'rand':
            if rand_seed >= 0: random.seed(rand_seed)
            random.shuffle(numels_rel)
        elif scheme.endswith('_dec'):
            numels_rel = list_sort(numels_rel, [(1,True ), (0,False)]) # sort first by ratio (dec) then by id (inc).
        elif scheme.endswith('_inc'):
            numels_rel = list_sort(numels_rel, [(1,False), (0,False)]) # sort first by ratio (inc) then by id (inc).

        # set: res_{ids,r}.
        res_ids: Seq[Id]
        res_r  : float
        if scheme == 'rand' or scheme.startswith('topr_'):
            # check: r.
            assert(r_min is not None and 0.0 <= r_min <= 1.0)
            # main.
            if   r_min == 0.0: # choose nothing.
                res_ids = []
                res_r   = 0.0
            elif r_min == 1.0: # choose all.
                res_ids = [id for id, r in numels_rel]
                res_r   = 1.0
            else:          # choose some in a given order.
                res_ids = []
                res_r   = 0.0
                for id_cur, r_cur in numels_rel:
                    if res_r >= r_min: break
                    res_ids.append(id_cur)
                    res_r += r_cur
        elif scheme.startswith('topk_'):
            # check: k.
            assert(k is not None and 0 <= k < len(numels_rel))
            # main.
            res_ids = [id for id, r in numels_rel[:k]]
            res_rs  = [r  for id, r in numels_rel[:k]]
            res_r   = sum(res_rs)
        else: 
            raise ValueError

        # ret.
        return (res_ids, res_r)
            
    @classmethod
    def get_cur_exp_bias(cls) -> Tuple[int,int,float]:
        res_min = +2**100
        res_max = -2**100
        res_sum = 0
        res_cnt = 0
        for emodl in cls.get_emodls():
            for ttype in Ttype.get_all():
                for dtype in emodl.info_ts[ttype].dtype:
                    if dtype.exp_bias is None and dtype.cur_exp_bias is not None:
                        res_min = min(res_min, dtype.cur_exp_bias)
                        res_max = max(res_max, dtype.cur_exp_bias)
                        res_sum += dtype.cur_exp_bias
                        res_cnt += 1
        if res_cnt == 0: return 0, 0, 0
        else:            return res_min, res_max, (res_sum / res_cnt)

    @classmethod
    def get_undovrs(cls) -> Dict[Ttype, np.ndarray]:
        zeros = torch.zeros((2,), device='cuda')

        # calc: undovrs on GPU.
        undovrs: Dict[Ttype, TS]
        undovrs = {}
        for ttype in (Ttype.P, Ttype.Y, Ttype.GX, Ttype.GP): # MATCH WITH: cls.handle_overflows().
            # undovrs[ttype] = torch.zeros(200, 2)
            undovrs[ttype] = torch.stack(tuple(
                t if t is not None else zeros
                for emodl in cls.get_emodls_sort()           # MATCH WITH: as above.
                for t in emodl.info_ts[ttype].undovr ))      # MATCH WITH: as above.

        # copy: undovrs from GPU to CPU.
        undovrs_np: Dict[Ttype, np.ndarray]
        undovrs_np = {k: v.cpu().numpy() for k,v in undovrs.items()}
        return undovrs_np

    @classmethod
    def inc_ts_prec(cls, flag: Union[np.ndarray, TS], dtype_hi_dict: Dict[Ttype, Dtype]) -> None:
        # init: cls.inc_ts_prec_flag.
        if cls.inc_ts_prec_flag == []:
            cls.inc_ts_prec_flag = [0.0] * len(flag)

        # nop, if flag is all False.
        if flag.sum() == 0: return

        # update: info_ts[*].{dtype, rndmd}, cls.tss_inc_prec.
        cnt = -1
        for ttype in (Ttype.P, Ttype.Y):                                # MATCH WITH: as above. # (Ttype.P, Ttype.Y, Ttype.GX, Ttype.GP):
            dtype_hi = dtype_hi_dict[ttype]
            for emodl in cls.get_emodls_sort():                         # MATCH WITH: cls.get_undovers().
                for tsind, _ in enumerate(emodl.info_ts[ttype].undovr): # MATCH WITH: as above.
                    cnt += 1

                    # detect: too big overflow ratio.
                    if flag[cnt] and emodl.info_ts[ttype].dtype[tsind] != dtype_hi:
                        # log.
                        logg_full('status',
                                  f'@@@OVERFLOW: {str(type(emodl)):40}: id={emodl.info_mdcur.id["idv_all"].val:3d}, '
                                  f'ttype={ttype}, tsind={tsind}\n') #, ovrfr_cur={ovrfr_cur}, ovrfr_thrs={ovrfr_thrs}\n')
                        # print(f'@@@OVERFLOW: {str(type(emodl)):40}: id={emodl.info_mdcur.id["idv_all"].val:3d}, '
                        #       f'ttype={ttype}, tsind={tsind}, rank={torch.distributed.get_rank()}\n', force=True)

                        # update: cls.inc_ts_prec_flag.
                        cls.inc_ts_prec_flag[cnt] = 1.0
                        
                        # update: (emodl_cur, ttype_cur)'s {dtype, rndmd}.
                        emodl.info_ts[ttype].dtype[tsind] = dtype_hi.clone()
                        if ttype == Ttype.Y:
                            # update: (emodl_cur, Y)'s rndmd.
                            emodl.info_ts[ttype].rndmd[tsind] = dtype_hi.get_rndmd()
                        else:
                            # don't need update as it was never set and will never be used.
                            pass

                        # update: (emodl_adj, ttype_adj)'s dtype.
                        if   ttype == Ttype.Y:
                            # update: (emodl_nxt, X)'s dtype.
                            for emodl_nxt, tsind_nxt in zip(emodl.info_mdnxt.mdref[tsind], emodl.info_mdnxt.tsind[tsind]):
                                emodl_nxt.info_ts[Ttype.X ].dtype[tsind_nxt] = dtype_hi.clone()
                        # elif ttype == Ttype.GX:
                        #     # update: (emodl_prv, GY)'s dtype.
                        #     for emodl_prv, tsind_prv in zip(emodl.info_mdprv.mdref[tsind], emodl.info_mdprv.tsind[tsind]):
                        #         emodl_prv.info_ts[Ttype.GY].dtype[tsind_prv] = dtype_hi.clone()

    @classmethod
    def get_inc_ts_prec_flag(cls) -> List[float]:
        res = cls.inc_ts_prec_flag
        cls.inc_ts_prec_flag = []
        return res

#----- OLD VER ------#                        

# @classmethod
# def set_info_mdadj_all(cls) -> None:
#     """ 
#     set emodl.info_mdadj.*.
#     - ASSUME: emodl.info_ts[*].tsptr.
#     """
#
#     # set: all_tsptrs_xy.
#     # [tsptr_x1, tsptr_y1, tsptr_x21, tsptr_x22, tsptr_y2, ...].
#     all_tsptrs_xy: List[int] = []
#     for emodl in cls.get_emodls():
#         for ttype in (Ttype.X, Ttype.Y):
#             all_tsptrs_xy.extend(emodl.info_ts[ttype].tsptr)
#
#     # set: equiv_tsptrs_sets.
#     # [{t1ptr, t1ptr', ...}, {t2ptr, t2ptr', ...}, ...].
#     equiv_tsptrs_list: List[Set[int]] = []
#
#     # --- add tsptrs of X and Y.
#     for tsptr in set(all_tsptrs_xy):
#         equiv_tsptrs_list.append(set([tsptr]))
#     # --- add (t1ptr, t2ptr)s from EFuncMgr.
#     for (t1ptr, t2ptr) in EFuncMgr.data_get():
#         found = False
#         for equiv_tsptrs in equiv_tsptrs_list:
#             if t1ptr in equiv_tsptrs or t2ptr in equiv_tsptrs:
#                 equiv_tsptrs.add(t1ptr)
#                 equiv_tsptrs.add(t2ptr)
#                 found = True; break
#         if found is False:
#             equiv_tsptrs = set([t1ptr, t2ptr])
#             equiv_tsptrs_list.append(equiv_tsptrs)
#
#     # set: normalize_tsptr.
#     # {t1ptr |-> t1ptr, t1ptr' |-> t1ptr, ..., t2ptr |-> t2ptr, ...}.
#     normalize_tsptr: Dict[int, int] = {}
#
#     for equiv_tsptrs in equiv_tsptrs_list:
#         val = min(equiv_tsptrs)
#         for tsptr in equiv_tsptrs:
#             normalize_tsptr[tsptr] = val
#
#     # set: yptr_to_emodl_yi.
#     # {yptr |-> (emodl, yi), ...}, where yptr is yi-th output of emodl.
#     yptr_to_emodl_yi: Dict[int, Tuple[EModl, int]] = {}
#
#     for emodl_cur in cls.get_emodls():
#         for (yi, yptr) in enumerate(emodl_cur.info_ts[Ttype.Y].tsptr):
#             yptr_nrm = normalize_tsptr[yptr]
#             yptr_to_emodl_yi[yptr_nrm] = (emodl_cur, yi)
#
#     # init: emodl.info_md{prv,nxt}.
#     for emodl in cls.get_emodls():
#         emodl.info_mdprv.mdref = [ [] for _ in emodl.info_ts[Ttype.X].tsptr ]
#         emodl.info_mdprv.tsind = [ [] for _ in emodl.info_ts[Ttype.X].tsptr ]
#         emodl.info_mdnxt.mdref = [ [] for _ in emodl.info_ts[Ttype.Y].tsptr ]
#         emodl.info_mdnxt.tsind = [ [] for _ in emodl.info_ts[Ttype.Y].tsptr ]
#
#     # set: emodl.info_md{prv,nxt}.
#     for emodl_nxt in cls.get_emodls():
#         for (xi, xptr) in enumerate(emodl_nxt.info_ts[Ttype.X].tsptr):
#             xptr_nrm = normalize_tsptr[xptr]
#             if xptr_nrm in yptr_to_emodl_yi:
#                 emodl_prv, yi = yptr_to_emodl_yi[xptr_nrm]
#                 if emodl_nxt != emodl_prv:
#                     emodl_prv.info_mdnxt.mdref[yi].append(emodl_nxt)
#                     emodl_prv.info_mdnxt.tsind[yi].append(xi)
#                     emodl_nxt.info_mdprv.mdref[xi].append(emodl_prv)
#                     emodl_nxt.info_mdprv.tsind[xi].append(yi)

# @classmethod
# def set_info_ts_dtype(cls, pasn: Pasn) -> None:
#     # NOTE: the order of (X, P, Y, ...) is important.
#     ttypes_all = (Ttype.X, Ttype.P, Ttype.Y, Ttype.GX, Ttype.GP, Ttype.GY)
#
#     # set: emodl.info_ts[*].dtype. first round. fills with Union[Dtype, Ttype, None].
#     for emodl in cls.get_emodls():
#         dtype_dict = pasn.rule(emodl.info_mdcur.id)
#         for ttype in ttypes_all:
#             emodl.info_ts[ttype].dtype = [dtype_dict[ttype]] * len(emodl.info_ts[ttype].tsptr) # type: ignore
#
#     # set: emodl.info_ts[*].dtype. second round. fills with Dtype.
#     for emodl in cls.get_emodls():
#         for ttype in ttypes_all:
#             for (tsind, dtype_draft) in enumerate(emodl.info_ts[ttype].dtype):
#                 # dtype_draft: Union[Dtype, Ttype, None].
#
#                 # set: ref_{emodl,ttype,tsind}.
#                 ref_emodls: List[EModl]
#                 ref_ttype : Ttype
#                 ref_tsinds: List[int]
#                 if isinstance(dtype_draft, Dtype):
#                     ref_emodls = [emodl]
#                     ref_ttype  = ttype
#                     ref_tsinds = [tsind]
#                 elif isinstance(dtype_draft, Ttype): 
#                     if Ttype.has_same_shape(ttype, dtype_draft) is True:
#                         ref_emodls = [emodl]
#                         ref_ttype  = dtype_draft
#                         ref_tsinds = [tsind]
#                     else:
#                         ref_emodls = [emodl]
#                         ref_ttype  = dtype_draft
#                         ref_tsinds = [0] # use 0 due to lack of correspondence.
#                 elif dtype_draft is None:
#                     if   ttype == Ttype.X:
#                         ref_emodls = emodl.info_mdprv.mdref[tsind]
#                         ref_ttype  = Ttype.Y
#                         ref_tsinds = emodl.info_mdprv.tsind[tsind]
#                     elif ttype == Ttype.GX:
#                         ref_emodls = emodl.info_mdprv.mdref[tsind]
#                         ref_ttype  = Ttype.GY
#                         ref_tsinds = emodl.info_mdprv.tsind[tsind]
#                     elif ttype == Ttype.Y:
#                         ref_emodls = emodl.info_mdnxt.mdref[tsind]
#                         ref_ttype  = Ttype.X
#                         ref_tsinds = emodl.info_mdnxt.tsind[tsind]
#                     elif ttype == Ttype.GY:
#                         ref_emodls = emodl.info_mdnxt.mdref[tsind]
#                         ref_ttype  = Ttype.GX
#                         ref_tsinds = emodl.info_mdnxt.tsind[tsind]
#                     else:
#                         raise ValueError
#                 else:
#                     raise ValueError
#
#                 # set: emodl.info_ts[*].dtype[*] = min(ref_dtypes).
#                 dtype_final = FP32
#                 for (ref_emodl, ref_tsind) in zip(ref_emodls, ref_tsinds):
#                     dtype_cur = ref_emodl.info_ts[ref_ttype].dtype[ref_tsind]
#                     assert(isinstance(dtype_cur, Dtype))
#                     dtype_final = Dtype.min(dtype_final, dtype_cur)
#                 emodl.info_ts[ttype].dtype[tsind] = dtype_final
