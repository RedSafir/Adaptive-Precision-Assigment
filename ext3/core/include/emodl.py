from ext3.typing import *
from .dtype      import Dtype, DtypeRndModl
from .ttype      import Ttype
from .id         import Id
from .tsdictmgr  import TsDictMgr

import torch, time, functools

__all__ = [ 'EModl', '_hook_fwd_main', '_hook_bwd_main' ]

#=======#
# EModl #
#=======#
#
# SPEC of EModl.
#
# - info_ts[tt:Ttype]: TInfoTs.
#   - tsptr: List[int]. data_ptr of tensors.             [set by self._hook_fwd().]
#   - numel: Dict[str, List[int]]. numel of tensors.
#     - 'raw': raw numel for the already run batch_size. [set by self._hook_fwd().]
#     - 'eff': effective numel for future batch_size.    [set by EModlObjMgr.set_info_ts_numel().]
#   - id   : List[Id]. id of tensors.                    [set by self._hook_fwd().]
#   - dtype: List[Dtype]. dtype of tensors.              [set by EModlObjMgr.set_info_ts_dtype().]
#   - rndmd: List[DtypeRndModl]. rounding modules.       [set by EModlObjMgr.set_info_ts_rndmd().]
#   - undovr: List[Opt[TS]]. {under,over}flow ratio of tensors. [init by self.{set_param_forward_pre, _hook_fwd}(); set by Dtype.round_*().]
#
# - info_mdcur: TInfoMdcur.
#   - time: int. time when cur modl is executed. [set by self._hook_fwd().]
#   - id  : Dict[str, Id].                       [set by EModlObjMgr.set_info_mdcur_id().]
#     - 'idv_all': Id. individual id of cur modl, among all modules.
#     - 'idv_mdl': Id. individual id of cur modl, among the same kind of modls.
#     - 'grp_all': Id. group id of cur group (where cur modl belongs), among all groups.
#
# - info_md{prv,nxt}: TInfoMdadj..
#   - mdref: List[List[EModl]]. see below. [set by self._hook_fwd().]
#   - tsind: List[List[int]].   see below. [set by self._hook_fwd().]
#   
#   - info_mdprv.mdref[i] = refs of emodls that generates i-th input  to cur emodl.
#   - info_mdnxt.mdref[i] = refs of emodls that consumes  i-th output of cur emodl.
#   - info_mdprv.tsind[i] = indices of i-th input  to cur emodl, viewed as output of info_mdprv.mdref[i].
#   - info_mdnxt.tsind[i] = indices of i-th output of cur emodl, viewed as input  to info_mdnxt.mdref[i].
#   - E.g.:
#       x --[m1]--> t1 --[m2]--> t2 --[m3]--> t3 --[m5]--> t5
#                    |                              |
#                    |----------------[m4]--> t4 ---|
#       ===>
#            mdprv       mdprv     mdnxt     mdnxt
#            .mdref      .tsind    .mdref    .tsind
#       --------------------------------------------
#       m1   [[]       ] [[]     ] [[m2,m4]] [[0,0]]
#       m2   [[m1]     ] [0      ] [[m3]   ] [[0]  ]
#       m3   [[m2]     ] [0      ] [[m5]   ] [[0]  ]
#       m4   [[m1]     ] [0      ] [[m5]   ] [[1]  ]
#       m5   [[m3],[m4]] [[0],[0]] [[]     ] [[]   ]
#
class EModl(torch.nn.Module):
    #------#
    # vars #
    #------#
    # class var.
    # info_basecls: Type[torch.nn.Module]

    # flag for resetting self.info_*.
    flag_reset_info: bool

    # info on tensors.
    class TInfoTs():
        tsptr: List[int]
        numel: Dict[str, List[int]]
        id   : List[Id]
        dtype: List[Dtype]       ; dtype_backup: List[Dtype]
        rndmd: List[DtypeRndModl]; rndmd_backup: List[DtypeRndModl]
        undovr: List[Opt[TS]]
        def __init__(self):
            # NOTE: initiating self.undovr=[] is important because:
            # (1) it may not be even inited, if not here; but
            # (2) emodlobjmgr.py:get_undovrs() assumes that it is well-defined and a list.
            self.undovr = []
        def set_dtype_rndmd_backup(self):
            self.dtype_backup = list(self.dtype)
            if hasattr(self, 'rndmd'):
                self.rndmd_backup = list(self.rndmd)
        def set_dtype_rndmd_restore(self):
            self.dtype = self.dtype_backup
            if hasattr(self, 'rndmd'):
                self.rndmd = self.rndmd_backup
    info_ts: Dict[Ttype, TInfoTs]
    
    # info on cur emodl.
    class TInfoMdcur():
        time: int
        id  : Dict[str, Id]
        def __init__(self):
            self.time = -1
    info_mdcur: TInfoMdcur

    # info on adj emodls.
    class TInfoMdadj():
        mdref: List[List['EModl']]
        tsind: List[List[int]]
    info_mdprv: TInfoMdadj
    info_mdnxt: TInfoMdadj

    # values of param_{master,backup}.
    param_master: Dict[int, TS]
    param_backup: Dict[int, TS]
    
    # hooks of param.
    param_hook: Dict[int, Any]

    #------#
    # init #
    #------#
    def __init__(self, *args, **kwargs) -> None:
        super(EModl, self).__init__()
        # rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
        # print(f'{type(self).__qualname__:20}: init (rank={rank}).\t')

        # init: hooks.
        self.register_forward_hook      (self._hook_fwd) # type: ignore
        self.register_full_backward_hook(self._hook_bwd) # type: ignore

        # init: vars.
        self.init_vars()

    def init_vars(self) -> None:
        # init: reset_info.
        self.flag_reset_info = False
        # init: info_ts.
        self.info_ts = {tt: self.TInfoTs() for tt in (Ttype.X, Ttype.P, Ttype.Y, Ttype.GX, Ttype.GP, Ttype.GY)}
        # init: info_md{cur,prv,nxt}.
        self.info_mdcur = self.TInfoMdcur()
        self.info_mdprv = self.TInfoMdadj()
        self.info_mdnxt = self.TInfoMdadj()
        # init: param_*.
        self.param_master = {}
        self.param_backup = {}
        self.param_hook   = {}

    def reset_info(self, v: bool) -> None:
        # set: flag_reset_info.
        self.flag_reset_info = v

    #-------#
    # print #
    #-------#
    def print_info_dtype(self, out) -> None:
        """ print self.info_ts[*].dtype. """

        # set: res.
        res  = f'{type(self).__qualname__:15}: '
        res += f'id=['
        for id in self.info_mdcur.id.values():
            res += f'({id.knd[0]},{id.val:3d}), '
        res += f'], '
        res += f'dtype=['
        for ttype in (Ttype.Y, Ttype.GY, Ttype.P, Ttype.GP, Ttype.X, Ttype.GX):
            res += f'{self.info_ts[ttype].dtype}, '
        res += f']'
        res  = res.replace(', ]', ']')

        # out: res.
        end = '' if out == print else '\n'
        out(res + end)

    def print_info_mdadj(self, out) -> None:
        """ print self.info_md{prv,nxt}. """

        # set: res.
        res  = f'{type(self).__qualname__:15}: '
        res += f'id=['
        for id in self.info_mdcur.id.values():
            res += f'({id.knd[0]},{id.val:3d}), '
        res += f'], '
        res += f'mdprv=['
        for mdrefs in self.info_mdprv.mdref:
            res += f'['
            for mdref in mdrefs:
                res += f'{type(mdref).__qualname__:15}, '
            res += f'], '
        res += f'], '
        res += f'mdnxt=['
        for mdrefs in self.info_mdnxt.mdref:
            res += f'['
            for mdref in mdrefs:
                res += f'{type(mdref).__qualname__:15}, '
            res += f'], '
        res += f']'
        res  = res.replace(', ]', ']')

        # out: res.
        end = '' if out == print else '\n'
        out(res + end)

    #-------#
    # hooks # (for X, Y, GX, GY)
    #-------#
    def _hook_fwd(self, modl: torch.nn.Module, xs: Seq[TS], y: TS) -> TS:
        """ round y and set info_* of fwd tensors. """
        # rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
        # print(f'{type(self).__qualname__:20}: fwd_hook (rank={rank}).\t') #, force=True)

        # do: forward hook.
        ys: Seq[TS] # mypy hint.
        xs, ys, ps = xs, [y], list(self.parameters())
        ys = _hook_fwd_main(xs, ys, ps, self)
        y = ys[0]

        # ret.
        return y

    def _hook_bwd(self, modl: torch.nn.Module, gxs: Seq[Opt[TS]], gys: Seq[Opt[TS]]) -> Seq[Opt[TS]]:
        """ round gxs and set info_* of bwd tensors. """
        # rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
        # print(f'{type(self).__qualname__:20}: bwd_hook (rank={rank}).\t')

        # do: backward hook.
        gxs, gys, ps = gxs, gys, list(self.parameters())
        gxs = _hook_bwd_main(gxs, gys, ps, self)
        
        # ret.
        return gxs

    #-------#
    # hooks # (for P, GP)
    #-------#
    def set_param_forward_pre(self) -> None:
        """
        set {param, param_master}.data.
        - param_master.data = param.data.
        - param.data = rnd(param.data).
        """
        # init: emodl.info_ts[{P,GP}].undovr.
        if self.flag_reset_info is True:
            self.info_ts[Ttype.P] .undovr = [None for _ in self.parameters()]
            self.info_ts[Ttype.GP].undovr = [None for _ in self.parameters()]

        # with torch.no_grad():
        for i, param in enumerate(self.parameters()):
            # set: param_master.data.
            param_master = torch.empty(0)
            param_master.data = param.data
            self.param_master[i] = param_master

            # set: param.data (round).
            if hasattr(self.info_ts[Ttype.P], 'dtype'):
                # NOTE: disallow +-inf in roundings.
                param.data = Dtype.round_dtype(param.data, self.info_ts[Ttype.P].dtype[i], False, self, Ttype.P, i) # type: ignore

            # set: param.grad (round)---will be executed at the end of a backward pass.
            if hasattr(self.info_ts[Ttype.GP], 'dtype'):
                # NOTE: allow +-inf in roundings.
                def hook(j, t):
                    # if self.info_ts[Ttype.GP].dtype[j].exp_bias is None:
                    #     print(f'{type(self).__qualname__:15} {self.info_mdcur.id["idv_all"].val} P{j} {self.info_ts[Ttype.GP].dtype[j].cur_exp_bias}')
                    return Dtype.round_dtype(t, self.info_ts[Ttype.GP].dtype[j], True, self, Ttype.GP, j)

                # remove an old hook registered to param (if exists).
                if i in self.param_hook: self.param_hook[i].remove()
                # register a new hook to param.
                self.param_hook[i] = param.register_hook(functools.partial(hook, i))

    def set_param_backward_pos(self, loss_scale: float) -> None:
        """
        set {param, param.grad}.data.
        - param.data = param_master.data.
        - param.grad.data = rnd(param.grad.data)
        - param.grad.data /= loss_scale.
        """
        # with torch.no_grad():
        for i, param in enumerate(self.parameters()):
            # set: param.data.
            param.data = self.param_master[i].data

            # # set: param.grad.data (round).
            # if hasattr(self.info_ts[Ttype.GP], 'dtype'):
            #     # NOTE: allow +-inf in roundings.
            #     param.grad.data = Dtype.round_dtype(param.grad.data, self.info_ts[Ttype.GP].dtype[i], True)

        if loss_scale != 1.0:
            loss_scale_inv = 1./loss_scale
            # with torch.no_grad():
            for i, param in enumerate(self.parameters()):
                # set: param.grad.data (loss-scale).
                param.grad.data.mul_(loss_scale_inv)

    #-------#
    # param #
    #-------#
    def set_param_backup(self) -> None:
        """
        set {param, param_backup}.data.
        - param_backup.data = param.data.
        - param.data = clone(param.data).
        """
        # with torch.no_grad():
        for i, param in enumerate(self.parameters()):
            # set: param_backup.data.
            param_backup = torch.empty(0)
            param_backup.data = param.data
            self.param_backup[i] = param_backup

            # set: param.data.
            param.data = param.data.clone()

    def set_param_restore(self) -> None:
        """
        set param.data.
        - param.data = param_backup.data.
        """
        # with torch.no_grad():
        for i, param in enumerate(self.parameters()):
            # set: param.data.
            param.data = self.param_backup[i].data

    #-------------#
    # dtype_rndmd #
    #-------------#
    def set_dtype_rndmd_backup(self) -> None:
        for ttype in Ttype.get_all():
            self.info_ts[ttype].set_dtype_rndmd_backup()
    def set_dtype_rndmd_restore(self) -> None:
        for ttype in Ttype.get_all():
            self.info_ts[ttype].set_dtype_rndmd_restore()

#========#
# helper #
#========#
def _hook_fwd_main(xs: Seq[TS], ys: Seq[TS], ps: Seq[TS], emodl: EModl) -> Seq[TS]:
    """ round y and set emodl.info_* of fwd tensors. """
    # rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
    # print(f'rank={rank} | emodl={type(emodl).__qualname__:15} '
    #       f'| emodl.flag_reset_info={emodl.flag_reset_info} ', force=True)
    #       # f'| id[idv_all]={emodl.info_mdcur.id["idv_all"].val:3d} '
    #       # f'| dtype={[emodl.info_ts[Ttype.Y].dtype[i] for i in range(len(ys))]}\n', end='', force=True)

    # init: emodl.info_ts[{Y,GX}].undovr.
    if emodl.flag_reset_info is True:
        emodl.info_ts[Ttype.Y] .undovr = [None for _ in ys]
        emodl.info_ts[Ttype.GX].undovr = [None for _ in xs]

    # set: y (round).
    if hasattr(emodl.info_ts[Ttype.Y], 'dtype'):
        # NOTE: use info_ts[Y].rndmd and round_rndmd(...) for rounding.
        # NOTE: MUST use tuple(...) instead of list(...). Using list raises pytorch error.
        # NOTE: disallow +-inf in roundings of fwd pass; nop in roundings of bwd pass.
        ys = tuple(Dtype.round_rndmd(ys[i], emodl.info_ts[Ttype.Y].rndmd[i], False, emodl, Ttype.Y, i) for i,_ in enumerate(ys)) # type: ignore

    # set: emodl.{info_ts[*].{id,tsptr,numel['raw'],id}, info_mdcur.time, info_md{prv,nxt}.*}, TsDictMgr.
    if emodl.flag_reset_info is True:
        # set: emodl.info_ts[*].{tsptr,numel['raw'],id}.
        emodl.info_ts[Ttype.X ].tsptr = [t.data_ptr() for t in xs]
        emodl.info_ts[Ttype.P ].tsptr = [t.data_ptr() for t in ps]
        emodl.info_ts[Ttype.Y ].tsptr = [t.data_ptr() for t in ys]
        emodl.info_ts[Ttype.GX].tsptr = [-1           for t in xs] # NOTE: garbage vals.
        emodl.info_ts[Ttype.GP].tsptr = [-1           for t in ps] # NOTE: garbage vals.
        emodl.info_ts[Ttype.GY].tsptr = [-1           for t in ys] # NOTE: garbage vals.
        emodl.info_ts[Ttype.X ].numel = {'raw': [t.numel() for t in xs]}
        emodl.info_ts[Ttype.P ].numel = {'raw': [t.numel() for t in ps]}
        emodl.info_ts[Ttype.Y ].numel = {'raw': [t.numel() for t in ys]}
        emodl.info_ts[Ttype.GX].numel = {'raw': [t.numel() for t in xs]}
        emodl.info_ts[Ttype.GP].numel = {'raw': [t.numel() for t in ps]}
        emodl.info_ts[Ttype.GY].numel = {'raw': [t.numel() for t in ys]}
        emodl.info_ts[Ttype.X ].id = []
        emodl.info_ts[Ttype.P ].id = [Id.get_newid(('ts_all', None)) for t in ps]
        emodl.info_ts[Ttype.Y ].id = [Id.get_newid(('ts_all', None)) for t in ys]
        emodl.info_ts[Ttype.GX].id = []
        emodl.info_ts[Ttype.GP].id = [Id.get_newid(('ts_all', None)) for t in ps]
        emodl.info_ts[Ttype.GY].id = [Id.get_newid(('ts_all', None)) for t in ys]

        # set: emodl.info_mdcur.time.
        emodl.info_mdcur.time = time.time_ns()

        # init: emodl.info_md{prv,nxt}.*.
        emodl.info_mdprv.mdref = [ [] for _ in xs ]
        emodl.info_mdprv.tsind = [ [] for _ in xs ]
        emodl.info_mdnxt.mdref = [ [] for _ in ys ]
        emodl.info_mdnxt.tsind = [ [] for _ in ys ]
            
        # set: emodl.info_md{prv,nxt}.*.
        emodl_nxt = emodl
        for xi, xt in enumerate(xs):
            emodl_prv_info = TsDictMgr.get(xt.data_ptr())
            if emodl_prv_info is not None:
                emodl_prv, yi = emodl_prv_info
                emodl_prv.info_mdnxt.mdref[yi].append(emodl_nxt)
                emodl_prv.info_mdnxt.tsind[yi].append(xi)
                emodl_nxt.info_mdprv.mdref[xi].append(emodl_prv)
                emodl_nxt.info_mdprv.tsind[xi].append(yi)
                
        # set: TsDictMgr.
        for yi, yt in enumerate(ys):
            TsDictMgr.update(yt.data_ptr(), (emodl, yi))
            
    # ret.
    return ys

def _hook_bwd_main(gxs: Seq[Opt[TS]], gys: Seq[Opt[TS]], ps: Seq[TS], emodl: EModl) -> Seq[Opt[TS]]:
    """ round gxs and set emodl.info_* of bwd tensors. """
    # rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
    # print(f'rank={rank} | emodl={type(emodl).__qualname__:15} | id[idv_all]={emodl.info_mdcur.id["idv_all"].val:3d} | '
    #       f'dtype={[emodl.info_ts[Ttype.GX].dtype[i] for i in range(len(gxs))]}\n', end='', force=True)

    # set: gxs (round).
    if hasattr(emodl.info_ts[Ttype.GX], 'dtype'):        
        # NOTE: use info_ts[GX].dtype and round_dtype(...) for rounding.
        # NOTE: MUST use tuple(...) instead of list(...). Using list raises pytorch error.
        # NOTE: allow +-inf in the roundings.
        # for j in range(len(gxs)):
        #     if emodl.info_ts[Ttype.GX].dtype[j].exp_bias is None:
        #         print(f'{type(emodl).__qualname__:15} {emodl.info_mdcur.id["idv_all"].val} X{j} {emodl.info_ts[Ttype.GX].dtype[j].cur_exp_bias}')
        gxs = tuple(Dtype.round_dtype(gxs[i], emodl.info_ts[Ttype.GX].dtype[i], True, emodl, Ttype.GX, i) for i,_ in enumerate(gxs))

    # ret.
    return gxs
