from ext3.typing import *

import logging, pathlib, os, string, torch, math, concurrent.futures

__all__ = [
    'logg_get_fname', 'logg_init', 'logg_full', 'logg',
    'torch_l1_norm', 'torch_l2_norm', 'torch_lf_norm',
    'torch_l1_mean', 'torch_l2_mean', 'torch_lf_mean',
    'torch_nonparallel_modl', '_make_divisible',
    'dict_add_', 'dict_append_', 'dict_map', 'dict_join', 'dict_ratio',
    'list_sort', 'list_flatten', 'list_exist_in',
]

#======#
# logg #
#======#
_global_loggers = {}
_global_log_resdir = './res'
_global_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1) 

# logg_*.
def logg_get_fname(lname: str, args, fname_base: str='', fname_sffx: str='') -> str:
    # set: dname.
    dname = os.path.join(_global_log_resdir, lname)

    # set: fname.
    if fname_base != '':
        fname_base = os.path.splitext(os.path.basename(fname_base))[0] # remove dirs and ext in fname_base.
        fname = f'{dname}/{fname_base}{fname_sffx}.log'
    else:
        if hasattr(args, 'resume') and args.resume != '':
            fname_base = os.path.basename(os.path.dirname(args.resume))
            fname = f'{dname}/{fname_base}0.log'
        else:
            fname = f'{dname}/{args.data}-{args.model}-wm{args.model_wm}-{args.logg_uid}-log.log'
    return fname

def logg_init(lname: str, fname: str, use_stdout: bool=True) -> None:
    # set: logger.
    logger = logging.getLogger(lname)
    logger.setLevel(logging.WARNING)
    logger.propagate = False # do not print WARNING msgs to stdout automatically.
    if fname != '.':
        pathlib.Path(os.path.dirname(fname)).mkdir(parents=True, exist_ok=True)
        logger.addHandler(logging.FileHandler(fname))
    logger.use_stdout = use_stdout # type: ignore
    for h in logger.handlers:
        h.terminator = '' # type: ignore # no newline.
    # if use_stdout is True:
    #     logger.addHandler(logging.StreamHandler())

    # set: _global_loggers.
    global _global_loggers
    _global_loggers[lname] = logger

def logg_full(lname: str, *args) -> None:
    # set: logger.
    if lname not in _global_loggers:
        return
    logger = _global_loggers[lname]

    # log: args.
    res = ' '.join([f'{arg}' for arg in args])
    # logger.warning(res)
    _global_executor.submit(logger.warning, res)
    if logger.use_stdout: # type: ignore
        print(res, end='', flush=True)
    # for h in logger.handlers: h.flush() # flush.
    
def logg(*args) -> None:
    logg_full('.', *args)


#=======#
# torch #
#=======#
# torch_l*_norm.
def torch_l1_norm(xs: Seq[TS], dim=None) -> TS:
    kwargs = {'dim': dim} if dim is not None else {}
    res = torch.tensor([x.abs()   .sum(**kwargs) for x in xs]).sum(dim=0)
    return res
def torch_l2_norm(xs: Seq[TS], dim=None) -> TS:
    kwargs = {'dim': dim} if dim is not None else {}
    res = torch.tensor([x.square().sum(**kwargs) for x in xs]).sum(dim=0)
    return torch.sqrt(res)
def torch_lf_norm(xs: Seq[TS], dim=None) -> TS:
    kwargs = {'dim': dim} if dim is not None else {}
    res = torch.tensor([x.abs()   .max(**kwargs) for x in xs]).max(dim=0)
    return res[0]

# torch_l*_mean.
def torch_l1_mean(xs: Seq[TS], dim=None) -> TS: # (|x1| + ... + |xn|) / n.
    n = sum(x.shape[dim] if dim is not None else x.numel() for x in xs)
    return torch_l1_norm(xs, dim) / n
def torch_l2_mean(xs: Seq[TS], dim=None) -> TS: # (|x1|^2 + ... + |xn|^2)^(1/2) / n^(1/2).
    n = sum(x.shape[dim] if dim is not None else x.numel() for x in xs)
    return torch_l2_norm(xs, dim) / math.sqrt(n)
def torch_lf_mean(xs: Seq[TS], dim=None) -> TS: # max(|x1|, ..., |xn|).
    return torch_lf_norm(xs, dim)

# get_nonparallel_modl.
def torch_nonparallel_modl(modl: torch.nn.Module) -> torch.nn.Module:
    if isinstance(modl, (torch.nn.DataParallel, torch.nn.parallel.distributed.DistributedDataParallel)):
        return modl.module
    return modl

# _make_divisible. (from torchvision.models._utils)
def _make_divisible(v: float, divisor: int, min_value: Opt[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

#========#
# python #
#========#
# dict.
def dict_add_(d: dict, k, v_add, v_init=0) -> None:
    """d[k]=v_init if d[k] is undefined; d[k] += v_add."""
    d[k] = d.get(k, v_init) + v_add

def dict_append_(d: dict, k, v_append) -> None:
    if k not in d: d[k] = [v_append]
    else:          d[k].append(v_append)

def dict_map(f: Callable, d: dict) -> dict:
    # d={k1:v1, ...} ---> {k1:f(v1), ...}
    return {k: f(v) for k, v in d.items()}

def dict_join(f: Callable, d_l: Seq[dict]) -> dict:
    # d_l=[{k1:v1, ...}, {k1:v1', ...}, ...] ---> {k1: f([v1,v1',...]), ...}
    return {k: f([d[k] for d in d_l]) for k in d_l[0].keys()}

def dict_ratio(d: dict) -> dict:
    # d={k1:v1, ...} ---> {k1:v1/s, ...} (s = v1+...).
    tot = sum(d.values())
    return dict_map(lambda v: float(v/tot), d)

# list.
def list_sort(l: Seq[Any], config: List[Tuple[int, bool]]) -> list:
    # config=[(i1,rev1),...] ---> sort by i1-th component with order rev1; for ties, sort by i2-th with order rev2; ...
    # rev=True <==> descending order.
    res = list(l)
    for i, rev in reversed(config):
        res.sort(key=lambda elt: elt[i], reverse=rev)
    return res

def list_flatten(l: List[List[Any]]) -> List[Any]:
    # l=[l1, l2, ...] --> l1 @ l2 @ ...
    res = []
    for l_sub in l: res.extend(l_sub)
    return res

def list_exist_in(l1: Seq, l2: Seq) -> bool:
    # l1=[v1, v2, ...], l2 --> \exists i. ai \in l2.
    return any((v in l2) for v in l1)
