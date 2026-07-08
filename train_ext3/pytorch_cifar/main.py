# based on: https://github.com/kuangliu/pytorch-cifar

'''Train CIFAR10 with PyTorch.'''
import torch
#===== 
# import torch.nn as nn
# import torch.nn.functional as F
from   ext3.typing import *
import ext3
import time, contextlib, copy, random, string, sys
import numpy as np
#=====
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision # type: ignore
import torchvision.transforms as transforms # type: ignore
import os
import argparse

#===== 
from .models import *
from .utils import progress_bar
#=====

#===== 
# do_exit:      bool
# args.
args:         argparse.Namespace
# data.
dataload_trn: torch.utils.data.DataLoader
dataload_tst: torch.utils.data.DataLoader
data_meas:    List[List[TS]] # [[inputs1, targets1], ...].
data_sngl:    List[List[TS]] # [[inputs1, targets1], ...].
# model.
net:          torch.nn.Module
_net:         torch.nn.Module
criterion:    torch.nn.Module
_criterion:   torch.nn.Module
best_acc:     float
start_epoch:  int
# opt.
optimizer:    torch.optim.Optimizer
scheduler:    torch.optim.lr_scheduler._LRScheduler
lscaler:      torch.cuda.amp.GradScaler
#=====

def set_args() -> None:
    global args # out.
    print('==> Setting args..')

    # set: parser.
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    #===== 
    # misc.
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument("--logg-uid", default='', type=str, help="unique string to be appended to output filenames (to prevent overwriting).")
    parser.add_argument("--logg-undovr-freq", default=100, type=int, help="log {under,over}flow ratios every [--logg-undovr-freq] steps.")

    # compute.
    parser.add_argument('--cuda', default=0, type=int, help='cuda device id (0,1,... / -1 / -2). -1: cuda with DataParallel. -2: cpu.')

    # model.
    parser.add_argument('--model', required=True, type=str, choices=[
        'res18', 'mblv2', 'shfv2', 'sqz',
    ], help='model name')
    parser.add_argument('--model-wm', default=1.0, type=float, help="model's width multiplier")
    parser.add_argument('--resume', default="", type=str, help='path of checkpoint')

    # data.
    parser.add_argument('--data-path', default='./dataset/', type=str, help='path to the dir that contains the dataset')
    parser.add_argument('--data', required=True, type=str, choices=['cifar10', 'cifar100'], help='dataset')
    parser.add_argument('--workers', default=2, type=int, help="number of data loading workers")
    parser.add_argument('--bs-tr', default=256, type=int, help='batch size for training (over all GPUs)')
    parser.add_argument('--bs-ts', default=200, type=int, help='batch size for test (over all GPUs)')
    parser.add_argument('--bs-ms', default=None, type=int, help='batch size for measuring fp error')
    parser.add_argument('--bn-ms', default=4, type=int, help='batch number for measuring fp error')

    # train.
    parser.add_argument('--ep', default=200, type=int, help='num epochs for training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--mt', default=0.9, type=float, help='momentum')
    parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--lsc-init', default=1.0, type=float, help='loss scale: init_scale=[--ls-init]')
    parser.add_argument('--lsc-evry-nep', default=-1.0, type=float, help='loss scale: growth_interval=[# batches]*[--ls-evry-nep]')
    parser.add_argument('--lr-schd', default='cos', type=str, choices=['stp', 'mulstp', 'cos'], help='lr scheduler')
    parser.add_argument('--lr-steps', default=[100, 150, 200], type=int, nargs='+',
                        help='decrease lr (StepLR==>every s0 epochs | MultiStepLR==>at each si epoch) (args.lr_steps=[s0,s1,...])')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr (StepLR==>by a factor of args.lr_gamma)')

    # prec.
    parser.add_argument('--pa-def-dt',   default=['FP32'], type=str, nargs='+', help='pasn: default dtype. nargs=1,2 (1: fwd=bwd / 2: fwd, bwd).')
    parser.add_argument('--pa-upd-schm', default=None,   type=str, choices=[
        'std', # this implements HFP8 [NIPS'19].
        'rand',
        'topr_dec', 'topr_inc',
        'topk_dec', 'topk_inc',
    ], help='pasn: update scheme.\n'
                        '  - std    = update layers as most other papers do.\n'
                        '  - rand   = update id["grp_all"]s chosen randomly with ratio r        such that r is the smallest possible in [r1,r2].\n'
                        '  - topr_* = update id["grp_all"]s having top-r(ratio) numels(dec/inc) such that r is the smallest possible in [r1,r2].\n'
                        '  - topk_* = update id["grp_all"]s having top-k(num  ) numels(dec/inc).\n')
    parser.add_argument('--pa-upd-args', default=None,   type=str, nargs='+', help='pasn: update scheme\'s args.\n'
                        '  - std    = [].\n'
                        '  - rand   = [r1,r2].\n'
                        '  - topr_* = [r1,r2].\n'
                        '  - topk_* = [k].\n')
    parser.add_argument('--pa-upd-dts',  default=None,   type=str, nargs=4, help='pasn: update dtype for [P, Y, GY, GP].')
    parser.add_argument('--pa-upd-seed', default=-1,     type=int, help='pasn: random seed for schm=rand. -1 denotes None.')
    parser.add_argument('--pa-upd-no-end', action='store_true', help='pasn: do not update dtype of first layer and last group.')
    parser.add_argument('--pa-upd-no-bwd', action='store_true', help='pasn: do not update dtype of bwd tensors.')
    parser.add_argument('--pa-ovr-thrs',  default=-1.0, type=float,
                        help='pasn: threshold for overflow ratio. None means no threshold checks.'
                        'when exceeding the threshold, a tensor will be stored in high-precision until the end of training.')

    # measure.
    parser.add_argument('--meas-no-fperr', action='store_true', help='measure: do not measure fp errs during training.')
    #=====

    # set: args.
    args = parser.parse_args()

    #===== 
    # compute.
    # set: args.{device,data_parallel}.
    if 0 <= args.cuda and args.cuda < torch.cuda.device_count():
        args.device = torch.device(f'cuda:{args.cuda}')
    elif args.cuda == -1:
        assert(torch.cuda.is_available())
        args.device = torch.device(f'cuda')
    elif args.cuda == -2:
        args.device = torch.device(f'cpu')
    else: assert(False)
    args.data_parallel = (str(args.device)=='cuda') # only cuda. not cuda:0, etc.
    # set: cuda device.
    if args.cuda >= 0:
        torch.cuda.set_device(args.cuda)

    # data.
    # set: args.bs_ms.
    if args.bs_ms is None:
        # args.bs_ms = max(args.bs_tr // args.bn_ms, 1)
        args.bs_ms = args.bs_tr

    # train.
    # set: args.{lr_tmax}.
    args.lr_tmax = args.ep

    # prec.
    pass

    # logg.
    # set: args.logg_uid.
    if args.logg_uid == '':
        args.logg_uid = time.strftime("%Y%m%d%H%M%S") + ''.join(random.choices(string.ascii_lowercase, k=3)) # cur time + rand str.
    # set: args.logg_fname.
    args.logg_fname = ext3.util.logg_get_fname('.', args)
    args.logg_fname_undovr = ext3.util.logg_get_fname('undovr', args, fname_base=args.logg_fname, fname_sffx=f'')
    args.logg_fname_undovr = os.path.join(os.path.splitext(args.logg_fname_undovr)[0], f'undovr-rk0.log')
    ext3.util.logg_init('.',      args.logg_fname,        use_stdout=True)
    ext3.util.logg_init('undovr', args.logg_fname_undovr, use_stdout=False)
    #=====

def print_args() -> None:
    global args # in.

    # print: args.
    ext3.util.logg(f'===== CONFIG =====\n')
    ext3.util.logg(f'logg    : logg_uid={args.logg_uid}, logg_undovr_freq={args.logg_undovr_freq}, '
                   f'name={args.logg_fname}, name_(undovr)=({args.logg_fname_undovr})\n')
    ext3.util.logg(f'comp    : name={args.device}, cuda(id)={args.cuda}, data_parallel={args.data_parallel}, #worker={args.workers}\n')
    ext3.util.logg(f'model   : name={args.model}, width_multiplier={args.model_wm}\n')
    ext3.util.logg(f'data    : name={args.data}, batch_size(train,test,meas)=({args.bs_tr},{args.bs_ts},{args.bs_ms}), batch_num(meas)=({args.bn_ms})\n')
    ext3.util.logg(f'optim   : name=SGD, epochs={args.ep}, lr={args.lr}, momentum={args.mt}, weight_decay={args.wd}, '
                   f'loss_scale_(init,every_nepoch)=({args.lsc_init}, {args.lsc_evry_nep})\n')
    ext3.util.logg(f'lr_schd : name={args.lr_schd}, '
                   f'[StepLR==>(step_sizes={args.lr_steps}, gamma={args.lr_gamma}), '
                   f'CosineLR==>(T_max={args.lr_tmax})]\n')
    ext3.util.logg(f'pasn    : def_dt={args.pa_def_dt}, upd_(schm,args,dts,seed)=({args.pa_upd_schm}, {args.pa_upd_args}, {args.pa_upd_dts}, {args.pa_upd_seed}), '
                   f'upd_no_(end,bwd)=({args.pa_upd_no_end},{args.pa_upd_no_bwd}), ovr_thrs={args.pa_ovr_thrs}\n')
    ext3.util.logg(f'meas    : no_fperr={args.meas_no_fperr}\n')
    ext3.util.logg(f'------------------\n')
    ext3.util.logg(f'args = {args}\n')
    ext3.util.logg(f'==================\n\n')

def set_data() -> None:
    global args # in.
    global args, dataload_trn, dataload_tst, data_meas, data_sngl # out.
    print('==> Setting data..')

    #===== 
    # set: dataset_{trn,tst}, args.data_num_classes.
    if args.data == 'cifar10':
        transform_trn = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
        transform_tst = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
        dataset_trn = torchvision.datasets.CIFAR10(root=f'{args.data_path}/cifar10', train=True,  download=True, transform=transform_trn)
        dataset_tst = torchvision.datasets.CIFAR10(root=f'{args.data_path}/cifar10', train=False, download=True, transform=transform_tst)
        args.data_num_classes = len(dataset_trn.classes) #= 10.
    elif args.data == 'cifar100':
        # ref: https://blog.jovian.ai/image-classification-of-cifar100-dataset-using-pytorch-8b7145242df1
        transform_trn = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025)), ])
        transform_tst = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025)), ])
        dataset_trn = torchvision.datasets.CIFAR100(root=f'{args.data_path}/cifar100', train=True,  download=True, transform=transform_trn)
        dataset_tst = torchvision.datasets.CIFAR100(root=f'{args.data_path}/cifar100', train=False, download=True, transform=transform_tst)
        args.data_num_classes = len(dataset_trn.classes) #= 100.
    else:
        raise NotImplemented

    # set: dataload_{trn,tst,sngl,meas}.
    dataload_trn  = torch.utils.data.DataLoader(dataset_trn, batch_size=args.bs_tr, shuffle=True,  num_workers=args.workers, pin_memory=True)
    dataload_tst  = torch.utils.data.DataLoader(dataset_tst, batch_size=args.bs_ts, shuffle=False, num_workers=args.workers, pin_memory=True)
    dataload_sngl = torch.utils.data.DataLoader(dataset_trn, batch_size=1,          shuffle=True,  num_workers=1, pin_memory=True)
    dataload_meas = torch.utils.data.DataLoader(dataset_trn, batch_size=args.bs_ms, shuffle=True,  num_workers=1, pin_memory=True)

    # set: data_{sngl,meas}.
    data_sngl = []
    for data in dataload_sngl:
        if len(data_sngl) >= 1: break
        data_sngl.append([t.clone().to(args.device) for t in data])
    data_meas = []
    for data in dataload_meas:
        if len(data_meas) >= args.bn_ms: break
        data_meas.append([t.clone().to(args.device) for t in data])
    #=====

def print_data() -> None:
    global args, dataload_trn, dataload_tst, data_meas, data_sngl # in.

    # print: data info.
    ext3.util.logg(f'===== DATA =====\n')
    ext3.util.logg(f'info         : num_classes={args.data_num_classes}\n')
    ext3.util.logg(f'dataload_trn : len={len(dataload_trn)}, {dataload_trn}\n')
    ext3.util.logg(f'dataload_tst : len={len(dataload_tst)}, {dataload_tst}\n')
    ext3.util.logg(f'data_sngl    : len={len(data_sngl)}, [(shp,val),..]={[(list(t.shape), t.view(-1)[:5]) for t in data_sngl[0]]}\n')
    ext3.util.logg(f'data_meas    : len={len(data_meas)}, [(shp,val),..]={[(list(t.shape), t.view(-1)[:5]) for t in data_meas[0]]}\n')
    ext3.util.logg(f'================\n\n')

def set_model() -> None:
    global args # in.
    global net, criterion, best_acc, start_epoch # out.
    print('==> Setting model..')

    #===== 
    # set: net.
    if args.model not in ('res18', 'mblv2', 'shfv2', 'sqz'):
        assert(args.model_wm == 1.0)
    net_dict = { # model_str: (model_cls, model_args)
        'res18'  : (ResNet18,      [args.model_wm]),
        'mblv2'  : (MobileNetV2,   [args.model_wm]),
        'shfv2'  : (ShuffleNetV2,  [args.model_wm]),
        'sqz'    : (SqueezeNet,    [args.model_wm]),
    }
    net_info = net_dict[args.model]
    net = net_info[0](*net_info[1], num_classes=args.data_num_classes) # type: ignore
    #=====
    net = net.to(args.device)

    #===== 
    # set: net (for cuda).
    if args.data_parallel:
        net = torch.nn.DataParallel(net)
        # cudnn.benchmark = True
    if args.cuda >= -1: # use cuda.
        cudnn.benchmark = True

    # set: criterion.
    # criterion = nn.CrossEntropyLoss()
    criterion = ext3.nn.CrossEntropyLoss()
    #=====

    # set: best_acc, start_epoch.
    best_acc = 0.  # best test accuracy
    # start_epoch = -1  # start from epoch -1, precisely to measure fp err before start training.
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # # set: net, best_acc, start_epoch (for resume).
    # if args.resume:
    #     # Load checkpoint.
    #     print('==> Resuming from checkpoint..')
    #     assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    #     checkpoint = torch.load('./checkpoint/ckpt.pth')
    #     net.load_state_dict(checkpoint['net'])
    #     best_acc = checkpoint['acc']
    #     start_epoch = checkpoint['epoch']

def print_model() -> None:
    global net # in.

    # print: model info.
    mem_params = sum([param.nelement()*param.element_size() for param in net.parameters()])
    mem_bufs   = sum([buf  .nelement()*buf  .element_size() for buf   in net.buffers   ()])
    mem = mem_params + mem_bufs # in bytes
    ext3.util.logg(f'===== NET(basic) =====\n')
    ext3.util.logg(f'mem : mem_(tot,params,bufs)(MB)=({mem/(2**20):.3f},{mem_params/(2**20):.3f},{mem_bufs/(2**20):.3f})\n')
    ext3.util.logg(f'======================\n\n')
    # params_name = list(map(lambda v: v[0], net.named_parameters()))
    # ext3.util.logg(f'params : name={params_name}\n')
    # ext3.util.logg(f'net : {net}\n')
    # for name, modl in net.named_modules():
    #     ext3.util.logg(f'{name}: {modl}\n')

def set_opt() -> None:
    global args, net # in.
    global optimizer, scheduler, lscaler # out.
    print('==> Setting opt..')
    
    #===== 
    # set: optimizer, scheduler, lscaler.
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.mt, weight_decay=args.wd)
    if args.lr_schd == 'stp':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_steps[0], gamma=args.lr_gamma)
    elif args.lr_schd == 'mulstp':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    elif args.lr_schd == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.lr_tmax)
    else:
        raise NotImplemented
    _num_batches     = len(dataload_trn)
    _init_scale      = args.lsc_init
    _growth_interval = int(args.lsc_evry_nep * _num_batches) if args.lsc_evry_nep > 0 else -1
    lscaler = torch.cuda.amp.GradScaler(init_scale = _init_scale, growth_interval = _growth_interval,
                                        growth_factor=2.0, backoff_factor=0.5, enabled=True) # default args.
    #=====

def print_opt() -> None:
    global optimizer, scheduler # in.
    
    # print: opt.
    ext3.util.logg(f'===== OPT =====\n')
    ext3.util.logg(f'optimizer : {optimizer}\n')
    ext3.util.logg(f'scheduler : {type(scheduler)}:{scheduler.state_dict()}\n')
    ext3.util.logg(f'lscaler   : {type(lscaler)}:{lscaler.state_dict()}\n')
    ext3.util.logg(f'===============\n\n')

def set_more() -> None:
    global args, net, criterion, data_sngl # in.
    global _net, _criterion # out.
    print('==> Setting model(more)..')
    
    # do not update internal states. (required to make running mean of batchnorms unchanged.)
    net.eval() 

    # set: modls in ext3.
    ext3.core.EModlObjMgr.register(net, criterion)

    # set: emodl.info_ts.{tsptr,numel}, emodl.info_mdcur.time, emodl.info_md{prv,nxt}.
    # NOTE: must use a non-parallel (i.e., original sequential) version of model.
    ext3.core.EModlObjMgr.reset_info(True)
    ext3.core.EModlObjMgr.set_param_forward_pre()
    _inputs, _targets = data_sngl[0]
    _net       = ext3.util.torch_nonparallel_modl(net)
    _criterion = ext3.util.torch_nonparallel_modl(criterion)
    _loss      = _criterion(_net(_inputs), _targets)
    ext3.core.EModlObjMgr.set_param_backward_pos(1.0)
    ext3.core.EModlObjMgr.reset_info(False)

    # set: emodl.info_ts[*].numel.
    bs_cur, bs_fut = _inputs.shape[0], args.bs_tr
    ext3.core.EModlObjMgr.set_info_ts_numel(bs_cur, bs_fut)

    # set: emodl.info_mdcur.id.
    def new_id_grp_all(emodl: ext3.core.EModl) -> bool:
        """assign new id['grp_all'] for Input/Linear/Conv2d modls."""
        return isinstance(emodl, (ext3.nn.Input, ext3.nn.Conv2d, ext3.nn.Linear))
    ext3.core.EModlObjMgr.set_info_mdcur_id(new_id_grp_all)

    # allow updating internal states.
    net.train()

    # # temp.
    # nls_mb = ext3.map_dict(lambda v: v*4/(2**20), nls) # size(t.data) in MB.
    # nls_mb_s = ext3.sorted_by(nls_mb.items(), 1, True)
    # print(nls_mb_s, sum(nls_mb.values())*2)
    # param_nl = 0
    # for param in net.parameters():
    #     param_nl += param.numel()
    # print(param_nl * 4 / (2**20))

def print_more() -> None:
    global args # in.

    # NOTE: this is moved to print_pasn(...).
    #
    # # get: emodl's numel (proportional to args.bs_tr) by id['grp_all'].
    # numels_abs:      Dict[ext3.core.Id, int]
    # numels_rel:      Dict[ext3.core.Id, float]
    # numels_rel_sort: List[Tuple[ext3.core.Id, float]]
    # numels_abs      = ext3.core.EModlObjMgr.get_numels_by_id('grp_all')
    # numels_rel      = ext3.util.dict_ratio(numels_abs)
    # numels_rel_sort = ext3.util.list_sort(list(numels_rel.items()), [(1,True), (0,True)]) # sort first by ratio (dec) then by id (dec).
    #
    # # print: numels_rel_sort.
    # ext3.util.logg(f'===== NET(numel) ===== (for now, ignore params.)\n')
    # for id, r in numels_rel_sort:
    #     ext3.util.logg(f'id["grp_all"]', f'\t{id.val}\t',
    #                    f'numel(rel,%)',  f'\t{r*100:.3f}\t', '\n')
    # ext3.util.logg(f'======================\n\n')

def set_pasn() -> None:
    global args # in.
    global args # out.
    print('==> Setting pasn..')

    #----------------#
    # args.pa_def_dt #
    #----------------#
    # set: dtype_def, pasn_upd.
    dtype_def: List[ext3.core.Dtype]
    pasn_upd : ext3.core.Pasn

    # check: pa_def_dt must be (dt_fwd,) or (dt_fwd, dt_bwd).
    assert(1<= len(args.pa_def_dt) <= 2)
    dtype_def = [ext3.core.Dtype.from_str(s) for s in args.pa_def_dt]
    pasn_upd  = ext3.core.Pasn(ext3.core.EModlObjMgr.get_emodls_sort(), *dtype_def)

    # set: args.pa_def_dt_dict.
    args.pa_def_dt_dict = {
        ext3.core.Ttype.P : dtype_def[0],
        ext3.core.Ttype.X : dtype_def[0],
        ext3.core.Ttype.Y : dtype_def[0],
        ext3.core.Ttype.GP: dtype_def[-1],
        ext3.core.Ttype.GX: dtype_def[-1],
        ext3.core.Ttype.GY: dtype_def[-1],
    }

    #-----------------#
    # args.pa_upd_dts #
    #-----------------#
    # set: dtypes_upd.
    dtypes_upd: Dict[str, ext3.core.Dtype]

    if args.pa_upd_dts is not None:
        _dtypes_upd = [ext3.core.Dtype.from_str(v) for v in args.pa_upd_dts]
        dtypes_upd  = {'P' : _dtypes_upd[0],
                       'Y' : _dtypes_upd[1],
                       'GY': _dtypes_upd[2],
                       'GP': _dtypes_upd[3]}

    #-------------------------#
    # args.pa_upd_{schm,args} #
    #-------------------------#
    # set: pasn_upd.
    upd_idvals: Seq[int]
    upd_dtplan: Dict[str, ext3.core.DtypePlan]

    if args.pa_upd_schm is None:
        upd_idvals = []
        upd_dtplan = {}
    
    elif any(args.pa_upd_schm.startswith(schm) for schm in ('rand', 'topr', 'topk')):
        # set: upd_ids.
        upd_ids: Seq[ext3.core.Id]
        
        if args.pa_upd_schm == 'rand' or args.pa_upd_schm.startswith('topr'):
            # set: r_{min,max}.
            [r_min, r_max] = [float(v) for v in args.pa_upd_args]
            # set: upd_ids.
            trial = 1000 if args.pa_upd_schm == 'rand' else 1
            found = False
            for _ in range(trial):
                upd_ids, r_cur = ext3.core.EModlObjMgr.get_ids_chosen('grp_all', args.pa_upd_schm, r_min=r_min, rand_seed=args.pa_upd_seed)
                if r_min in (0.0, 1.0) or r_min <= r_cur <= r_max:
                    # accept: only when r_min <= r_cur <= r_max.
                    found = True; break
            # handle: if upd_ids is not found.
            if not found:
                upd_ids = []
                ext3.util.logg(f'*** No proper pasn_upd for args.pa_upd_args={args.pa_upd_args} is available. ***\n\n')
                exit(0)

        elif args.pa_upd_schm.startswith('topk'):
            # set: k.
            [k] = [int(v) for v in args.pa_upd_args]
            # set: upd_ids.
            upd_ids, _ = ext3.core.EModlObjMgr.get_ids_chosen('grp_all', args.pa_upd_schm, k=k)

        # set: upd_{idvals, dtplan}.
        upd_idvals = [id.val for id in upd_ids]
        upd_dtplan = { 'cur': {ext3.core.Ttype.Y : dtypes_upd['Y' ],
                               ext3.core.Ttype.GY: dtypes_upd['GY']} ,
                       'prv': {ext3.core.Ttype.P : dtypes_upd['P' ],
                               ext3.core.Ttype.GP: dtypes_upd['GP']} }
        
        # update: pasn_upd (all emodls with id['grp_all'] in upd_idvals).
        pasn_upd.update_by_id_grp_all('cur', 'id', upd_dtplan['cur'], upd_idvals)
        pasn_upd.update_by_id_grp_all('prv', 'id', upd_dtplan['prv'], upd_idvals)

    elif args.pa_upd_schm == 'std':
        # set: upd_{idvals, dtplan, emdtps}.
        upd_emdtps: Seq[Type[ext3.core.EModl]]
        upd_idvals = [-1] # =anything.
        upd_dtplan = { 'cur': {ext3.core.Ttype.P : dtypes_upd['P' ],
                               ext3.core.Ttype.GY: dtypes_upd['GY'],
                               ext3.core.Ttype.GP: dtypes_upd['GP']} ,
                       'nxt': {ext3.core.Ttype.Y : dtypes_upd['Y' ]} }
        upd_emdtps = [ext3.nn.Conv2d, ext3.nn.Linear,] # only gemm ops.
        # [
        #     ext3.nn.BatchNorm2d, ext3.nn.ReLU, ext3.nn.Cat, ext3.nn.MaxPool2d,
        #     ext3.nn.AvgPool2d, ext3.nn.LogSoftmax, ext3.nn.NLLLoss, ext3.nn.Input,
        # ]

        # update: pasn_upd (all emodls with emodltyp in upd_emdtps).
        pasn_upd.update_by_id_idv_mdl('cur', 'knd', upd_dtplan['cur'], upd_idvals, upd_emdtps)
        pasn_upd.update_by_id_idv_mdl('nxt', 'knd', upd_dtplan['nxt'], upd_idvals, upd_emdtps)

    # set: args.pa_upd_idvals
    args.pa_upd_idvals = upd_idvals # : Seq[int]

    #--------------------------#
    # args.pa_upd_no_{end,bwd} #
    #--------------------------#
    # set: pasn_upd.
    if args.pa_upd_no_end is True:
        # set: upd_dtplan.
        upd_dtplan = { 'cur': {ext3.core.Ttype.Y : dtype_def[0],
                               ext3.core.Ttype.GY: dtype_def[1]} ,
                       'prv': {ext3.core.Ttype.Y : dtype_def[0],
                               ext3.core.Ttype.P : dtype_def[0],
                               ext3.core.Ttype.GY: dtype_def[1],
                               ext3.core.Ttype.GP: dtype_def[1]} }
        # update: pasn_upd (first layer).
        pasn_upd.update_by_id_idv_all('cur', 'id', upd_dtplan['cur'], [0 ])
        pasn_upd.update_by_id_idv_all('prv', 'id', upd_dtplan['prv'], [0 ])

        # set: upd_dtplan.
        upd_dtplan = { 'cur': {ext3.core.Ttype.Y : dtype_def[0],
                               ext3.core.Ttype.P : dtype_def[0],
                               ext3.core.Ttype.GY: dtype_def[1],
                               ext3.core.Ttype.GP: dtype_def[1]} ,
                       'nxt': {ext3.core.Ttype.Y : dtype_def[0],
                               ext3.core.Ttype.GY: dtype_def[1]} }
        # update: pasn_upd (last group).
        pasn_upd.update_by_id_grp_all('cur', 'id', upd_dtplan['cur'], [-1])
        pasn_upd.update_by_id_grp_all('nxt', 'id', upd_dtplan['nxt'], [-1])

    if args.pa_upd_no_bwd is True:
        # set: upd_{idvals, dtplan}.
        upd_idvals = [-1] # =anything.
        upd_dtplan = { 'cur': {ext3.core.Ttype.GY: dtype_def[1],
                               ext3.core.Ttype.GP: dtype_def[1]} }

        # update: pasn_upd (all emodls).
        pasn_upd.update_by_id_grp_all('cur', 'knd', upd_dtplan['cur'], upd_idvals)

    #------------#
    # apply pasn #
    #------------#
    # set: model.
    set_model_prec(pasn_upd)
    set_model_prec_backup()

    #---------------#
    # get pasn stat #
    #---------------#
    # set: upd_ts_numel_ratio [= numel(ts in updated prec) / numel(ts in all prec) * 100].
    all_bit_num: int = ext3.core.EModlObjMgr.get_numbit(args.pa_def_dt_dict)
    upd_bit_num: int = ext3.core.EModlObjMgr.get_numbit(None)
    upd_ts_numel_percent: float = (all_bit_num - upd_bit_num) / all_bit_num * 2 * 100.
    
    #-----------#
    # save pasn #
    #-----------#
    # set: args.pa_{cst, upd, upd_stat}.
    args.pa_cst = {dtype: ext3.core.Pasn(ext3.core.EModlObjMgr.get_emodls_sort(), dtype)
                   for dtype in (ext3.core.FP32,)} # : Dict[ext3.core.Dtype, Pasn]
    args.pa_upd = pasn_upd # : ext3.core.Pasn
    args.pa_upd_stat = {'pasn_upd_r' : upd_ts_numel_percent,
                        'all_bit_num': all_bit_num} # : Dict[str, Union[float,int]]

def print_pasn() -> None:
    global args, net, criterion # in.

    # set: numels_rel_sort. (= emodl's numel for args.bs_tr, grouped by id['grp_all'])
    numels_abs:      Dict[ext3.core.Id, int]
    numels_rel:      Dict[ext3.core.Id, float]
    numels_rel_sort: List[Tuple[ext3.core.Id, float]]

    numels_abs      = ext3.core.EModlObjMgr.get_numels_by_id('grp_all')
    numels_rel      = ext3.util.dict_ratio(numels_abs)
    numels_rel_sort = ext3.util.list_sort(list(numels_rel.items()), [(1,True), (0,False)]) # sort first by ratio (dec) then by id (dec).

    # print: numels_rel_sort, args.pa_upd, emodl.info*.
    ext3.util.logg(f'===== NET(pasn) =====\n')
    ext3.util.logg(f'pasn_upd_stat: pasn_upd(r%)={args.pa_upd_stat["pasn_upd_r"]:.3f}\n')
    for id_cur, r_cur in numels_rel_sort:
        ext3.util.logg(f'id["grp_all"]', f'\t{id_cur.val}\t',
                       f'numel(rel,%)',  f'\t{r_cur * 100:.6f}\t',
                       f'upd_chosen',    f'\t{"O" if id_cur.val in args.pa_upd_idvals else "X"}\t', '\n')
    ext3.util.logg(f'-------------------------\n')
    ext3.util.logg(f'pasn_upd: (dtype=[Y,GY,P,GP,X,GX])\n')
    ext3.util.logg(f'{args.pa_upd}\n')
    ext3.util.logg(f'-------------------------\n')
    ext3.util.logg(f'model_info(dtype): (dtype=[Y,GY,P,GP,X,GX])\n')
    ext3.core.EModlObjMgr.print_info_dtype(ext3.util.logg)
    ext3.util.logg(f'-------------------------\n')
    ext3.util.logg(f'model_info(mdadj):\n')
    ext3.core.EModlObjMgr.print_info_mdadj(ext3.util.logg)
    ext3.util.logg(f'-------------------------\n')
    ext3.util.logg(f'model_info(nn.Module):\n')
    ext3.util.logg(f'{net}\n')
    ext3.util.logg(f'-------------------------\n')
    ext3.util.logg(f'criterion_info(nn.Module):\n')
    ext3.util.logg(f'{criterion}\n')
    ext3.util.logg(f'====================\n\n')

def set_model_prec(pasn: ext3.core.Pasn) -> None:
    # apply: pasn.
    ext3.core.EModlObjMgr.set_info_ts_dtype(pasn)
    ext3.core.EModlObjMgr.set_info_ts_rndmd()

def set_model_prec_backup() -> None:
    ext3.core.EModlObjMgr.set_dtype_rndmd_backup()

def set_model_prec_restore() -> None:
    ext3.core.EModlObjMgr.set_dtype_rndmd_restore()

def _comp_param_grad(inputs: TS, targets: TS, lscaler: Union[float, torch.cuda.amp.GradScaler]) -> Tuple[TS, TS]:
    global args, net, criterion # in.

    if isinstance(lscaler, float):
        # do: pre.
        ext3.core.EModlObjMgr.set_param_forward_pre()

        # set: outputs, loss, grads.
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        (loss * lscaler).backward()

        # do: post.
        ext3.core.EModlObjMgr.set_param_backward_pos(lscaler)
        
    elif isinstance(lscaler, torch.cuda.amp.GradScaler):
        # do: pre.
        ext3.core.EModlObjMgr.set_param_forward_pre()

        # set: outputs, loss, grads.
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        lscaler.scale(loss).backward()

        # do: post.
        ext3.core.EModlObjMgr.set_param_backward_pos(1.0)

    else: raise ValueError
    return outputs, loss

def _save_param_data(clone: bool) -> Dict[str, TS]:
    global net # in.

    # set: res.
    res: Dict[str, TS] = {}
    for name, param in net.named_parameters():
        res[name] = param.detach()
    if clone is True:
        for name in res:
            res[name] = res[name].clone()
    return res
    
def _save_param_grad(clone: bool) -> Dict[str, TS]:
    global net # in.

    # set: res.
    res: Dict[str, TS] = {}
    for name, param in net.named_parameters():
        res[name] = param.grad.detach()
    if clone is True:
        for name in res:
            res[name] = res[name].clone()
    return res
            
def meas_fperr() -> None:
    global args, net, optimizer, lscaler, data_meas # in.

    # helper funcs: __{save,load}_state_{bn,param_opt}.
    def __save_state_bn() -> None:
        # save: bn.momentum.
        # set bn.momentum to 0.0 not to update bn.state (i.e., running mean/variance).
        for emodl in ext3.core.EModlObjMgr.get_emodls():
            if isinstance(emodl, ext3.nn.BatchNorm2d):
                emodl.momentum_saved = emodl.momentum # type: ignore
                emodl.momentum = 0.0 # type: ignore

    def __load_state_bn() -> None:
        # load: bn.momentum.
        for emodl in ext3.core.EModlObjMgr.get_emodls():
            if isinstance(emodl, ext3.nn.BatchNorm2d):
                emodl.momentum = emodl.momentum_saved # type: ignore

    def __save_state_param_opt() -> None:
        # save: net.param, optimizer.state (e.g., momentum).
        ext3.core.EModlObjMgr.set_param_backup()
        optimizer.state_dict_saved = copy.deepcopy(optimizer.state_dict()) # type: ignore

    def __load_state_param_opt() -> None:
        # load: net.param, optimizer.state.
        ext3.core.EModlObjMgr.set_param_restore()
        optimizer.load_state_dict(optimizer.state_dict_saved) # type: ignore
    
    # set: net.
    __save_state_bn()
    net.train()

    # set: stat.
    stat: Dict[str, List[np.float64]]
    stat = { 
        # DEF:
        # - n = {1, 2, ..., num_batches_meas}. f = inf.
        # - p = params. p is fixed over n.
        # - gp_n = (grad p)(x_n).
        # - cp_n = p' - p, where p' = updated param after processing x_n (e.g., p' = p-lr*gp_n).
        #
        # pgrd: size, abs_err, rel_err.
        'pgrd_sz(l1)': [], # avg_n (|gp_n|_1 / dim(p)    ).
        'pgrd_ea(l1)': [], # avg_n (|err(gp_n)|_1 / dim(p)    ).
        'pgrd_er(l1)': [], # avg_n (|err(gp_n)|_1 / |gp_n|_1).
        # pchg: size, abs_err, rel_err.
        'pchg_sz(l1)': [], # avg_n (|cp_n|_1 / dim(p)    ).
        'pchg_ea(l1)': [], # avg_n (|err(cp_n)|_1 / dim(p)    ).
        'pchg_er(l1)': [], # avg_n (|err(cp_n)|_1 / |cp_n|_1).
        # 'pgrd_sz(l2)': [], # avg_n (|gp_n|_2 / dim(p)^0.5).
        # 'pgrd_sz(lf)': [], # avg_n (|gp_n|_f             ).
        # 'pgrd_ea(l2)': [], # avg_n (|err(gp_n)|_2 / dim(p)^0.5).
        # 'pgrd_ea(lf)': [], # avg_n (|err(gp_n)|_f             ).
        # 'pgrd_er(l2)': [], # avg_n (|err(gp_n)|_2 / |gp_n|_2).
        # 'pgrd_er(lf)': [], # avg_n (|err(gp_n)|_f / |gp_n|_f).
        # 'pchg_sz(l2)': [], # avg_n (|cp_n|_2 / dim(p)^0.5).
        # 'pchg_sz(lf)': [], # avg_n (|cp_n|_f             ).
        # 'pchg_ea(l2)': [], # avg_n (|err(cp_n)|_2 / dim(p)^0.5).
        # 'pchg_ea(lf)': [], # avg_n (|err(cp_n)|_f             ).
        # 'pchg_er(l2)': [], # avg_n (|err(cp_n)|_2 / |cp_n|_2).
        # 'pchg_er(lf)': [], # avg_n (|err(cp_n)|_f / |cp_n|_f).
    }

    # set: stat.
    for data in data_meas:
        # set: inputs, targets, p{grd,chg}.
        # inputs, targets = [t.to(args.device) for t in data]
        inputs, targets = data
        pgrd, pchg = {}, {}

        # NOTE: torch.optim._function.sgd_get_dp().
        # - src: implemented in /scratch2/[user]/python3/lib/python3.8/site-packages/torch/optim/_functional.py
        # - def: dp means dp in p' = p - lr*dp. That is, change in p after opt.step() divided by -lr.

        # measure: pasn. (p{grd,chg}['pasn'].)
        # - opt: pre.
        __save_state_param_opt()
        set_model_prec(args.pa_upd)
        # - opt: main.
        optimizer.zero_grad()
        _comp_param_grad(inputs, targets, lscaler.get_scale())
        optimizer.step()
        # - opt: post.
        pgrd['pasn'] = list(_save_param_grad(True).values())
        pchg['pasn'] = optim._functional.sgd_get_dp() # type: ignore # see above NOTE.
        __load_state_param_opt()

        # measure: fp32. (p{grd,chg}['fp32'].)
        # - opt: pre.
        __save_state_param_opt()
        set_model_prec(args.pa_cst[ext3.core.FP32])
        # - opt: main.
        optimizer.zero_grad()
        _comp_param_grad(inputs, targets, lscaler.get_scale())
        optimizer.step()
        # - opt: post.
        pgrd['fp32'] = list(_save_param_grad(True).values())
        pchg['fp32'] = optim._functional.sgd_get_dp() # type: ignore # see above NOTE.
        __load_state_param_opt()
            
        # set: p{grd,chg}_err.
        pgrd_err = [t1-t2 for t1,t2 in zip(pgrd['pasn'], pgrd['fp32'])]
        pchg_err = [t1-t2 for t1,t2 in zip(pchg['pasn'], pchg['fp32'])]
        
        # save: res(cur).
        p_dim = sum(t.numel() for t in pgrd['pasn'])
        lr = np.float64(optimizer.param_groups[0]['lr'])
        # - pgrd.
        pgrd_l1     = np.float64(ext3.util.torch_l1_norm(pgrd['fp32']).item())
        pgrd_err_l1 = np.float64(ext3.util.torch_l1_norm(pgrd_err    ).item())
        # - pchg.
        pchg_l1     = np.float64(ext3.util.torch_l1_norm(pchg['fp32']).item())
        pchg_err_l1 = np.float64(ext3.util.torch_l1_norm(pchg_err    ).item())
        # - pgrd.
        stat['pgrd_sz(l1)'].append(pgrd_l1 / p_dim)
        stat['pgrd_ea(l1)'].append(pgrd_err_l1 / p_dim)
        stat['pgrd_er(l1)'].append(pgrd_err_l1 / pgrd_l1)
        # - pchg.
        stat['pchg_sz(l1)'].append(lr * pchg_l1 / p_dim)
        stat['pchg_ea(l1)'].append(lr * pchg_err_l1 / p_dim)
        stat['pchg_er(l1)'].append(pchg_err_l1 / pchg_l1)
        # pgrd_l2     = np.float64(ext3.util.torch_l2_norm(pgrd['fp32']).item())
        # pgrd_lf     = np.float64(ext3.util.torch_lf_norm(pgrd['fp32']).item())
        # pgrd_err_l2 = np.float64(ext3.util.torch_l2_norm(pgrd_err    ).item())
        # pgrd_err_lf = np.float64(ext3.util.torch_lf_norm(pgrd_err    ).item())
        # pchg_l2     = np.float64(ext3.util.torch_l2_norm(pchg['fp32']).item())
        # pchg_lf     = np.float64(ext3.util.torch_lf_norm(pchg['fp32']).item())
        # pchg_err_l2 = np.float64(ext3.util.torch_l2_norm(pchg_err    ).item())
        # pchg_err_lf = np.float64(ext3.util.torch_lf_norm(pchg_err    ).item())
        # stat['pgrd_sz(l2)'].append(pgrd_l2 / np.sqrt(p_dim))
        # stat['pgrd_sz(lf)'].append(pgrd_lf)
        # stat['pgrd_ea(l2)'].append(pgrd_err_l2 / np.sqrt(p_dim))
        # stat['pgrd_ea(lf)'].append(pgrd_err_lf)
        # stat['pgrd_er(l2)'].append(pgrd_err_l2 / pgrd_l2)
        # stat['pgrd_er(lf)'].append(pgrd_err_lf / pgrd_lf)
        # stat['pchg_sz(l2)'].append(lr * pchg_l2 / np.sqrt(p_dim))
        # stat['pchg_sz(lf)'].append(lr * pchg_lf)
        # stat['pchg_ea(l2)'].append(lr * pchg_err_l2 / np.sqrt(p_dim))
        # stat['pchg_ea(lf)'].append(lr * pchg_err_lf)
        # stat['pchg_er(l2)'].append(pchg_err_l2 / pchg_l2)
        # stat['pchg_er(lf)'].append(pchg_err_lf / pchg_lf)

    # print: res(all).
    for v_name in stat:
        v_l = stat[v_name]
        v_l_one, v_l_avg, n = v_l[0], np.mean(v_l), len(v_l)
        ext3.util.logg(f'{v_name}[n=1]'  , f'\t{v_l_one:.3e}\t')
        ext3.util.logg(f'{v_name}[n={n}]', f'\t{v_l_avg:.3e}\t')

    # set: net.
    __load_state_bn()

def train(epoch: int) -> None:
    global args, dataload_trn, net, criterion, optimizer, lscaler # in.

    net.train()
    train_loss = 0.0
    correct = 0
    total = 0

    #===== 
    # set: dataload_cur, lscale_{min,max}, param_{old,new}_data, stat.
    dataload_cur: Union[list, torch.utils.data.DataLoader]
    lsc_old: float
    lsc_lg_min: float
    lsc_lg_max: float
    lsc_infs_tot: int
    pasn_upd_r_tot: float
    # param_old_data: List[TS]
    # param_new_data: List[TS]
    # stat   : Dict[str, np.float64]
    
    if epoch >= 0: dataload_cur = dataload_trn
    else:          dataload_cur = []
    lsc_old = lscaler.get_scale()
    lsc_lg_min = +999999.
    lsc_lg_max = -999999.
    lsc_infs_tot = 0
    pasn_upd_r_tot = 0.0
    # param_old_data = []
    # param_new_data = list(_save_param_data(True).values())
    # stat       = { 
    #     # DEF:
    #     # - t = {1, 2, ..., num_batches_train}. f = inf.
    #     # - p_t = param after processing x_1, ..., x_{t-1}. p changes over t.
    #     # - gp_t = (grad p_t)(x_t).
    #     # - cp_t = p_t - p_{t-1}.
    #     #
    #     # pval.
    #     'pval_sz(l1)': np.float64(0.0), # avg_t (| p_t|_1 / dim(p_t)    ).
    #     # pgrd.
    #     'pgrd_sz(l1)': np.float64(0.0), # avg_t (|gp_t|_1 / dim(p_t)    ).
    #     # pchg.
    #     'pchg_sz(l1)': np.float64(0.0), # avg_t (|cp_t|_1 / dim(p_t)    ).
    #     # 'pval_sz(l2)': np.float64(0.0), # avg_t (| p_t|_2 / dim(p_t)^0.5).
    #     # 'pval_sz(lf)': np.float64(0.0), # avg_t (| p_t|_f               ).
    #     # 'pgrd_sz(l2)': np.float64(0.0), # avg_t (|gp_t|_2 / dim(p_t)^0.5).
    #     # 'pgrd_sz(lf)': np.float64(0.0), # avg_t (|gp_t|_f               ).
    #     # 'pchg_sz(l2)': np.float64(0.0), # avg_t (|cp_t|_2 / dim(p_t)^0.5).
    #     # 'pchg_sz(lf)': np.float64(0.0), # avg_t (|cp_t|_f               ).
    # }
    #=====
    
    with contextlib.nullcontext(): # nop---to match with below test(..).
        # for batch_idx, (inputs, targets) in enumerate(dataload_trn):
        for batch_idx, data in enumerate(dataload_cur):
            #===== 
            # inputs, targets = inputs.to(args.device), targets.to(args.device)
            # optimizer.zero_grad()
            # outputs = net(inputs)
            # loss = criterion(outputs, targets)
            # loss.backward()
            # optimizer.step()
            #
            # train_loss  += loss.item()
            # _, predicted = outputs.max(1)
            # total   += targets.size(0)
            # correct += predicted.eq(targets).sum().item()

            # opt: pre. (inputs, targets, param_old_data.)
            inputs, targets = [t.to(args.device) for t in data]
            # del param_old_data; param_old_data = param_new_data

            # opt: main. (outputs, loss, model-param, optimizer, lscaler.)
            optimizer.zero_grad() # set grad to 0.
            outputs, loss = _comp_param_grad(inputs, targets, lscaler) # do forward(output,loss) + backward(grad).
            lscaler.step(optimizer) # update grad.

            # lscaler: update. (lscaler, lsc_{cur,old}, lsc_infs_cur.)
            lscaler.update() # update scaler.
            lsc_cur = lscaler.get_scale() #= cur loss scale.
            lsc_infs_cur = int(lsc_cur < lsc_old) #= 1[cur grad contains infs].
            if args.lsc_evry_nep < 0: # do not change loss scale.
                lscaler.update(new_scale = args.lsc_init) 
                lsc_cur = args.lsc_init
            lsc_old = lsc_cur #= old loss scale (for the next iter).

            # opt: post. (param_new_{data,grad}.)
            # - set: undovrs / handle: overflows.
            undovrs = ext3.core.EModlObjMgr.get_undovrs()
            if args.pa_ovr_thrs >= 0:
                flag = np.concatenate([ undovrs[ttype][:,1] >= args.pa_ovr_thrs  # NOTE: 0=undf_ratio, 1=ovrf_ratio.
                                        for ttype in (ext3.core.Ttype.P,         # MATCH WITH: emodlobjmgr.inc_ts_prec().
                                                      ext3.core.Ttype.Y) ])
                ext3.core.EModlObjMgr.inc_ts_prec(flag, args.pa_def_dt_dict)
            # - set: undfrs, ovrfrs.
            undfrs, ovrfrs = {}, {}
            for ttype, undovr in undovrs.items():
                undfrs[ttype] = undovr[:,0]
                ovrfrs[ttype] = undovr[:,1]
            # param_new_data = list(_save_param_data(True).values())
            # param_new_grad = list(_save_param_grad(True).values())

            # save: res(cur). (train_loss, total, correct.)
            train_loss   += loss.item()
            _, predicted = outputs.max(1)
            total        += targets.size(0)
            correct      += predicted.eq(targets).sum().item() # type: ignore

            # save. (lsc_{lg_min,lg_max,infs_tot}, pasn_upd_r_tot.)
            all_bit_num = args.pa_upd_stat['all_bit_num']
            upd_bit_num = ext3.core.EModlObjMgr.get_numbit(None)
            pasn_upd_r  = (all_bit_num - upd_bit_num) / all_bit_num * 2 * 100.
            lsc_lg_min     = min(np.log2(lsc_cur), lsc_lg_min)
            lsc_lg_max     = max(np.log2(lsc_cur), lsc_lg_max)
            lsc_infs_tot   += lsc_infs_cur
            pasn_upd_r_tot += pasn_upd_r

            # # save: stat(cur). (stat.)
            # param_chg_data = [p_new - p_old for p_new, p_old in zip(param_new_data, param_old_data)]
            # stat['pval_sz(l1)'] += np.float64(ext3.util.torch_l1_mean(param_new_data).item())
            # stat['pgrd_sz(l1)'] += np.float64(ext3.util.torch_l1_mean(param_new_grad).item())
            # stat['pchg_sz(l1)'] += np.float64(ext3.util.torch_l1_mean(param_chg_data).item())
            # # stat['pval_sz(l2)'] += np.float64(ext3.util.torch_l2_mean(param_new_data).item())
            # # stat['pval_sz(lf)'] += np.float64(ext3.util.torch_lf_mean(param_new_data).item())
            # # stat['pgrd_sz(l2)'] += np.float64(ext3.util.torch_l2_mean(param_new_grad).item())
            # # stat['pgrd_sz(lf)'] += np.float64(ext3.util.torch_lf_mean(param_new_grad).item())
            # # stat['pchg_sz(l2)'] += np.float64(ext3.util.torch_l2_mean(param_chg_data).item())
            # # stat['pchg_sz(lf)'] += np.float64(ext3.util.torch_lf_mean(param_chg_data).item())

            # save more.
            # - undovr_res.
            if args.logg_undovr_freq > 0 and (batch_idx+1) % args.logg_undovr_freq == 0:
                undovr_res = f'ep: {epoch} it: {batch_idx} '
                for ratios_name, ratios in {'und': undfrs, 'ovr': ovrfrs}.items():
                    for ttype, ratio in ratios.items():
                        undovr_res += f'{ratios_name}_{str(ttype)[6:]}: {ratio} '
                ext3.util.logg_full('undovr', f'{undovr_res}\n')

            # print: res(cur).
            if args.quiet is False:
                progress_bar(batch_idx, len(dataload_trn), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
            #=====

    #===== 
    # save: res(all). (lr, all_{loss,acc,pasn_upd_r}.)
    if epoch >= 0:
        lr = optimizer.param_groups[0]['lr']
        all_loss, all_acc = train_loss/(batch_idx+1), 100.*correct/total
        all_pasn_upd_r = pasn_upd_r_tot/(batch_idx+1)
        # for v_name in stat:
        #     stat[v_name] = stat[v_name] / (batch_idx+1)
    else:
        lr = 0.0
        all_loss, all_acc = 0.0, 0.0
        all_pasn_upd_r = 0.0
        # for v_name in stat:
        #     stat[v_name] = np.float64(0.0)

    # print: res(all). (lr, all_{loss,acc}, lscale_{min,max,infs_tot}, pasn_upd_r, stat.)
    log_msg = '\t '.join([
        'res:',
        'epoch',        f'{epoch}',
        'lr',           f'{lr:.3e}',
        'loss(train)',  f'{all_loss:.3e}',
        'acc(train,%)', f'{all_acc:.3f}',
        'lsc(min,lg)',  f'{lsc_lg_min:.1f}',
        'lsc(max,lg)',  f'{lsc_lg_max:.1f}',
        'lsc(#infgrd)', f'{lsc_infs_tot:.0f}',
        'pasn_upd(r%)', f'{all_pasn_upd_r:.3f}',
        ''])
    ext3.util.logg(log_msg)
    # for v_name in stat:
    #     ext3.util.logg(f'{v_name}[all]', f'\t{stat[v_name]:.3e}\t')
    #=====

#===== 
# def test(epoch: int) -> None:
def test(epoch: int, log_suffix="") -> None:
#=====
    global args, dataload_tst, net, criterion, best_acc # in.
    global best_acc # out.

    net.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    #===== 
    # set: dataload_cur.
    dataload_cur: Union[list, torch.utils.data.DataLoader]
    if epoch >= 0: dataload_cur = dataload_tst
    else:          dataload_cur = []
    #=====

    with torch.no_grad():
        # for batch_idx, (inputs, targets) in enumerate(dataload_tst):
        for batch_idx, data in enumerate(dataload_cur):
            #===== 
            # inputs, targets = inputs.to(args.device), targets.to(args.device)

            # eval: pre. (inputs, targets.)
            inputs, targets = [t.to(args.device) for t in data]
            #=====

            # eval: main. (outputs, loss.)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            # save: res(cur).
            test_loss   += loss.item()
            _, predicted = outputs.max(1)
            total  += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            #===== 
            # print: res(cur).
            if args.quiet is False:
                progress_bar(batch_idx, len(dataload_tst), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            #=====

    #===== 
    # save: res(all).
    if epoch >= 0:
        all_loss, all_acc = test_loss/(batch_idx+1), 100.*correct/total
    else:
        all_loss, all_acc = 0.0, 0.0

    # print: res(all).
    log_msg = '\t '.join([
        f'loss(test_{log_suffix})',   f'{all_loss:.3e}',
        f'acc1(test_{log_suffix},%)', f'{all_acc:.3f}',
        ''])
    ext3.util.logg(log_msg)

    # # save: checkpoint.
    # acc = 100.*correct/total
    # if acc > best_acc:
    #     print('Saving..')
    #     state = {'net': net.state_dict(), 'acc': acc, 'epoch': epoch}
    #     if not os.path.isdir('checkpoint'):
    #         os.mkdir('checkpoint')
    #     torch.save(state, './checkpoint/ckpt.pth')
    #     best_acc = acc
    #=====

def main() -> None:
    global args, start_epoch # in.
    
    #------#
    # init #
    #------#
    # # set: do_exit.
    # global do_exit; do_exit = False

    # set, print.
    set_args  ()  # -> args.
    print_args()  # args -> .

    set_data  ()  # args -> dataload_{trn,tst}, data_{meas,sngl}.
    print_data()  # dataload_{trn,tst}, data_{meas,sngl} -> .

    set_model  () # args -> net, criterion, best_acc, start_epoch.
    print_model() # net -> .

    set_opt  ()   # args, net -> optimizer, scheduler.
    print_opt()   # optimizer, scheduler -> .

    set_more  ()  # args, net, criterion -> _net, _criterion.
    print_more()  # args -> .

    set_pasn  ()  # args -> args.
    print_pasn()  # args -> .

    # # check: do_exit.
    # if do_exit is True: 
    #     ext3.util.logg(f'*** No proper pasn_upd for args.pa_upd_args={args.pa_upd_args} is available. ***\n\n')
    #     exit(0)

    #-----------#
    # main loop #
    #-----------#
    print("==> Start training..")
    start_time = time.time()
    epoch_time = time.time()
    inc_ts_prec_cnt = 0
    np.set_printoptions(linewidth=sys.maxsize, threshold=sys.maxsize,
                        formatter={'float': lambda x: format(x, '.2e') if x!=0. else '0'})

    for epoch in range(start_epoch, args.ep): # start_epoch -> .
        # train(epoch)
        # test(epoch)
        # scheduler.step()

        # pre.
        print(f'\nEpoch: {epoch}')

        # train.
        set_model_prec_restore()
        train(epoch) # args, dataload_trn, net, criterion, optimizer -> .
        if epoch >= 0: scheduler.step()
        set_model_prec_backup()

        # test (cur_prec).
        set_model_prec_restore()
        test(epoch, log_suffix='cur') # args, testloaader, net, criterion, best_acc, start_time -> best_acc.
        set_model_prec_restore()

        # test (all_FP32).
        set_model_prec(args.pa_cst[ext3.core.FP32])
        test(epoch, log_suffix='FP32') # args, testloaader, net, criterion, best_acc, start_time -> best_acc.

        # measure err.
        if not args.meas_no_fperr:
            meas_fperr() # ? -> .

        # sync: inc_ts_prec.
        # - sync: inc_ts_prec_flag.
        inc_ts_prec_flag = ext3.core.EModlObjMgr.get_inc_ts_prec_flag()
        # - update: inc_ts_prec_cnt.
        inc_ts_prec_cnt += int(sum(inc_ts_prec_flag))

        # log.
        log_msg = '\t '.join([
            '#inc_ts_prec',  f'{inc_ts_prec_cnt}', 
            'time(epoch,m)', f'{(time.time()-epoch_time)/60:.2f}',
            'time(tot,h)',   f'{(time.time()-start_time)/3600:.3f}',
            ''])
        ext3.util.logg(log_msg)
        ext3.util.logg('\n')
        epoch_time = time.time()

main()
