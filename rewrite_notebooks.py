import json
import os
import sys

def find_and_replace_cell(nb, keyword, new_source):
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and keyword in ''.join(cell['source']):
            cell['source'] = [line + '\n' for line in new_source.split('\n')][:-1]
            return True
    return False

def sanitize_notebook_unicode(nb):
    """Sanitize all code cells to remove characters that fail to encode in Windows CP1252."""
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = []
            for line in cell['source']:
                line = line.replace('✓', '[OK]')
                line = line.replace('α', 'alpha')
                line = line.replace('σ', 'sigma')
                line = line.replace('→', '->')
                source.append(line)
            cell['source'] = source

def rewrite_tf32_notebook():
    with open('vgg16_cifar10_tf32_fp8.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # 1. Imports
    imports_source = '''# ============================================================
# Cell 2: Imports & Environment Check
# ============================================================

import os, sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# ---- Add project root to sys.path ----
PROJECT_ROOT = os.path.abspath('.')
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
print(f'[INFO] Project root: {PROJECT_ROOT}')

# ---- Import ext3 modules ----
from ext3.nn import flatten
from ext3.nn.nn_native import (
    NativeConv2d, NativeLinear, NativeBatchNorm2d,
    NativeReLU, NativeMaxPool2d, NativeAdaptiveAvgPool2d,
    NativeDropout, reset_fp8_manager
)
from ext3.core.include.native_precision import (
    NativePrecisionMode, check_fp8_support, enable_tf32, FP8Config
)
from ext3.core.emodlobj import EModlObjMgr
from ext3.core.include.dtype import Dtype, FP32
from ext3.core.include.pasn import Pasn
from ext3.core.include.ttype import Ttype
from ext3.util.apa_manager import (
    assign_precision, APAStabilityMonitor, StabilityEvent
)

# Environment Check
print(f"\\n=== Environment Check ===")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name       : {torch.cuda.get_device_name(0)}")
print(f"FP8 Supported  : {check_fp8_support()}")
'''
    find_and_replace_cell(nb, 'Cell 2: Imports', imports_source)

    # 2. Config
    config_source = '''# ============================================================
# Cell 3: Global Configuration
# ============================================================

enable_tf32()

CONFIG = {
    'batch_size': 128,
    'epochs': 50,
    'lr': 0.01,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'num_workers': 0,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # --- Native APA (Adaptive Precision Assignment) Config ---
    'pa_upd_schm': 'topr_dec',    # Demote layer terbesar lebih dulu
    'pa_upd_rmin': 0.3,           # Target 30% dari total parameter
    'pa_upd_rmax': 0.4,           
}

print("\\n=== Training Configuration ===")
for k, v in CONFIG.items():
    print(f"  {k:20s}: {v}")
'''
    find_and_replace_cell(nb, 'Cell 3: Global Configuration', config_source)

    # 2b. VGG16Native definition - replace torch.flatten with flatten from ext3.nn
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and 'Cell 4: VGG16 Model Definition' in ''.join(cell['source']):
            source = ''.join(cell['source'])
            source = source.replace('torch.flatten(x, 1)', 'flatten(x, 1)')
            cell['source'] = [line + '\n' for line in source.split('\n')][:-1]
            break

    # 3. Precision Assignment
    setup_source = '''# ============================================================
# Cell 5: Precision Assignment Setup
# ============================================================

# APA v2.0 — imported via ext3.nn imports cell above
print("APA v2.0: assign_precision + APAStabilityMonitor imported [OK]")
'''
    find_and_replace_cell(nb, 'Cell 5: Precision Assignment', setup_source)

    # 4. Train Epoch (with Promotion)
    train_source = '''# ============================================================
# Cell 8: Training Function (APA v2.0 — EMA-Gated Monitor)
# ============================================================

def train_epoch(model, loader, criterion, optimizer, device, monitor):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    epoch_events = {"stable": 0, "nan": 0, "spike": 0, "overflow": 0}
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # --- APA v2.0: EMA-Gated Stability Monitor ---
        # Reuse loss.item() for both logging AND monitoring (zero extra sync)
        loss_val = loss.item()
        event = monitor.step(loss_val, model, FP32)
        
        if event == StabilityEvent.LOSS_NAN:
            epoch_events["nan"] += 1
        elif event == StabilityEvent.LOSS_SPIKE:
            epoch_events["spike"] += 1
        elif event == StabilityEvent.GRADIENT_OVERFLOW:
            epoch_events["overflow"] += 1
        
        total_loss += loss_val * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    # Map promotions
    flags = EModlObjMgr.get_inc_ts_prec_flag()
    promoted_details = []
    if len(flags) > 0 and sum(flags) > 0:
        cnt = -1
        mapping = {}
        from ext3.core.include.ttype import Ttype
        for ttype in (Ttype.P, Ttype.Y):
            for emodl in EModlObjMgr.get_emodls_sort():
                if ttype in emodl.info_ts and hasattr(emodl.info_ts[ttype], 'undovr'):
                    for tsind, _ in enumerate(emodl.info_ts[ttype].undovr):
                        cnt += 1
                        mapping[cnt] = (ttype, emodl, tsind)
        for idx, f in enumerate(flags):
            if f > 0.0:
                tt = mapping.get(idx)
                if tt:
                    ttype, emodl, tsind = tt
                    gid = emodl.info_mdcur.id['grp_all'].val
                    promoted_details.append({
                        'group_id': gid,
                        'layer_name': type(emodl).__name__,
                        'ttype': 'Parameter' if ttype == Ttype.P else 'Activation',
                        'index': tsind
                    })
    
    total_promotions = len(promoted_details)
    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy, total_promotions, promoted_details, epoch_events

print("train_epoch() defined [OK] [APA v2.0: EMA-Gated]")
'''
    find_and_replace_cell(nb, 'Cell 8: Training Function', train_source)

    # 5. Main Loop
    main_source = '''# ============================================================
# Cell 10: Main Training Loop (APA v2.0)
# ============================================================

model = VGG16Native(num_classes=10).to(CONFIG['device'])

reset_fp8_manager()
pasn_manager = assign_precision(model, CONFIG)

# --- Grouping & Demotion Analysis ---
group_stats = {}
total_params_all = 0
total_params_fp8 = 0

for emodl in EModlObjMgr.get_emodls_sort():
    gid = emodl.info_mdcur.id['grp_all'].val
    if gid not in group_stats:
        group_stats[gid] = {
            'layers': [],
            'params': 0,
            'mode_y': 'unknown',
            'mode_p': 'unknown'
        }
    
    n_params = sum(p.numel() for p in emodl.parameters())
    total_params_all += n_params
    group_stats[gid]['params'] += n_params
    group_stats[gid]['layers'].append(type(emodl).__name__)
    
    dtype_y = (emodl.info_ts[Ttype.Y].dtype[0]
               if hasattr(emodl.info_ts[Ttype.Y], 'dtype')
               and len(emodl.info_ts[Ttype.Y].dtype) > 0
               else FP32)
    dtype_p = (emodl.info_ts[Ttype.P].dtype[0]
               if hasattr(emodl.info_ts[Ttype.P], 'dtype')
               and len(emodl.info_ts[Ttype.P].dtype) > 0
               else FP32)
    mode_y = dtype_y.to_native().mode_name
    mode_p = dtype_p.to_native().mode_name
    group_stats[gid]['mode_y'] = mode_y
    group_stats[gid]['mode_p'] = mode_p
    
    if mode_p == 'low_fp8':
        total_params_fp8 += n_params

print("\\n" + "="*80)
print("ANALISIS TENSOR GROUPING & DEMOTION")
print("="*80)
print(f"Total Parameter Model: {total_params_all:,}")
print(f"Total Parameter FP8  : {total_params_fp8:,} ({total_params_fp8/total_params_all*100:.2f}%)")
print("-"*80)
print(f"{'Group ID':<10} | {'Jumlah Layer':<12} | {'Total Params':<15} | {'Activation (Y)':<15} | {'Parameter (P)':<15}")
print("-"*80)
for gid, stats in sorted(group_stats.items()):
    print(f"{gid:<10d} | {len(stats['layers']):<12d} | {stats['params']:<15,d} | {stats['mode_y']:<15s} | {stats['mode_p']:<15s}")
print("="*80)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=CONFIG['lr'], 
                      momentum=CONFIG['momentum'], weight_decay=CONFIG['weight_decay'])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])

# --- APA v2.0: Initialize EMA-Gated Stability Monitor ---
monitor = APAStabilityMonitor(
    ema_alpha=0.1,            # EMA smoothing (alpha)
    variance_threshold=2.0,   # Spike sensitivity in sigma (sigma)
    ovr_thrs=0.0,             # Any overflow triggers promotion
    warmup_steps=10,          # EMA calibration period
)
print(f"APA v2.0 Monitor initialized: alpha={0.1}, sigma_thresh={2.0}, warmup={10}")

history = {
    'train_loss': [],
    'train_acc': [],
    'test_acc': [],
    'vram': [],
    'lr': [],
    'promotions': [],
    'spike_events': [],
    'promotion_details': [],
}

print("\\n" + "=" * 105)
print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Test Acc':>8} | {'Promoted':>8} | {'Spikes':>6} | {'GPU Syncs':>9} | {'VRAM MB':>8} | {'Time':>6} | {'LR':>8}")
print("-" * 105)

best_acc = 0.0
cumulative_promotions = 0

for epoch in range(1, CONFIG['epochs'] + 1):
    start_time = time.time()
    current_lr = optimizer.param_groups[0]['lr']
    
    train_loss, train_acc, promotions, promoted_details, events = train_epoch(
        model, trainloader, criterion, optimizer, CONFIG['device'], monitor
    )
    _, test_acc = evaluate(model, testloader, criterion, CONFIG['device'])
    scheduler.step()
    
    epoch_time = time.time() - start_time
    vram = get_vram_mb()
    cumulative_promotions += promotions
    stats = monitor.get_stats()
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['test_acc'].append(test_acc)
    history['vram'].append(vram)
    history['lr'].append(current_lr)
    history['promotions'].append(cumulative_promotions)
    history['spike_events'].append(stats['spike_events'])
    history['promotion_details'].append(promoted_details)
    
    best_marker = " *" if test_acc > best_acc else ""
    best_acc = max(best_acc, test_acc)
    
    print(f"{epoch:5d} | {train_loss:10.4f} | {train_acc:8.2f}% | {test_acc:7.2f}% | {promotions:5d} -> | {stats['spike_events']:6d} | {stats['tier2_checks']:9d} | {vram:7.1f}M | {epoch_time:5.1f}s | {current_lr:.6f}{best_marker}")

print("=" * 105)
print(f"Training Complete! Best Test Accuracy: {best_acc:.2f}%")
print(f"Total layers promoted to TF32/FP32: {cumulative_promotions}")
print(f"APA v2.0 Efficiency: {stats['sync_efficiency']}")
'''
    find_and_replace_cell(nb, 'Main Training Loop', main_source)

    # 6. Visualization
    viz_source = '''# ============================================================
# Cell 11: Visualization (Termasuk Promotion Plot & List)
# ============================================================

plt.style.use('ggplot')
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 5))

# 1. Loss Curve
ax1.plot(history['train_loss'], color='#E24A33', linewidth=2, marker='o', markersize=4)
ax1.set_title('Training Loss', fontsize=14)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Cross Entropy', fontsize=12)

# 2. Accuracy Curve
ax2.plot(history['train_acc'], label='Train Acc', color='#348ABD', linewidth=2)
ax2.plot(history['test_acc'], label='Test Acc', color='#988ED5', linewidth=2, marker='s', markersize=4)
ax2.set_title('Accuracy', fontsize=14)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.legend()

# 3. VRAM Usage
ax3.plot(history['vram'], color='#8EBA42', linewidth=2, fillstyle='bottom')
ax3.fill_between(range(len(history['vram'])), history['vram'], alpha=0.3, color='#8EBA42')
ax3.set_title('VRAM Allocation', fontsize=14)
ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('VRAM (MB)', fontsize=12)

# 4. Cumulative Promotions
ax4.plot(history['promotions'], color='#FBC15E', linewidth=2, drawstyle='steps-post')
ax4.fill_between(range(len(history['promotions'])), history['promotions'], alpha=0.3, color='#FBC15E', step='post')
ax4.set_title('Cumulative Precision Promotions', fontsize=14)
ax4.set_xlabel('Epoch', fontsize=12)
ax4.set_ylabel('Number of Promoted Layers', fontsize=12)

plt.tight_layout()
plt.savefig('vgg16_tf32_fp8_results.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Print Promotion History ---
print("\\n" + "="*80)
print("RIWAYAT PROMOSI TENSOR (FALLBACK KE TF32/FP32)")
print("="*80)
promotion_events = []
for epoch_idx, epoch_proms in enumerate(history.get('promotion_details', [])):
    epoch_num = epoch_idx + 1
    for prom in epoch_proms:
        print(f"Epoch {epoch_num:2d} | Group {prom['group_id']:3d} | Layer {prom['layer_name']:20s} | Type {prom['ttype']} | Index {prom['index']}")
        promotion_events.append(prom)
if not promotion_events:
    print("Tidak ada tensor yang dipromosikan (training berjalan stabil dalam FP8).")
print("="*80)
'''
    find_and_replace_cell(nb, 'Visualization', viz_source)

    sanitize_notebook_unicode(nb)

    with open('vgg16_cifar10_tf32_fp8.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

def rewrite_fp16_notebook():
    with open('vgg16_cifar10_fp16_fp8.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # 1. Imports
    imports_source = '''# ============================================================
# Cell 2: Imports & Environment Check
# ============================================================

import os, sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.cuda.amp import GradScaler, autocast

# ---- Add project root to sys.path ----
PROJECT_ROOT = os.path.abspath('.')
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
print(f'[INFO] Project root: {PROJECT_ROOT}')

# ---- Import ext3 modules ----
from ext3.nn import flatten
from ext3.nn.nn_native import (
    NativeConv2d, NativeLinear, NativeBatchNorm2d,
    NativeReLU, NativeMaxPool2d, NativeAdaptiveAvgPool2d,
    NativeDropout, reset_fp8_manager
)
from ext3.core.include.native_precision import (
    NativePrecisionMode, check_fp8_support, FP8Config
)
from ext3.core.emodlobj import EModlObjMgr
from ext3.core.include.dtype import Dtype, FP16
from ext3.core.include.pasn import Pasn
from ext3.core.include.ttype import Ttype
from ext3.util.apa_manager import (
    assign_precision, APAStabilityMonitor, StabilityEvent
)

# Environment Check
print(f"\\n=== Environment Check ===")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name       : {torch.cuda.get_device_name(0)}")
print(f"FP8 Supported  : {check_fp8_support()}")
'''
    find_and_replace_cell(nb, 'Cell 2: Imports', imports_source)

    # 2. Config
    config_source = '''# ============================================================
# Cell 3: Global Configuration
# ============================================================

# NOTE: TF32 NOT enabled here — we use FP16 as base
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

CONFIG = {
    'batch_size': 128,
    'epochs': 50,
    'lr': 0.01,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'num_workers': 0,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # --- Native APA (Adaptive Precision Assignment) Config ---
    'pa_upd_schm': 'topr_dec',
    'pa_upd_rmin': 0.3,
    'pa_upd_rmax': 0.4,
    
    # GradScaler config
    'grad_scaler_init_scale': 2.**16,
    'grad_scaler_growth_interval': 2000,
}

print("\\n=== Training Configuration ===")
for k, v in CONFIG.items():
    print(f"  {k:20s}: {v}")
'''
    find_and_replace_cell(nb, 'Cell 3: Global Configuration', config_source)

    # 2b. VGG16Native definition - replace torch.flatten with flatten from ext3.nn
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and 'Cell 4: VGG16 Model Definition' in ''.join(cell['source']):
            source = ''.join(cell['source'])
            source = source.replace('torch.flatten(x, 1)', 'flatten(x, 1)')
            cell['source'] = [line + '\n' for line in source.split('\n')][:-1]
            break

    # 3. Precision Assignment
    setup_source = '''# ============================================================
# Cell 5: Precision Assignment Setup
# ============================================================

# APA v2.0 — imported via ext3.nn imports cell above
print("APA v2.0: assign_precision + APAStabilityMonitor imported [OK]")
'''
    find_and_replace_cell(nb, 'Cell 5: Precision Assignment', setup_source)

    # 4. Train Epoch (with Promotion and GradScaler)
    train_source = '''# ============================================================
# Cell 8: Training Function (APA v2.0 — GradScaler + EMA-Gated Monitor)
# ============================================================

def train_epoch(model, loader, criterion, optimizer, scaler, device, monitor):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    total_grad_overflows = 0
    epoch_events = {"stable": 0, "nan": 0, "spike": 0, "overflow": 0}
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with autocast(dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scale_before = scaler.get_scale()
            scaler.update()
            scale_after = scaler.get_scale()
            if scale_after < scale_before:
                total_grad_overflows += 1
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        # --- APA v2.0: EMA-Gated Stability Monitor ---
        # Reuse loss.item() for both logging AND monitoring (zero extra sync)
        loss_val = loss.item()
        event = monitor.step(loss_val, model, FP16)
        
        if event == StabilityEvent.LOSS_NAN:
            epoch_events["nan"] += 1
        elif event == StabilityEvent.LOSS_SPIKE:
            epoch_events["spike"] += 1
        elif event == StabilityEvent.GRADIENT_OVERFLOW:
            epoch_events["overflow"] += 1
        
        total_loss += loss_val * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    # Map promotions
    flags = EModlObjMgr.get_inc_ts_prec_flag()
    promoted_details = []
    if len(flags) > 0 and sum(flags) > 0:
        cnt = -1
        mapping = {}
        from ext3.core.include.ttype import Ttype
        for ttype in (Ttype.P, Ttype.Y):
            for emodl in EModlObjMgr.get_emodls_sort():
                if ttype in emodl.info_ts and hasattr(emodl.info_ts[ttype], 'undovr'):
                    for tsind, _ in enumerate(emodl.info_ts[ttype].undovr):
                        cnt += 1
                        mapping[cnt] = (ttype, emodl, tsind)
        for idx, f in enumerate(flags):
            if f > 0.0:
                tt = mapping.get(idx)
                if tt:
                    ttype, emodl, tsind = tt
                    gid = emodl.info_mdcur.id['grp_all'].val
                    promoted_details.append({
                        'group_id': gid,
                        'layer_name': type(emodl).__name__,
                        'ttype': 'Parameter' if ttype == Ttype.P else 'Activation',
                        'index': tsind
                    })
    
    total_promotions = len(promoted_details)
    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy, total_promotions, total_grad_overflows, promoted_details, epoch_events

print("train_epoch() defined [OK] [APA v2.0: GradScaler + EMA-Gated]")
'''
    find_and_replace_cell(nb, 'Cell 8: Training Function', train_source)

    # 4b. Evaluate Function (CPU-compatible)
    eval_source = '''# ============================================================
# Cell 9: Evaluation Function
# ============================================================

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        if device == 'cuda':
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

print("evaluate() defined [OK]")
'''
    find_and_replace_cell(nb, 'Cell 9: Evaluation Function', eval_source)

    # 5. Main Loop
    main_source = '''# ============================================================
# Cell 11: Main Training Loop (APA v2.0)
# ============================================================

model = VGG16Native(num_classes=10).to(CONFIG['device'])

reset_fp8_manager()
pasn_manager = assign_precision(model, CONFIG)

# --- Grouping & Demotion Analysis ---
group_stats = {}
total_params_all = 0
total_params_fp8 = 0

for emodl in EModlObjMgr.get_emodls_sort():
    gid = emodl.info_mdcur.id['grp_all'].val
    if gid not in group_stats:
        group_stats[gid] = {
            'layers': [],
            'params': 0,
            'mode_y': 'unknown',
            'mode_p': 'unknown'
        }
    
    n_params = sum(p.numel() for p in emodl.parameters())
    total_params_all += n_params
    group_stats[gid]['params'] += n_params
    group_stats[gid]['layers'].append(type(emodl).__name__)
    
    dtype_y = (emodl.info_ts[Ttype.Y].dtype[0]
               if hasattr(emodl.info_ts[Ttype.Y], 'dtype')
               and len(emodl.info_ts[Ttype.Y].dtype) > 0
               else FP16)
    dtype_p = (emodl.info_ts[Ttype.P].dtype[0]
               if hasattr(emodl.info_ts[Ttype.P], 'dtype')
               and len(emodl.info_ts[Ttype.P].dtype) > 0
               else FP16)
    mode_y = dtype_y.to_native().mode_name
    mode_p = dtype_p.to_native().mode_name
    group_stats[gid]['mode_y'] = mode_y
    group_stats[gid]['mode_p'] = mode_p
    
    if mode_p == 'low_fp8':
        total_params_fp8 += n_params

print("\\n" + "="*80)
print("ANALISIS TENSOR GROUPING & DEMOTION")
print("="*80)
print(f"Total Parameter Model: {total_params_all:,}")
print(f"Total Parameter FP8  : {total_params_fp8:,} ({total_params_fp8/total_params_all*100:.2f}%)")
print("-"*80)
print(f"{'Group ID':<10} | {'Jumlah Layer':<12} | {'Total Params':<15} | {'Activation (Y)':<15} | {'Parameter (P)':<15}")
print("-"*80)
for gid, stats in sorted(group_stats.items()):
    print(f"{gid:<10d} | {len(stats['layers']):<12d} | {stats['params']:<15,d} | {stats['mode_y']:<15s} | {stats['mode_p']:<15s}")
print("="*80)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=CONFIG['lr'], 
                      momentum=CONFIG['momentum'], weight_decay=CONFIG['weight_decay'])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])

scaler = GradScaler(
    init_scale=CONFIG['grad_scaler_init_scale'],
    growth_interval=CONFIG['grad_scaler_growth_interval']
) if CONFIG['device'] == 'cuda' else None

# --- APA v2.0: Initialize EMA-Gated Stability Monitor ---
monitor = APAStabilityMonitor(
    ema_alpha=0.1,            # EMA smoothing (alpha)
    variance_threshold=2.0,   # Spike sensitivity in sigma (sigma)
    ovr_thrs=0.0,             # Any overflow triggers promotion
    warmup_steps=10,          # EMA calibration period
)
print(f"APA v2.0 Monitor initialized: alpha={0.1}, sigma_thresh={2.0}, warmup={10}")

history = {
    'train_loss': [],
    'train_acc': [],
    'test_acc': [],
    'vram': [],
    'lr': [],
    'promotions': [],
    'grad_overflows': [],
    'spike_events': [],
    'promotion_details': [],
}

print("\\n" + "=" * 120)
print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Test Acc':>8} | {'Promoted':>8} | {'Grad Ovr':>8} | {'Spikes':>6} | {'GPU Syncs':>9} | {'VRAM MB':>8} | {'Time':>6} | {'LR':>8}")
print("-" * 120)

best_acc = 0.0
cumulative_promotions = 0

for epoch in range(1, CONFIG['epochs'] + 1):
    start_time = time.time()
    current_lr = optimizer.param_groups[0]['lr']
    
    train_loss, train_acc, promotions, grad_overflows, promoted_details, events = train_epoch(
        model, trainloader, criterion, optimizer, scaler, CONFIG['device'], monitor
    )
    _, test_acc = evaluate(model, testloader, criterion, CONFIG['device'])
    scheduler.step()
    
    epoch_time = time.time() - start_time
    vram = get_vram_mb()
    cumulative_promotions += promotions
    stats = monitor.get_stats()
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['test_acc'].append(test_acc)
    history['vram'].append(vram)
    history['lr'].append(current_lr)
    history['promotions'].append(cumulative_promotions)
    history['grad_overflows'].append(grad_overflows)
    history['spike_events'].append(stats['spike_events'])
    history['promotion_details'].append(promoted_details)
    
    best_marker = " *" if test_acc > best_acc else ""
    best_acc = max(best_acc, test_acc)
    
    print(f"{epoch:5d} | {train_loss:10.4f} | {train_acc:8.2f}% | {test_acc:7.2f}% | {promotions:5d} -> | {grad_overflows:8d} | {stats['spike_events']:6d} | {stats['tier2_checks']:9d} | {vram:7.1f}M | {epoch_time:5.1f}s | {current_lr:.6f}{best_marker}")

print("=" * 120)
print(f"Training Complete! Best Test Accuracy: {best_acc:.2f}%")
print(f"Total layers promoted to FP16: {cumulative_promotions}")
print(f"APA v2.0 Efficiency: {stats['sync_efficiency']}")
'''
    find_and_replace_cell(nb, 'Main Training Loop', main_source)

    # 6. Visualization
    viz_source = '''# ============================================================
# Cell 12: Visualization (Termasuk Promotion Plot & List)
# ============================================================

plt.style.use('ggplot')
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 5))

# 1. Loss Curve
ax1.plot(history['train_loss'], color='#E24A33', linewidth=2, marker='o', markersize=4)
ax1.set_title('Training Loss', fontsize=14)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Cross Entropy', fontsize=12)

# 2. Accuracy Curve
ax2.plot(history['train_acc'], label='Train Acc', color='#348ABD', linewidth=2)
ax2.plot(history['test_acc'], label='Test Acc', color='#988ED5', linewidth=2, marker='s', markersize=4)
ax2.set_title('Accuracy', fontsize=14)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.legend()

# 3. VRAM Usage
ax3.plot(history['vram'], color='#8EBA42', linewidth=2, fillstyle='bottom')
ax3.fill_between(range(len(history['vram'])), history['vram'], alpha=0.3, color='#8EBA42')
ax3.set_title('VRAM Allocation', fontsize=14)
ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('VRAM (MB)', fontsize=12)

# 4. Cumulative Promotions
ax4.plot(history['promotions'], color='#FBC15E', linewidth=2, drawstyle='steps-post')
ax4.fill_between(range(len(history['promotions'])), history['promotions'], alpha=0.3, color='#FBC15E', step='post')
ax4.set_title('Cumulative Precision Promotions', fontsize=14)
ax4.set_xlabel('Epoch', fontsize=12)
ax4.set_ylabel('Number of Promoted Layers', fontsize=12)

plt.tight_layout()
plt.savefig('vgg16_fp16_fp8_training_results.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Print Promotion History ---
print("\\n" + "="*80)
print("RIWAYAT PROMOSI TENSOR (FALLBACK KE FP16)")
print("="*80)
promotion_events = []
for epoch_idx, epoch_proms in enumerate(history.get('promotion_details', [])):
    epoch_num = epoch_idx + 1
    for prom in epoch_proms:
        print(f"Epoch {epoch_num:2d} | Group {prom['group_id']:3d} | Layer {prom['layer_name']:20s} | Type {prom['ttype']} | Index {prom['index']}")
        promotion_events.append(prom)
if not promotion_events:
    print("Tidak ada tensor yang dipromosikan (training berjalan stabil dalam FP8).")
print("="*80)
'''
    find_and_replace_cell(nb, 'Visualization', viz_source)

    sanitize_notebook_unicode(nb)

    with open('vgg16_cifar10_fp16_fp8.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

if __name__ == '__main__':
    rewrite_tf32_notebook()
    rewrite_fp16_notebook()
    print("Successfully rewrote both notebooks.")
