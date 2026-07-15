import json
import os
import sys

def find_and_replace_cell(nb, keyword, new_source):
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and keyword in ''.join(cell['source']):
            cell['source'] = [line + '\n' for line in new_source.split('\n')][:-1]
            return True
    return False

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
    find_and_replace_cell(nb, 'Cell 2: Imports & Environment Check', imports_source)

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
print("APA v2.0: assign_precision + APAStabilityMonitor imported ✓")
'''
    find_and_replace_cell(nb, 'Cell 5: Precision Assignment Setup', setup_source)

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
    
    # Dapatkan jumlah promosi epoch ini
    flags = EModlObjMgr.get_inc_ts_prec_flag()
    total_promotions = sum([1 for f in flags if f > 0.0])
    
    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy, total_promotions, epoch_events

print("train_epoch() defined ✓ [APA v2.0: EMA-Gated]")
'''
    find_and_replace_cell(nb, 'Cell 8: Training Function', train_source)

    # 5. Main Loop
    main_source = '''# ============================================================
# Cell 10: Main Training Loop (APA v2.0)
# ============================================================

model = VGG16Native(num_classes=10).to(CONFIG['device'])

reset_fp8_manager()
pasn_manager = assign_precision(model, CONFIG)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=CONFIG['lr'], 
                      momentum=CONFIG['momentum'], weight_decay=CONFIG['weight_decay'])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])

# --- APA v2.0: Initialize EMA-Gated Stability Monitor ---
monitor = APAStabilityMonitor(
    ema_alpha=0.1,            # EMA smoothing (tunable: 0.05-0.3)
    variance_threshold=2.0,   # Spike sensitivity in σ (tunable: 1.5-4.0)
    ovr_thrs=0.0,             # Any overflow triggers promotion
    warmup_steps=10,          # EMA calibration period
)
print(f"APA v2.0 Monitor initialized: α={0.1}, σ_thresh={2.0}, warmup={10}")

history = {
    'train_loss': [],
    'train_acc': [],
    'test_acc': [],
    'vram': [],
    'lr': [],
    'promotions': [],
    'spike_events': [],
}

print("\\n" + "=" * 105)
print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Test Acc':>8} | {'Promoted':>8} | {'Spikes':>6} | {'GPU Syncs':>9} | {'VRAM MB':>8} | {'Time':>6} | {'LR':>8}")
print("-" * 105)

best_acc = 0.0
cumulative_promotions = 0

for epoch in range(1, CONFIG['epochs'] + 1):
    start_time = time.time()
    current_lr = optimizer.param_groups[0]['lr']
    
    train_loss, train_acc, promotions, events = train_epoch(
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
    
    best_marker = " *" if test_acc > best_acc else ""
    best_acc = max(best_acc, test_acc)
    
    print(f"{epoch:5d} | {train_loss:10.4f} | {train_acc:8.2f}% | {test_acc:7.2f}% | {promotions:5d} ↑ | {stats['spike_events']:6d} | {stats['tier2_checks']:9d} | {vram:7.1f}M | {epoch_time:5.1f}s | {current_lr:.6f}{best_marker}")

print("=" * 105)
print(f"Training Complete! Best Test Accuracy: {best_acc:.2f}%")
print(f"Total layers promoted to TF32/FP32: {cumulative_promotions}")
print(f"APA v2.0 Efficiency: {stats['sync_efficiency']}")
'''
    find_and_replace_cell(nb, 'Cell 10: Main Training Loop', main_source)

    # 6. Visualization
    viz_source = '''# ============================================================
# Cell 11: Visualization (Termasuk Promotion Plot)
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
'''
    find_and_replace_cell(nb, 'Cell 11: Visualization (Termasuk Promotion Plot)', viz_source)

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
    find_and_replace_cell(nb, 'Cell 2: Imports & Environment Check', imports_source)

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
print("APA v2.0: assign_precision + APAStabilityMonitor imported ✓")
'''
    find_and_replace_cell(nb, 'Cell 5: Precision Assignment Setup', setup_source)

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
        
    flags = EModlObjMgr.get_inc_ts_prec_flag()
    total_promotions = sum([1 for f in flags if f > 0.0])
    
    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy, total_promotions, total_grad_overflows, epoch_events

print("train_epoch() defined ✓ [APA v2.0: GradScaler + EMA-Gated]")
'''
    find_and_replace_cell(nb, 'Cell 8: Training Function (with GradScaler)', train_source)

    # 5. Main Loop
    main_source = '''# ============================================================
# Cell 11: Main Training Loop (APA v2.0)
# ============================================================

model = VGG16Native(num_classes=10).to(CONFIG['device'])

reset_fp8_manager()
pasn_manager = assign_precision(model, CONFIG)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=CONFIG['lr'], 
                      momentum=CONFIG['momentum'], weight_decay=CONFIG['weight_decay'])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])

scaler = GradScaler(
    init_scale=CONFIG['grad_scaler_init_scale'],
    growth_interval=CONFIG['grad_scaler_growth_interval']
)

# --- APA v2.0: Initialize EMA-Gated Stability Monitor ---
monitor = APAStabilityMonitor(
    ema_alpha=0.1,            # EMA smoothing (tunable: 0.05-0.3)
    variance_threshold=2.0,   # Spike sensitivity in σ (tunable: 1.5-4.0)
    ovr_thrs=0.0,             # Any overflow triggers promotion
    warmup_steps=10,          # EMA calibration period
)
print(f"APA v2.0 Monitor initialized: α={0.1}, σ_thresh={2.0}, warmup={10}")

history = {
    'train_loss': [],
    'train_acc': [],
    'test_acc': [],
    'vram': [],
    'lr': [],
    'promotions': [],
    'grad_overflows': [],
    'spike_events': [],
}

print("\\n" + "=" * 120)
print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Test Acc':>8} | {'Promoted':>8} | {'Grad Ovr':>8} | {'Spikes':>6} | {'GPU Syncs':>9} | {'VRAM MB':>8} | {'Time':>6} | {'LR':>8}")
print("-" * 120)

best_acc = 0.0
cumulative_promotions = 0

for epoch in range(1, CONFIG['epochs'] + 1):
    start_time = time.time()
    current_lr = optimizer.param_groups[0]['lr']
    
    train_loss, train_acc, promotions, grad_overflows, events = train_epoch(
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
    
    best_marker = " *" if test_acc > best_acc else ""
    best_acc = max(best_acc, test_acc)
    
    print(f"{epoch:5d} | {train_loss:10.4f} | {train_acc:8.2f}% | {test_acc:7.2f}% | {promotions:5d} ↑ | {grad_overflows:8d} | {stats['spike_events']:6d} | {stats['tier2_checks']:9d} | {vram:7.1f}M | {epoch_time:5.1f}s | {current_lr:.6f}{best_marker}")

print("=" * 120)
print(f"Training Complete! Best Test Accuracy: {best_acc:.2f}%")
print(f"Total layers promoted to FP16: {cumulative_promotions}")
print(f"APA v2.0 Efficiency: {stats['sync_efficiency']}")
'''
    find_and_replace_cell(nb, 'Cell 11: Main Training Loop', main_source)

    # 6. Visualization
    viz_source = '''# ============================================================
# Cell 12: Visualization
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
'''
    find_and_replace_cell(nb, 'Cell 12: Visualization', viz_source)

    with open('vgg16_cifar10_fp16_fp8.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

if __name__ == '__main__':
    rewrite_tf32_notebook()
    rewrite_fp16_notebook()
    print("Successfully rewrote both notebooks.")
