import json
import os
import sys

def rewrite_tf32_notebook():
    with open('vgg16_cifar10_tf32_fp8.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # 1. Imports
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and 'torchvision' in ''.join(cell['source']):
            source = '''# ============================================================
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

# Environment Check
print(f"\\n=== Environment Check ===")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name       : {torch.cuda.get_device_name(0)}")
print(f"FP8 Supported  : {check_fp8_support()}")
'''
            cell['source'] = [line + '\n' for line in source.split('\n')][:-1]
            break

    # 2. Config
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and 'CONFIG' in ''.join(cell['source']):
            source = '''# ============================================================
# Cell 3: Global Configuration
# ============================================================

enable_tf32()

CONFIG = {
    'batch_size': 128,
    'epochs': 50,
    'lr': 0.01,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'num_workers': 2,
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
            cell['source'] = [line + '\n' for line in source.split('\n')][:-1]
            break

    # 3. Precision Assignment
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and 'assign_precision' in ''.join(cell['source']):
            source = '''# ============================================================
# Cell 5: Precision Assignment via Original EModlObjMgr & Pasn
# ============================================================

def assign_precision(model: nn.Module, config: dict) -> Pasn:
    print("\\n" + "=" * 80)
    print("PRECISION ASSIGNMENT (Original APA Architecture)")
    print("=" * 80)
    
    # 1. Register modules
    EModlObjMgr.unregister_all()
    EModlObjMgr.register(model)
    
    # 2. Topological Mapping (Dummy forward)
    def new_id_grp_all(emodl):
        return isinstance(emodl, (NativeConv2d, NativeLinear))
        
    EModlObjMgr.set_info_mdcur_id(new_id_grp_all)
    EModlObjMgr.reset_info(True)
    with torch.no_grad():
        dummy_input = torch.randn(2, 3, 32, 32).to(config['device'])
        model(dummy_input)
    EModlObjMgr.reset_info(False)
    
    # 3. Create Pasn & Dtypes
    # Gunakan Dtype FP8 representasi (8-bit total = low_fp8)
    FP8 = Dtype(4, 3, 0) 
    
    pasn = Pasn(EModlObjMgr.get_emodls(), dtype_fwd=FP32)
    
    # 4. Demotion logic (mirip asli)
    target_ratio = config['pa_upd_rmin']
    
    # Pilih ID untuk demotion berdasarkan ukuran layer
    ids, r = EModlObjMgr.get_ids_chosen('grp_all', config['pa_upd_schm'], r_min=target_ratio)
    
    # Proteksi layer pertama dan terakhir
    # Asumsikan ID grp_all secara berurutan sesuai eksekusi
    if len(ids) > 0:
        sorted_emodls = EModlObjMgr.get_emodls_sort()
        first_id = sorted_emodls[0].info_mdcur.id['grp_all']
        last_id = sorted_emodls[-1].info_mdcur.id['grp_all']
        ids = [i for i in ids if i != first_id and i != last_id]
    
    # Update DtypePlan
    upd_dtplan = {Ttype.Y: FP8}
    pasn.update_by_id_grp_all('cur', 'id', upd_dtplan, [i.val for i in ids])
    
    # Apply to Graph
    EModlObjMgr.set_info_ts_dtype(pasn)
    
    print(f"  Target Demotion Ratio : {target_ratio:.3f}")
    print(f"  Total Layers          : {len(EModlObjMgr.get_emodls_sort())}")
    print(f"  Demoted Layers        : {len(ids)}")
    
    print("\\n=== Grouping & Demotion Map ===")
    for emodl in EModlObjMgr.get_emodls_sort():
        uid = emodl.info_mdcur.id['grp_all'].val
        dtype = emodl.info_ts[Ttype.Y].dtype[0]
        mode = dtype.to_native().mode_name
        marker = " [DEMOTED]" if mode == "low_fp8" else ""
        print(f"Layer ID {uid:3d} | {str(type(emodl)):40s} | {mode:10s}{marker}")
    
    print("=" * 80)
    return pasn

print("assign_precision() defined ✓")
'''
            cell['source'] = [line + '\n' for line in source.split('\n')][:-1]
            break

    # 4. Train Epoch (with Promotion)
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and 'train_epoch' in ''.join(cell['source']):
            source = '''# ============================================================
# Cell 8: Training Function
# ============================================================

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # --- NATIVE APA: Precision Promotion ---
        # 1. Ambil tensor undovr dari seluruh layer
        undovrs = EModlObjMgr.get_undovrs()
        # 2. Check apakah ada overflow (indeks 1 dari tensor [underflow, overflow])
        # Kita menggunakan Ttype.Y sesuai injeksi di NativeConv2d
        overflow_counts = undovrs[Ttype.Y][:, 1] 
        flag = overflow_counts > 0
        # 3. Jika ada, lakukan promosi kembali ke FP32
        EModlObjMgr.inc_ts_prec(flag, {Ttype.Y: FP32, Ttype.P: FP32})
        
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    # Dapatkan jumlah promosi epoch ini
    # inc_ts_prec_flag merekam flag promosi yang di-trigger selama training loop
    flags = EModlObjMgr.get_inc_ts_prec_flag()
    total_promotions = sum([1 for f in flags if f > 0.0])
    
    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy, total_promotions

print("train_epoch() defined ✓")
'''
            cell['source'] = [line + '\n' for line in source.split('\n')][:-1]
            break

    # 5. Main Loop
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and 'epochs' in ''.join(cell['source']) and 'optimizer' in ''.join(cell['source']):
            source = '''# ============================================================
# Cell 10: Main Training Loop
# ============================================================

reset_fp8_manager()
pasn_manager = assign_precision(model, CONFIG)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=CONFIG['lr'], 
                      momentum=CONFIG['momentum'], weight_decay=CONFIG['weight_decay'])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])

history = {
    'train_loss': [],
    'train_acc': [],
    'test_acc': [],
    'vram': [],
    'lr': [],
    'promotions': [],
}

print("\\n" + "=" * 90)
print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Test Acc':>8} | {'Promoted':>8} | {'VRAM MB':>8} | {'Time':>6} | {'LR':>8}")
print("-" * 90)

best_acc = 0.0
cumulative_promotions = 0

for epoch in range(1, CONFIG['epochs'] + 1):
    start_time = time.time()
    current_lr = optimizer.param_groups[0]['lr']
    
    train_loss, train_acc, promotions = train_epoch(
        model, trainloader, criterion, optimizer, CONFIG['device']
    )
    test_acc = evaluate(model, testloader, criterion, CONFIG['device'])
    scheduler.step()
    
    epoch_time = time.time() - start_time
    vram = get_vram_mb()
    cumulative_promotions += promotions
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['test_acc'].append(test_acc)
    history['vram'].append(vram)
    history['lr'].append(current_lr)
    history['promotions'].append(cumulative_promotions)
    
    best_marker = " *" if test_acc > best_acc else ""
    best_acc = max(best_acc, test_acc)
    
    print(f"{epoch:5d} | {train_loss:10.4f} | {train_acc:8.2f}% | {test_acc:7.2f}% | {promotions:5d} ↑ | {vram:7.1f}M | {epoch_time:5.1f}s | {current_lr:.6f}{best_marker}")

print("=" * 90)
print(f"Training Complete! Best Test Accuracy: {best_acc:.2f}%")
print(f"Total layers promoted to TF32/FP32: {cumulative_promotions}")
'''
            cell['source'] = [line + '\n' for line in source.split('\n')][:-1]
            break

    # 6. Visualization
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and 'plt.subplots' in ''.join(cell['source']):
            source = '''# ============================================================
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
            cell['source'] = [line + '\n' for line in source.split('\n')][:-1]
            break

    with open('vgg16_cifar10_tf32_fp8.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

def rewrite_fp16_notebook():
    with open('vgg16_cifar10_fp16_fp8.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Note: Using similar logic as TF32 but with GradScaler and FP16 base dtype
    
    # 1. Imports
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and 'torchvision' in ''.join(cell['source']):
            source = '''# ============================================================
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

# Environment Check
print(f"\\n=== Environment Check ===")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name       : {torch.cuda.get_device_name(0)}")
print(f"FP8 Supported  : {check_fp8_support()}")
'''
            cell['source'] = [line + '\n' for line in source.split('\n')][:-1]
            break

    # 2. Config
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and 'CONFIG' in ''.join(cell['source']):
            source = '''# ============================================================
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
    'num_workers': 2,
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
            cell['source'] = [line + '\n' for line in source.split('\n')][:-1]
            break

    # 3. Precision Assignment
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and 'assign_precision' in ''.join(cell['source']):
            source = '''# ============================================================
# Cell 5: Precision Assignment via Original EModlObjMgr & Pasn
# ============================================================

def assign_precision(model: nn.Module, config: dict) -> Pasn:
    print("\\n" + "=" * 80)
    print("PRECISION ASSIGNMENT (Original APA Architecture - FP16 Base)")
    print("=" * 80)
    
    EModlObjMgr.unregister_all()
    EModlObjMgr.register(model)
    
    def new_id_grp_all(emodl):
        return isinstance(emodl, (NativeConv2d, NativeLinear))
        
    EModlObjMgr.set_info_mdcur_id(new_id_grp_all)
    EModlObjMgr.reset_info(True)
    with torch.no_grad():
        with autocast(dtype=torch.float16):
            dummy_input = torch.randn(2, 3, 32, 32).to(config['device'])
            model(dummy_input)
    EModlObjMgr.reset_info(False)
    
    FP8 = Dtype(4, 3, 0)
    pasn = Pasn(EModlObjMgr.get_emodls(), dtype_fwd=FP16)
    
    target_ratio = config['pa_upd_rmin']
    ids, r = EModlObjMgr.get_ids_chosen('grp_all', config['pa_upd_schm'], r_min=target_ratio)
    
    if len(ids) > 0:
        sorted_emodls = EModlObjMgr.get_emodls_sort()
        first_id = sorted_emodls[0].info_mdcur.id['grp_all']
        last_id = sorted_emodls[-1].info_mdcur.id['grp_all']
        ids = [i for i in ids if i != first_id and i != last_id]
    
    upd_dtplan = {Ttype.Y: FP8}
    pasn.update_by_id_grp_all('cur', 'id', upd_dtplan, [i.val for i in ids])
    
    EModlObjMgr.set_info_ts_dtype(pasn)
    
    print(f"  Target Demotion Ratio : {target_ratio:.3f}")
    print(f"  Total Layers          : {len(EModlObjMgr.get_emodls_sort())}")
    print(f"  Demoted Layers        : {len(ids)}")
    
    print("\\n=== Grouping & Demotion Map ===")
    for emodl in EModlObjMgr.get_emodls_sort():
        uid = emodl.info_mdcur.id['grp_all'].val
        dtype = emodl.info_ts[Ttype.Y].dtype[0]
        mode = dtype.to_native().mode_name
        marker = " [DEMOTED]" if mode == "low_fp8" else ""
        print(f"Layer ID {uid:3d} | {str(type(emodl)):40s} | {mode:10s}{marker}")
    
    print("=" * 80)
    return pasn

print("assign_precision() defined ✓")
'''
            cell['source'] = [line + '\n' for line in source.split('\n')][:-1]
            break

    # 4. Train Epoch (with Promotion and GradScaler)
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and 'train_epoch' in ''.join(cell['source']):
            source = '''# ============================================================
# Cell 8: Training Function (with GradScaler)
# ============================================================

def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    total_grad_overflows = 0
    
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
            
        # --- NATIVE APA: Precision Promotion ---
        undovrs = EModlObjMgr.get_undovrs()
        overflow_counts = undovrs[Ttype.Y][:, 1] 
        flag = overflow_counts > 0
        # Promosi kembali ke FP16
        EModlObjMgr.inc_ts_prec(flag, {Ttype.Y: FP16, Ttype.P: FP16})
        
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    flags = EModlObjMgr.get_inc_ts_prec_flag()
    total_promotions = sum([1 for f in flags if f > 0.0])
    
    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy, total_promotions, total_grad_overflows

print("train_epoch() defined ✓")
'''
            cell['source'] = [line + '\n' for line in source.split('\n')][:-1]
            break
            
    # 5. Main Loop
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and 'epochs' in ''.join(cell['source']) and 'optimizer' in ''.join(cell['source']):
            source = '''# ============================================================
# Cell 11: Main Training Loop
# ============================================================

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

history = {
    'train_loss': [],
    'train_acc': [],
    'test_acc': [],
    'vram': [],
    'lr': [],
    'promotions': [],
    'grad_overflows': [],
}

print("\\n" + "=" * 105)
print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Test Acc':>8} | {'Promoted':>8} | {'Grad Ovr':>8} | {'VRAM MB':>8} | {'Time':>6} | {'LR':>8}")
print("-" * 105)

best_acc = 0.0
cumulative_promotions = 0

for epoch in range(1, CONFIG['epochs'] + 1):
    start_time = time.time()
    current_lr = optimizer.param_groups[0]['lr']
    
    train_loss, train_acc, promotions, grad_overflows = train_epoch(
        model, trainloader, criterion, optimizer, scaler, CONFIG['device']
    )
    test_acc = evaluate(model, testloader, criterion, CONFIG['device'])
    scheduler.step()
    
    epoch_time = time.time() - start_time
    vram = get_vram_mb()
    cumulative_promotions += promotions
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['test_acc'].append(test_acc)
    history['vram'].append(vram)
    history['lr'].append(current_lr)
    history['promotions'].append(cumulative_promotions)
    history['grad_overflows'].append(grad_overflows)
    
    best_marker = " *" if test_acc > best_acc else ""
    best_acc = max(best_acc, test_acc)
    
    print(f"{epoch:5d} | {train_loss:10.4f} | {train_acc:8.2f}% | {test_acc:7.2f}% | {promotions:5d} ↑ | {grad_overflows:8d} | {vram:7.1f}M | {epoch_time:5.1f}s | {current_lr:.6f}{best_marker}")

print("=" * 105)
print(f"Training Complete! Best Test Accuracy: {best_acc:.2f}%")
print(f"Total layers promoted to FP16: {cumulative_promotions}")
'''
            cell['source'] = [line + '\n' for line in source.split('\n')][:-1]
            break

    # 6. Visualization
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and 'plt.subplots' in ''.join(cell['source']):
            source = '''# ============================================================
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
            cell['source'] = [line + '\n' for line in source.split('\n')][:-1]
            break

    with open('vgg16_cifar10_fp16_fp8.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

rewrite_tf32_notebook()
rewrite_fp16_notebook()
print("Successfully rewrote both notebooks.")
