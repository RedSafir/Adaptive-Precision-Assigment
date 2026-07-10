import json

with open('vgg16_cifar10_tf32_fp8.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update Cell 2 (Imports)
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'Cell 2: Imports' in ''.join(cell['source']):
        source = ''.join(cell['source'])
        if 'NativePasn' not in source:
            source = source.replace('    FP8Config,\n)', '    FP8Config,\n)\nfrom ext3.core.include.pasn import NativePasn\n')
            cell['source'] = [line + '\n' if not line.endswith('\n') else line for line in source.split('\n')][:-1]

# Update Cell 3 (Configuration)
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'Cell 3: Global Configuration' in ''.join(cell['source']):
        new_source = '''# ============================================================
# Cell 3: Global Configuration
# ============================================================

# Aktifkan TF32 untuk semua matmul dan convolution operations
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
    # Diambil dari script_ext3/cifar_perf_main_2_heuris.sh
    'pa_upd_schm': 'topr_dec',    # Demote layer terbesar lebih dulu
    'pa_upd_rmin': 0.3,           # Target 30% dari total parameter turun ke FP8
    'pa_upd_rmax': 0.4,           
    'spike_threshold': 0.5,       # Ambang batas lonjakan AMAX (50% dari rata-rata). 
                                  # Ekuivalen dengan --pa_ovr_thrs 0.01 pada simulasi bit asli.
}

print("\\n=== Training Configuration ===")
for k, v in CONFIG.items():
    print(f"  {k:20s}: {v}")

print(f"\\n  TF32 matmul        : {torch.backends.cuda.matmul.allow_tf32}")
print(f"  TF32 cudnn         : {torch.backends.cudnn.allow_tf32}")
'''
        cell['source'] = [line + '\n' for line in new_source.split('\n')][:-1]

# Update Cell 5 (Precision Assignment)
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'Cell 5: Precision' in ''.join(cell['source']):
        new_source = '''# ============================================================
# Cell 5: Precision Assignment via NativePasn
# ============================================================

def assign_precision(model: nn.Module, config: dict) -> NativePasn:
    """
    Menggunakan NativePasn untuk melakukan Tensor Grouping dan Precision Demotion.
    Melindungi layer pertama dan terakhir secara otomatis.
    """
    print("\\n" + "=" * 60)
    print("PRECISION ASSIGNMENT (Native APA)")
    print("=" * 60)
    
    # 1. Initialize Pasn (Fase Grouping)
    pasn = NativePasn(model)
    
    # 2. Set global spike threshold untuk amax tracking
    from ext3.nn.nn_native import get_fp8_manager
    manager = get_fp8_manager()
    manager.set_spike_threshold(config['spike_threshold'])
    
    # 3. Apply Demotion
    stats = pasn.apply_demotion(
        scheme=config['pa_upd_schm'],
        r_min=config['pa_upd_rmin'],
        r_max=config['pa_upd_rmax'],
        base_mode=NativePrecisionMode.BASE_TF32,
        low_mode=NativePrecisionMode.LOW_FP8,
        protect_ends=True  # Sesuai dengan pa_upd_no_end
    )
    
    print(f"  Scheme          : {stats.get('scheme')}")
    print(f"  Target Ratio    : {stats.get('target_ratio')}")
    print(f"  Actual Ratio    : {stats.get('actual_ratio'):.3f}")
    print(f"  Demoted Layers  : {stats.get('demoted_layers')} / {stats.get('total_layers')}")
    print("=" * 60)
    
    return pasn

print("assign_precision() defined ✓")
'''
        cell['source'] = [line + '\n' for line in new_source.split('\n')][:-1]

# Update Cell 8 (Train Epoch)
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'Cell 8: Training Function' in ''.join(cell['source']):
        new_source = '''# ============================================================
# Cell 8: Training Function
# ============================================================

def train_epoch(model, loader, criterion, optimizer, device, pasn_manager: NativePasn):
    """
    Train model untuk satu epoch.
    Termasuk pemanggilan Fase 3 (Precision Promotion) setelah setiap backward pass.
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    total_promotions = 0
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # --- NATIVE APA: Precision Promotion ---
        # Mengecek apakah ada layer FP8 yang mengalami lonjakan amax ekstrim
        promoted = pasn_manager.check_and_promote(NativePrecisionMode.BASE_TF32)
        total_promotions += promoted
        
        # Accumulate statistics
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy, total_promotions

print("train_epoch() defined ✓")
'''
        cell['source'] = [line + '\n' for line in new_source.split('\n')][:-1]

# Update Cell 10 (Main Loop)
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'Cell 10: Main' in ''.join(cell['source']):
        source = ''.join(cell['source'])
        source = source.replace('assign_precision(model, CONFIG)', 'pasn_manager = assign_precision(model, CONFIG)')
        source = source.replace('train_loss, train_acc = train_epoch(\n        model, trainloader, criterion, optimizer, CONFIG[\'device\']\n    )', 
                                'train_loss, train_acc, promotions = train_epoch(\n        model, trainloader, criterion, optimizer, CONFIG[\'device\'], pasn_manager\n    )')
        source = source.replace("'lr': [],\n}", "'lr': [],\n    'promotions': [],\n}")
        source = source.replace("history['lr'].append(current_lr)", "history['lr'].append(current_lr)\n    history['promotions'].append(promotions)")
        
        # update print statement
        source = source.replace(
            'print(f"{epoch:5d} | {train_loss:10.4f} | {train_acc:8.2f}% | {test_acc:7.2f}% | "\\\n          f"{vram:7.1f}M | {epoch_time:5.1f}s | {current_lr:.6f}{best_marker}")',
            'print(f"{epoch:5d} | {train_loss:10.4f} | {train_acc:8.2f}% | {test_acc:7.2f}% | {promotions:5d} ↑ | "\\\n          f"{vram:7.1f}M | {epoch_time:5.1f}s | {current_lr:.6f}{best_marker}")'
        )
        source = source.replace(
            'f"{\'VRAM MB\':>8} | {\'Time\':>6} | {\'LR\':>8}")',
            'f"{\'Promoted\':>8} | {\'VRAM MB\':>8} | {\'Time\':>6} | {\'LR\':>8}")'
        )
        source = source.replace(
            'print(f"{\'Epoch\':>5} | {\'Train Loss\':>10} | {\'Train Acc\':>9} | {\'Test Acc\':>8} | "\\\n      f"{\'VRAM MB\':>8}',
            'print(f"{\'Epoch\':>5} | {\'Train Loss\':>10} | {\'Train Acc\':>9} | {\'Test Acc\':>8} | {\'Promoted\':>8} | "\\\n      f"{\'VRAM MB\':>8}'
        )
        # Update the line of hyphens to be longer
        source = source.replace('print("-" * 70)', 'print("-" * 85)')
        source = source.replace('print("\\n" + "=" * 70)', 'print("\\n" + "=" * 85)')
        
        cell['source'] = [line + '\n' if not line.endswith('\n') else line for line in source.split('\n')][:-1]

with open('vgg16_cifar10_tf32_fp8.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
