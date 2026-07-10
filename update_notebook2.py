import json

with open('vgg16_cifar10_fp16_fp8.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update Cell 2 (Imports)
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'Cell 2: Imports' in ''.join(cell['source']):
        source = ''.join(cell['source'])
        if 'NativePasn' not in source:
            source = source.replace('    check_fp8_support, enable_tf32, disable_tf32,\n)', '    check_fp8_support, enable_tf32, disable_tf32,\n)\nfrom ext3.core.include.pasn import NativePasn\n')
            cell['source'] = [line + '\n' if not line.endswith('\n') else line for line in source.split('\n')][:-1]

# Update Cell 3 (Configuration)
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'Cell 3: Configuration' in ''.join(cell['source']):
        new_source = '''# ============================================================
# Cell 3: Configuration
# ============================================================
CONFIG = {
    'batch_size': 128,
    'epochs': 50,
    'lr': 0.01,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'num_workers': 0,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # GradScaler config — initial scale dan growth interval
    'grad_scaler_init_scale': 2.**16,
    'grad_scaler_growth_interval': 2000,
    
    # --- Native APA (Adaptive Precision Assignment) Config ---
    'pa_upd_schm': 'topr_dec',    # Demote layer terbesar lebih dulu
    'pa_upd_rmin': 0.3,           # Target 30% dari total parameter turun ke FP8
    'pa_upd_rmax': 0.4,           
    'spike_threshold': 0.5,       # Ambang batas lonjakan AMAX (50% dari rata-rata).
}

# NOTE: TF32 NOT enabled here — kita menggunakan FP16 sebagai base precision
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

print('[CONFIG] Training Configuration:')
for k, v in CONFIG.items():
    print(f'  {k:30s}: {v}')
print(f'\\n[CONFIG] TF32 matmul  : {torch.backends.cuda.matmul.allow_tf32}')
print(f'[CONFIG] TF32 cuDNN  : {torch.backends.cudnn.allow_tf32}')
print(f'[CONFIG] Base mode   : FP16 (autocast + GradScaler)')
print(f'[CONFIG] Low mode    : FP8 E4M3 (forward) / E5M2 (backward)')
'''
        cell['source'] = [line + '\n' for line in new_source.split('\n')][:-1]

# Update Cell 5 (Precision Assignment)
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'Cell 5: Precision Assignment' in ''.join(cell['source']):
        new_source = '''# ============================================================
# Cell 5: Precision Assignment Function via NativePasn
# ============================================================

def assign_precision(model, config) -> NativePasn:
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
    
    # 3. Apply Demotion (BASE is FP16, LOW is FP8)
    stats = pasn.apply_demotion(
        scheme=config['pa_upd_schm'],
        r_min=config['pa_upd_rmin'],
        r_max=config['pa_upd_rmax'],
        base_mode=NativePrecisionMode.BASE_FP16,
        low_mode=NativePrecisionMode.LOW_FP8,
        protect_ends=True
    )
    
    print(f"  Scheme          : {stats.get('scheme')}")
    print(f"  Target Ratio    : {stats.get('target_ratio')}")
    print(f"  Actual Ratio    : {stats.get('actual_ratio'):.3f}")
    print(f"  Demoted Layers  : {stats.get('demoted_layers')} / {stats.get('total_layers')}")
    print("=" * 60)
    
    return pasn

print('[INFO] assign_precision() function defined.')
'''
        cell['source'] = [line + '\n' for line in new_source.split('\n')][:-1]

# Update Cell 8 (Train Epoch)
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'Cell 8: Training Function' in ''.join(cell['source']):
        new_source = '''# ============================================================
# Cell 8: Training Function with GradScaler & NativePasn
# ============================================================

def train_epoch(model, loader, criterion, optimizer, scaler, device, stability_checker, pasn_manager: NativePasn):
    """
    Train model untuk satu epoch dengan FP16 autocast + GradScaler + NativePasn Promotion.
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    overflow_count = 0
    promoted_count = 0
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        # ---- Forward with autocast FP16 ----
        with torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        # ---- Backward with GradScaler ----
        scaler.scale(loss).backward()
        
        # ---- NATIVE APA: Precision Promotion ----
        # Mengecek apakah ada layer FP8 yang mengalami lonjakan amax ekstrim.
        # Dipromosikan ke BASE_FP16.
        promoted = pasn_manager.check_and_promote(NativePrecisionMode.BASE_FP16)
        promoted_count += promoted
        
        # ---- Unscale gradients & Clipping ----
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # ---- Optimizer step & Scale Update ----
        scaler.step(optimizer)
        scale_before = scaler.get_scale()
        scaler.update()
        scale_after = scaler.get_scale()
        
        if scale_after < scale_before:
            overflow_count += 1
        
        # ---- Accumulate metrics ----
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    
    stability_checker.check(avg_loss)
    
    return avg_loss, accuracy, overflow_count, promoted_count

print('[INFO] train_epoch() function defined.')
'''
        cell['source'] = [line + '\n' for line in new_source.split('\n')][:-1]


# Update Cell 11 (Main Loop)
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'Cell 11: Main Loop' in ''.join(cell['source']) or 'Cell 11: Main Training Loop' in ''.join(cell['source']):
        source = ''.join(cell['source'])
        source = source.replace('assign_precision(model, CONFIG)', 'pasn_manager = assign_precision(model, CONFIG)')
        source = source.replace(
            'train_loss, train_acc, n_overflow = train_epoch(\n        model, trainloader, criterion, optimizer, scaler, device, stability_checker\n    )',
            'train_loss, train_acc, n_overflow, n_promotions = train_epoch(\n        model, trainloader, criterion, optimizer, scaler, device, stability_checker, pasn_manager\n    )'
        )
        source = source.replace("scaler_scales = []", "scaler_scales = []\npromotion_counts = []")
        source = source.replace("scaler_scales.append(current_scale)", "scaler_scales.append(current_scale)\n    promotion_counts.append(n_promotions)")
        
        source = source.replace(
            'print(f"Epoch {epoch:3d}/{CONFIG[\'epochs\']} | Train Loss: {train_loss:6.4f} | Train Acc: {train_acc:6.2f}% | Test Acc: {test_acc:6.2f}% | VRAM: {current_vram:7.1f}MB | Overflow: {n_overflow:3d} | Scale: {current_scale:.0f} | Time: {epoch_time:4.1f}s{best_str}")',
            'print(f"Epoch {epoch:3d}/{CONFIG[\'epochs\']} | Train Loss: {train_loss:6.4f} | Train Acc: {train_acc:6.2f}% | Test Acc: {test_acc:6.2f}% | Promoted: {n_promotions:2d} | VRAM: {current_vram:7.1f}MB | Overflow: {n_overflow:3d} | Scale: {current_scale:.0f} | Time: {epoch_time:4.1f}s{best_str}")'
        )
        
        cell['source'] = [line + '\n' if not line.endswith('\n') else line for line in source.split('\n')][:-1]

with open('vgg16_cifar10_fp16_fp8.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
