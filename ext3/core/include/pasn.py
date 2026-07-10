"""
pasn.py — Native Precision Assignment (APA)

Replikasi arsitektur Adaptive Precision Assignment asli dari wonyeol/mixed-prec-train,
disesuaikan untuk Native Hardware Precision (FP16/TF32/FP8) alih-alih simulator.
"""

from ext3.typing import *
import torch
import torch.nn as nn
import random

from ext3.core.include.native_precision import NativePrecisionMode

__all__ = ['NativePasn']

class NativePasn:
    """
    Manajer yang mengatur Siklus Grouping -> Demotion -> Promotion 
    sesuai dengan algoritma asli (menggantikan EModlObjMgr dan Pasn).
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.layers: List[nn.Module] = []
        self.group_sizes: List[int] = []
        self._group_layers()
        
    def _group_layers(self) -> None:
        """
        Fase 1: Tensor Grouping
        Mengidentifikasi dan mengelompokkan layer-layer komputasi berat 
        (Conv2d, Linear) yang memiliki `set_native_precision`.
        """
        self.layers = []
        self.group_sizes = []
        from ext3.nn.nn_native import NativeConv2d, NativeLinear
        for name, module in self.model.named_modules():
            if isinstance(module, (NativeConv2d, NativeLinear)):
                self.layers.append(module)
                # Estimasi ukuran berdasarkan parameter (mengikuti pendekatan get_numels_by_id)
                numel = sum(p.numel() for p in module.parameters() if p.requires_grad)
                # Jika tidak ada weight (misal freeze), gunakan 1 sebagai placeholder
                self.group_sizes.append(numel if numel > 0 else 1)

    def apply_demotion(
        self, 
        scheme: str, 
        r_min: float, 
        r_max: float, 
        base_mode: NativePrecisionMode, 
        low_mode: NativePrecisionMode,
        protect_ends: bool = True,
        seed: int = -1
    ) -> Dict[str, str]:
        """
        Fase 2: Precision Demotion (Dilakukan di awal training)
        Memaksa sebagian layer untuk turun ke low_mode berdasarkan argumen.
        
        Args:
            scheme: 'rand', 'topr_dec', 'topr_inc'
            r_min: rasio minimum (0.0 - 1.0)
            r_max: rasio maksimum (0.0 - 1.0)
            protect_ends: Jika True, layer pertama dan terakhir (pa_upd_no_end) selalu di base_mode.
            
        Returns:
            Dictionary statistik perubahan.
        """
        # 1. Default semua ke base_mode
        for layer in self.layers:
            layer.set_native_precision(base_mode)
            
        if not self.layers:
            return {}

        # 2. Hitung total size
        total_size = sum(self.group_sizes)
        
        # 3. Urutkan berdasarkan scheme
        # List of (index, size, ratio)
        rel_sizes = [(i, size, size / total_size) for i, size in enumerate(self.group_sizes)]
        
        if scheme == 'rand':
            if seed >= 0:
                random.seed(seed)
            random.shuffle(rel_sizes)
        elif scheme == 'topr_dec':
            # Sort by size descending
            rel_sizes.sort(key=lambda x: x[1], reverse=True)
        elif scheme == 'topr_inc':
            # Sort by size ascending
            rel_sizes.sort(key=lambda x: x[1], reverse=False)
        else:
            print(f"[NativePasn] Warning: Unknown scheme '{scheme}', fallback to base_mode only.")
            return {}

        # 4. Pilih layer untuk di-demote berdasarkan r_min dan r_max
        demote_indices = set()
        current_r = 0.0
        
        # Sesuai logika get_ids_chosen: accumulate ratio sampai r_min terpenuhi
        for idx, size, ratio in rel_sizes:
            if current_r >= r_min:
                break
            demote_indices.add(idx)
            current_r += ratio

        # 5. Terapkan demotion dan proteksi
        demoted_count = 0
        for i, layer in enumerate(self.layers):
            if protect_ends and (i == 0 or i == len(self.layers) - 1):
                # Protect first and last layer (pa_upd_no_end)
                layer.set_native_precision(base_mode)
                continue
                
            if i in demote_indices:
                layer.set_native_precision(low_mode)
                demoted_count += 1
                
        return {
            'total_layers': len(self.layers),
            'demoted_layers': demoted_count,
            'base_mode': str(base_mode),
            'low_mode': str(low_mode),
            'scheme': scheme,
            'target_ratio': r_min,
            'actual_ratio': current_r
        }

    def check_and_promote(self, base_mode: NativePrecisionMode) -> int:
        """
        Fase 3: Precision Promotion (Dipanggil selama training loop)
        Mengevaluasi amax spikes di FP8ScalingManager. Jika layer FP8 mengalami
        lonjakan ekstrim (menyimulasikan overflow ratio yang tinggi), maka layer 
        tersebut dipromosikan kembali ke base_mode.
        
        Returns:
            Jumlah layer yang berhasil di-promote pada iterasi ini.
        """
        from ext3.nn.nn_native import get_fp8_manager
        manager = get_fp8_manager()
        promoted_count = 0
        
        for layer in self.layers:
            # Hanya proses layer yang sedang dalam FP8
            if layer.get_native_precision() == NativePrecisionMode.LOW_FP8:
                layer_uid = layer._ensure_layer_uid()
                # Kita perlu cek fwd_input dan fwd_weight
                fwd_uid = f"{layer_uid}.fwd_input"
                wt_uid = f"{layer_uid}.fwd_weight"
                
                # Jika ada spike terdeteksi oleh manager, lakukan promotion
                if manager.has_spike(fwd_uid) or manager.has_spike(wt_uid):
                    layer.set_native_precision(base_mode)
                    manager.clear_spike(fwd_uid)
                    manager.clear_spike(wt_uid)
                    promoted_count += 1
                    print(f"*** [NativePasn] PROMOTION: Layer {layer_uid} mengalami AMAX spike. Promoted to {base_mode}.")
                    
        return promoted_count
