"""
nn_native.py — Native Hardware-Accelerated Precision Layer Wrappers.

Menyediakan NativePrecisionMixin dan wrapper classes (NativeConv2d, NativeLinear)
yang menggunakan native PyTorch precision (TF32/FP16/FP8) alih-alih
simulasi bit-truncation dari qtorch3.

Setiap layer bisa di-assign ke salah satu mode:
  - "base_tf32" : Operasi FP32 biasa (TF32 aktif via flag global)
  - "base_fp16" : Operasi di-wrap dengan torch.cuda.amp.autocast(float16)
  - "low_fp8"   : Operasi menggunakan native FP8 casting + delayed scaling
"""

from ext3.typing import *
from ext3.core.include.native_precision import (
    NativePrecisionMode, 
    FP8Config,
    FP8ScalingManager, 
    NativePrecisionContext,
    check_fp8_support,
    fp8_cast_forward,
)
from ext3.core import EModl
from ext3.core.include import Ttype

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from collections import deque

__all__ = [
    'NativePrecisionMixin',
    'NativeConv2d',
    'NativeLinear',
    'NativeBatchNorm2d',
    'NativeReLU',
    'NativeMaxPool2d',
    'NativeAdaptiveAvgPool2d',
    'NativeDropout',
]

# ============================================================
# Global FP8 Scaling Manager (shared across all layers)
# ============================================================
_global_fp8_manager = FP8ScalingManager(
    history_len=FP8Config.DEFAULT_HISTORY_LEN,
    margin=FP8Config.DEFAULT_MARGIN,
)

def get_fp8_manager() -> FP8ScalingManager:
    """Ambil global FP8 scaling manager."""
    return _global_fp8_manager

def reset_fp8_manager() -> None:
    """Reset semua scaling state (panggil saat awal training baru)."""
    _global_fp8_manager.reset()


# ============================================================
# NativePrecisionMixin
# ============================================================
class NativePrecisionMixin:
    """
    Mixin yang menambahkan native hardware-accelerated precision ke module.
    
    Setiap module yang menggunakan mixin ini mendapat:
    - Properti `_native_mode` untuk mengatur precision mode
    - Method `set_native_precision()` untuk mengubah mode saat runtime
    - Method `_native_forward_wrapper()` yang mem-dispatch ke handler 
      yang sesuai berdasarkan mode
    
    FP8 Forward Pipeline:
    1. Ambil scale factor dari FP8ScalingManager (delayed scaling)
    2. Cast input ke FP8 E4M3 dengan scale
    3. Cast weight ke FP8 E4M3 dengan scale
    4. Eksekusi operasi (dequantize ke FP32 → matmul → atau scaled_mm)
    5. Update amax history untuk scaling iterasi berikutnya
    
    FP8 Backward (Gradient):
    - Otomatis ditangani: backward tetap dalam FP32 (autograd default)
    - Gradient weight/input tidak di-cast ke FP8 di sini,
      karena gradient scaling sudah ditangani oleh GradScaler level training loop
    """
    
    _native_mode: Opt[NativePrecisionMode] = None
    _layer_uid: str = ""
    _fp8_fallback_warned: bool = False
    
    def set_native_precision(self, mode: Union[str, NativePrecisionMode]) -> None:
        """
        Assign native precision mode ke layer ini.
        
        Args:
            mode: "base_tf32", "base_fp16", "low_fp8", atau NativePrecisionMode enum.
        """
        if isinstance(mode, str):
            mode = NativePrecisionMode(mode)
        
        # Validate FP8 support
        if mode == NativePrecisionMode.LOW_FP8 and not check_fp8_support():
            if not self._fp8_fallback_warned:
                warnings.warn(
                    f"[{self._layer_uid}] FP8 tidak didukung pada GPU ini. "
                    f"Fallback ke BASE_FP16.",
                    RuntimeWarning
                )
                self._fp8_fallback_warned = True
            mode = NativePrecisionMode.BASE_FP16
        
        self._native_mode = mode
    
    def get_native_precision(self) -> Opt[NativePrecisionMode]:
        """Return current precision mode."""
        return self._native_mode
    
    def _ensure_layer_uid(self) -> str:
        """Generate unique layer ID jika belum ada."""
        if not self._layer_uid:
            # Gunakan class name + id() sebagai uid
            self._layer_uid = f"{type(self).__name__}_{id(self)}"
        return self._layer_uid
    
    def _native_forward_wrapper(
        self, 
        original_forward_fn: Callable,
        *args, 
        **kwargs
    ) -> torch.Tensor:
        """
        Dispatch forward pass berdasarkan precision mode.
        
        Args:
            original_forward_fn: Fungsi forward asli dari parent class.
            *args, **kwargs: Arguments untuk forward.
            
        Returns:
            Output tensor.
        """
        native_mode_obj = None
        # Gunakan dynamic precision dari Pasn/EModl jika tersedia
        if hasattr(self, 'info_ts') and hasattr(self.info_ts[Ttype.Y], 'dtype') and len(self.info_ts[Ttype.Y].dtype) > 0:
            target_dtype = self.info_ts[Ttype.Y].dtype[0]
            if hasattr(target_dtype, 'to_native'):
                native_mode_obj = target_dtype.to_native().mode_name
                
        # Fallback ke static mode jika disetel secara manual
        mode = native_mode_obj if native_mode_obj is not None else (self._native_mode.value if self._native_mode else None)
        
        if mode is None or mode == NativePrecisionMode.BASE_TF32.value:
            # ---- TF32 Mode ----
            # Tidak perlu wrapper apapun.
            # TF32 aktif secara global via torch.backends.cuda.matmul.allow_tf32
            return original_forward_fn(*args, **kwargs)
        
        elif mode == NativePrecisionMode.BASE_FP16.value:
            # ---- FP16 Mode ----
            # Bungkus komputasi dengan autocast FP16
            with torch.cuda.amp.autocast(dtype=torch.float16):
                return original_forward_fn(*args, **kwargs)
        
        elif mode == NativePrecisionMode.LOW_FP8.value:
            # ---- FP8 Mode ----
            return self._fp8_forward(original_forward_fn, *args, **kwargs)
        
        else:
            # Fallback: eksekusi normal
            return original_forward_fn(*args, **kwargs)
    
    def _fp8_forward(
        self, 
        original_forward_fn: Callable,
        *args, 
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass menggunakan FP8 dengan delayed scaling.
        
        Pipeline:
        1. Ambil input tensor (arg pertama)
        2. Hitung scale, cast input+weight ke FP8
        3. Dequantize → eksekusi operasi → output dalam FP32
        4. Update amax history
        
        CATATAN: Saat ini, implementasi ini melakukan:
          - FP8 quantize → dequantize → standard op
          - Ini memberikan efek precision FP8 sambil menjaga kompatibilitas
          - Pada GPU dengan torch._scaled_mm support, bisa dioptimalkan
            untuk true FP8 matmul
        """
        layer_uid = self._ensure_layer_uid()
        manager = get_fp8_manager()
        
        # Ambil input tensor
        if len(args) > 0:
            x = args[0]
        else:
            # Coba ambil dari kwargs
            x = kwargs.get('input', kwargs.get('x', None))
            if x is None:
                return original_forward_fn(*args, **kwargs)
        
        if not isinstance(x, torch.Tensor) or not x.is_cuda:
            return original_forward_fn(*args, **kwargs)
        
        # --- Input FP8 Casting ---
        fwd_uid = f"{layer_uid}.fwd_input"
        scale_input = manager.compute_scale(fwd_uid, FP8Config.FWD_DTYPE)
        
        # Update amax SEBELUM casting (gunakan nilai FP32 asli)
        manager.update_amax(x, fwd_uid, FP8Config.FWD_DTYPE)
        
        # Inject overflow_count into undovr for Pasn Tracking
        if hasattr(self, 'info_ts') and hasattr(self.info_ts[Ttype.Y], 'undovr') and len(self.info_ts[Ttype.Y].undovr) > 0:
            with torch.no_grad():
                overflow_count = (x.abs() > FP8Config.E4M3_MAX).sum().float()
                self.info_ts[Ttype.Y].undovr[0] = torch.tensor([0.0, overflow_count], device=x.device)
        
        # Cast to FP8 → immediately dequantize back
        # Ini mensimulasikan efek precision loss dari FP8
        # sambil tetap kompatibel dengan semua PyTorch ops
        x_fp8, inv_scale = fp8_cast_forward(x, scale_input, FP8Config.FWD_DTYPE)
        x_dequant = x_fp8.to(x.dtype) * inv_scale
        
        # --- Weight FP8 Casting (jika module punya weight) ---
        weight_restored = False
        original_weight_data = None
        
        if hasattr(self, 'weight') and self.weight is not None:  # type: ignore
            weight = self.weight  # type: ignore
            wt_uid = f"{layer_uid}.fwd_weight"
            scale_weight = manager.compute_scale(wt_uid, FP8Config.FWD_DTYPE)
            manager.update_amax(weight.data, wt_uid, FP8Config.FWD_DTYPE)
            
            # Cast weight ke FP8 → dequantize
            w_fp8, w_inv_scale = fp8_cast_forward(
                weight.data, scale_weight, FP8Config.FWD_DTYPE
            )
            original_weight_data = weight.data
            weight.data = w_fp8.to(weight.dtype) * w_inv_scale
            weight_restored = True
        
        # --- Execute the original operation ---
        try:
            if len(args) > 0:
                new_args = (x_dequant,) + args[1:]
                output = original_forward_fn(*new_args, **kwargs)
            else:
                kwargs_copy = dict(kwargs)
                if 'input' in kwargs_copy:
                    kwargs_copy['input'] = x_dequant
                elif 'x' in kwargs_copy:
                    kwargs_copy['x'] = x_dequant
                output = original_forward_fn(**kwargs_copy)
        finally:
            # Restore original weight
            if weight_restored and original_weight_data is not None:
                self.weight.data = original_weight_data  # type: ignore
        
        return output


# ============================================================
# Native Precision Layer Wrappers
# ============================================================
# Layer yang memiliki weight (Conv2d, Linear) → mendapat full FP8 support
# Layer tanpa weight (ReLU, Pooling, BN) → hanya TF32/FP16 support

class NativeConv2d(NativePrecisionMixin, nn.Conv2d, EModl):
    """
    Conv2d dengan native hardware precision support.
    
    Mode yang didukung:
    - base_tf32: Conv2d biasa, TF32 aktif via global flag
    - base_fp16: Conv2d dibungkus autocast FP16
    - low_fp8: Input dan weight di-cast ke FP8 E4M3, 
               dequantize, lalu eksekusi conv2d
    """
    
    def __init__(self, *args, **kwargs):
        # Extract native_mode jika diberikan sebelum super().__init__
        native_mode = kwargs.pop('native_mode', None)
        super(NativeConv2d, self).__init__(*args, **kwargs)
        self._layer_uid = f"NativeConv2d_{id(self)}"
        if native_mode is not None:
            self.set_native_precision(native_mode)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass dengan precision dispatch."""
        out = self._native_forward_wrapper(
            self._conv_forward_native, x
        )
        return out
    
    def _conv_forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """Execute conv2d operation."""
        return F.conv2d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups
        )


class NativeLinear(NativePrecisionMixin, nn.Linear, EModl):
    """
    Linear layer dengan native hardware precision support.
    
    Mode yang didukung:
    - base_tf32: Linear biasa, TF32 aktif via global flag
    - base_fp16: Linear dibungkus autocast FP16
    - low_fp8: Input dan weight di-cast ke FP8 E4M3,
               dequantize, lalu eksekusi linear
    """
    
    def __init__(self, *args, **kwargs):
        native_mode = kwargs.pop('native_mode', None)
        super(NativeLinear, self).__init__(*args, **kwargs)
        self._layer_uid = f"NativeLinear_{id(self)}"
        if native_mode is not None:
            self.set_native_precision(native_mode)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass dengan precision dispatch."""
        out = self._native_forward_wrapper(
            self._linear_forward_native, x
        )
        return out
    
    def _linear_forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """Execute linear operation."""
        return F.linear(x, self.weight, self.bias)


# ============================================================
# Non-compute-heavy layers: hanya TF32/FP16, tidak perlu FP8
# ============================================================

class NativeBatchNorm2d(nn.BatchNorm2d, EModl):
    """
    BatchNorm2d — selalu beroperasi di FP32 (best practice).
    
    BatchNorm memerlukan presisi tinggi untuk running stats (mean, variance).
    Bahkan di mixed-precision training, BN selalu di FP32.
    """
    pass


class NativeReLU(nn.ReLU, EModl):
    """ReLU — operasi element-wise, tidak memerlukan precision khusus."""
    pass


class NativeMaxPool2d(nn.MaxPool2d, EModl):
    """MaxPool2d — operasi comparison, presisi tidak berpengaruh."""
    pass


class NativeAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d, EModl):
    """AdaptiveAvgPool2d — averaging, tetap FP32 untuk akurasi."""
    pass


class NativeDropout(nn.Dropout, EModl):
    """Dropout — mask operation, presisi tidak berpengaruh."""
    pass
