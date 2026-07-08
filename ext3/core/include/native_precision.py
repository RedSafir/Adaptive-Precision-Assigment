"""
native_precision.py — Native Hardware-Accelerated Precision Manager.

Menyediakan infrastruktur untuk FP8 scaling (delayed scaling),
FP16 autocast context, dan runtime GPU capability detection.

Menggantikan simulasi bit-truncation dari qtorch3 dengan
native PyTorch 2.1+ operations.
"""

from ext3.typing import *

import torch
import enum
import warnings
import math
from collections import deque

__all__ = [
    'NativePrecisionMode',
    'FP8Config',
    'FP8ScalingManager',
    'NativePrecisionContext',
    'check_fp8_support',
    'fp8_cast_forward',
    'fp8_cast_backward',
    'fp8_scaled_mm',
]

# ============================================================
# Enum: NativePrecisionMode
# ============================================================
class NativePrecisionMode(enum.Enum):
    """
    Precision mode yang bisa di-assign ke setiap layer.
    
    - BASE_TF32  : Operasi dalam FP32 (TF32 aktif via flag global untuk matmul).
    - BASE_FP16  : Operasi dibungkus torch.cuda.amp.autocast(float16).
    - LOW_FP8    : Operasi menggunakan native FP8 dtypes dengan per-tensor scaling.
    """
    BASE_TF32 = "base_tf32"
    BASE_FP16 = "base_fp16"
    LOW_FP8   = "low_fp8"


# ============================================================
# FP8 Configuration Constants
# ============================================================
class FP8Config:
    """Konstanta dan konfigurasi untuk FP8 operations."""
    
    # Native FP8 dtypes (PyTorch 2.1+)
    FWD_DTYPE = torch.float8_e4m3fn   # Forward pass / weights: E4M3 (range ±448)
    BWD_DTYPE = torch.float8_e5m2     # Backward pass / grads:  E5M2 (range ±57344)
    
    # Maximum representable values (untuk scaling computation)
    E4M3_MAX = 448.0       # max finite value of float8_e4m3fn
    E5M2_MAX = 57344.0     # max finite value of float8_e5m2
    
    # Delayed scaling defaults
    DEFAULT_HISTORY_LEN = 16    # Jumlah iterasi amax history
    DEFAULT_MARGIN = 0.0        # Safety margin (log2 units)
    
    # Scale bounds (mencegah scale terlalu ekstrem)
    SCALE_MIN = 1e-12
    SCALE_MAX = 1e12


# ============================================================
# GPU Capability Check
# ============================================================
_fp8_support_cache: Opt[bool] = None

def check_fp8_support() -> bool:
    """
    Check apakah GPU saat ini mendukung native FP8 operations.
    
    FP8 memerlukan:
    - CUDA available
    - GPU dengan compute capability >= 8.9 (Ada Lovelace) atau >= 9.0 (Hopper)
    - PyTorch >= 2.1 dengan float8 dtype support
    
    Returns:
        bool: True jika FP8 native didukung.
    """
    global _fp8_support_cache
    if _fp8_support_cache is not None:
        return _fp8_support_cache
    
    # Check 1: CUDA available
    if not torch.cuda.is_available():
        _fp8_support_cache = False
        return False
    
    # Check 2: PyTorch version & dtype existence
    try:
        _ = torch.float8_e4m3fn
        _ = torch.float8_e5m2
    except AttributeError:
        warnings.warn(
            "PyTorch version does not support float8 dtypes. "
            "Upgrade ke PyTorch >= 2.1. FP8 path akan fallback ke FP16.",
            RuntimeWarning
        )
        _fp8_support_cache = False
        return False
    
    # Check 3: GPU compute capability
    try:
        device = torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(device)
        major, minor = capability
        cc = major * 10 + minor
        
        # Ada Lovelace (SM89) atau Hopper (SM90+) diperlukan
        if cc < 89:
            warnings.warn(
                f"GPU compute capability {major}.{minor} < 8.9. "
                f"FP8 native memerlukan Ada Lovelace (8.9) atau Hopper (9.0+). "
                f"FP8 path akan fallback ke FP16.",
                RuntimeWarning
            )
            _fp8_support_cache = False
            return False
    except Exception:
        _fp8_support_cache = False
        return False
    
    # Check 4: Functional test — try creating FP8 tensor
    try:
        test = torch.randn(2, 2, device='cuda')
        _ = test.to(torch.float8_e4m3fn)
        _ = test.to(torch.float8_e5m2)
    except Exception as e:
        warnings.warn(
            f"FP8 tensor creation failed: {e}. Fallback ke FP16.",
            RuntimeWarning
        )
        _fp8_support_cache = False
        return False
    
    _fp8_support_cache = True
    return True


# ============================================================
# FP8 Scaling Manager (Delayed Scaling)
# ============================================================
class FP8ScalingManager:
    """
    Per-tensor scale factor manager menggunakan delayed scaling.
    
    Delayed Scaling Algorithm:
    1. Setiap iterasi, catat amax (max absolute value) dari tensor.
    2. Simpan amax ke circular buffer (history_len iterasi).
    3. Hitung scale = max_representable_value / max(amax_history).
    4. Gunakan scale ini untuk iterasi berikutnya (delayed by 1 step).
    
    Ini lebih stabil daripada per-tensor scaling yang hanya melihat
    batch saat ini, karena outlier sesaat tidak langsung mengubah scale
    secara drastis.
    
    Referensi:
    - FP8 Formats for Deep Learning (Micikevicius et al., 2022)
    - NVIDIA TransformerEngine delayed scaling
    """
    
    def __init__(
        self,
        history_len: int = FP8Config.DEFAULT_HISTORY_LEN,
        margin: float = FP8Config.DEFAULT_MARGIN,
    ):
        """
        Args:
            history_len: Panjang circular buffer untuk amax history.
            margin: Safety margin dalam log2 units. Scale dikurangi 2^margin
                    untuk memberi headroom terhadap overflow.
        """
        self.history_len = history_len
        self.margin = margin
        
        # Per-layer state: key = layer_id (str), value = dict of state
        self._state: Dict[str, dict] = {}
    
    def _get_or_create_state(self, layer_id: str, dtype_max: float) -> dict:
        """Get atau inisialisasi state untuk sebuah layer."""
        if layer_id not in self._state:
            self._state[layer_id] = {
                'amax_history': deque(
                    [dtype_max],  # Init dengan max value agar scale awal = 1.0
                    maxlen=self.history_len
                ),
                'scale': 1.0,           # Current scale factor
                'dtype_max': dtype_max,  # Max representable value of target dtype
            }
        return self._state[layer_id]
    
    def compute_scale(
        self, 
        layer_id: str,
        fp8_dtype: torch.dtype = FP8Config.FWD_DTYPE,
    ) -> float:
        """
        Hitung scale factor untuk layer berdasarkan amax history.
        
        Args:
            layer_id: Identifier unik untuk layer (e.g., "layer.0.conv.fwd")
            fp8_dtype: Target FP8 dtype (E4M3 atau E5M2)
            
        Returns:
            float: Scale factor. tensor_fp8 = (tensor_fp32 * scale).to(fp8_dtype)
        """
        dtype_max = (
            FP8Config.E4M3_MAX if fp8_dtype == FP8Config.FWD_DTYPE 
            else FP8Config.E5M2_MAX
        )
        state = self._get_or_create_state(layer_id, dtype_max)
        return state['scale']
    
    def update_amax(
        self, 
        tensor: torch.Tensor,
        layer_id: str,
        fp8_dtype: torch.dtype = FP8Config.FWD_DTYPE,
    ) -> None:
        """
        Update amax history dan recompute scale factor.
        
        Dipanggil SETELAH operasi forward/backward selesai dengan
        tensor SEBELUM casting ke FP8, sehingga amax dihitung
        dari nilai FP32/FP16 asli.
        
        Args:
            tensor: Tensor yang akan di-quantize (masih FP32/FP16).
            layer_id: Identifier unik untuk layer.
            fp8_dtype: Target FP8 dtype.
        """
        dtype_max = (
            FP8Config.E4M3_MAX if fp8_dtype == FP8Config.FWD_DTYPE 
            else FP8Config.E5M2_MAX
        )
        state = self._get_or_create_state(layer_id, dtype_max)
        
        # Step 1: Compute amax dari tensor saat ini
        with torch.no_grad():
            amax = tensor.abs().max().item()
            # Clamp ke minimum untuk menghindari division by zero
            amax = max(amax, 1e-12)
        
        # Step 2: Update history
        state['amax_history'].append(amax)
        
        # Step 3: Recompute scale berdasarkan max dari history
        history_max = max(state['amax_history'])
        
        # scale = dtype_max / (history_max * 2^margin)
        # Sehingga tensor * scale tidak melebihi representable range
        scale = dtype_max / (history_max * (2.0 ** self.margin))
        
        # Clamp scale ke bounds yang wajar
        scale = max(FP8Config.SCALE_MIN, min(FP8Config.SCALE_MAX, scale))
        
        state['scale'] = scale
    
    def reset(self, layer_id: Opt[str] = None) -> None:
        """Reset state. Jika layer_id=None, reset semua layers."""
        if layer_id is None:
            self._state.clear()
        elif layer_id in self._state:
            del self._state[layer_id]
    
    def get_stats(self, layer_id: str) -> Opt[Dict[str, Any]]:
        """Ambil statistik scaling untuk monitoring."""
        if layer_id not in self._state:
            return None
        state = self._state[layer_id]
        return {
            'scale': state['scale'],
            'amax_history': list(state['amax_history']),
            'history_max': max(state['amax_history']),
            'dtype_max': state['dtype_max'],
        }


# ============================================================
# FP8 Casting Operations
# ============================================================
def fp8_cast_forward(
    tensor: torch.Tensor,
    scale: float,
    fp8_dtype: torch.dtype = FP8Config.FWD_DTYPE,
) -> Tuple[torch.Tensor, float]:
    """
    Cast tensor ke FP8 untuk forward pass dengan scaling.
    
    Pipeline:
      1. scaled = tensor * scale
      2. clamped = clamp(scaled, -max_val, +max_val)
      3. fp8 = clamped.to(fp8_dtype)
    
    Args:
        tensor: Input tensor (FP32 atau FP16).
        scale: Scale factor dari FP8ScalingManager.
        fp8_dtype: Target dtype (default: float8_e4m3fn).
        
    Returns:
        Tuple[fp8_tensor, inverse_scale]:
            - fp8_tensor: Tensor dalam format FP8.
            - inverse_scale: 1.0/scale, untuk dequantization.
    """
    dtype_max = (
        FP8Config.E4M3_MAX if fp8_dtype == FP8Config.FWD_DTYPE 
        else FP8Config.E5M2_MAX
    )
    
    # Scale → clamp → cast
    scaled = tensor.float() * scale  # Ensure FP32 sebelum scaling
    clamped = scaled.clamp(-dtype_max, dtype_max)
    fp8_tensor = clamped.to(fp8_dtype)
    
    inverse_scale = 1.0 / scale
    return fp8_tensor, inverse_scale


def fp8_cast_backward(
    grad: torch.Tensor,
    scale: float,
    fp8_dtype: torch.dtype = FP8Config.BWD_DTYPE,
) -> Tuple[torch.Tensor, float]:
    """
    Cast gradient tensor ke FP8 untuk backward pass.
    
    Menggunakan E5M2 (lebih besar dynamic range) karena
    gradien memiliki distribusi yang lebih lebar dari aktivasi.
    
    Args:
        grad: Gradient tensor (FP32).
        scale: Scale factor dari FP8ScalingManager.
        fp8_dtype: Target dtype (default: float8_e5m2).
        
    Returns:
        Tuple[fp8_grad, inverse_scale]
    """
    return fp8_cast_forward(grad, scale, fp8_dtype)


def fp8_scaled_mm(
    a_fp8: torch.Tensor,
    b_fp8: torch.Tensor,
    a_inverse_scale: float,
    b_inverse_scale: float,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    FP8 matrix multiplication dengan scaling.
    
    Compute: output = (A_fp8 @ B_fp8) * (a_inv_scale * b_inv_scale)
    
    Menggunakan torch._scaled_mm jika tersedia (PyTorch nightly),
    fallback ke manual dequant-then-matmul jika tidak.
    
    Args:
        a_fp8: Left matrix in FP8 format.
        b_fp8: Right matrix in FP8 format (akan di-transpose jika perlu).
        a_inverse_scale: 1/scale_a
        b_inverse_scale: 1/scale_b
        output_dtype: Output dtype (default FP32).
        
    Returns:
        Result tensor in output_dtype.
    """
    # Try torch._scaled_mm (available in PyTorch 2.2+ nightly with FP8 support)
    if hasattr(torch, '_scaled_mm'):
        try:
            # torch._scaled_mm expects scale tensors
            scale_a = torch.tensor(a_inverse_scale, dtype=torch.float32, device=a_fp8.device)
            scale_b = torch.tensor(b_inverse_scale, dtype=torch.float32, device=b_fp8.device)
            
            result = torch._scaled_mm(
                a_fp8, 
                b_fp8.t() if b_fp8.dim() == 2 else b_fp8,
                scale_a=scale_a,
                scale_b=scale_b,
                out_dtype=output_dtype,
            )
            # torch._scaled_mm returns a tuple (result, amax) in some versions
            if isinstance(result, tuple):
                result = result[0]
            return result
        except Exception:
            pass  # Fallback ke manual method
    
    # Fallback: dequantize → matmul
    a_dequant = a_fp8.to(output_dtype) * a_inverse_scale
    b_dequant = b_fp8.to(output_dtype) * b_inverse_scale
    return torch.matmul(a_dequant, b_dequant)


# ============================================================
# Native Precision Context Manager
# ============================================================
class NativePrecisionContext:
    """
    Context manager yang mengatur precision mode untuk sebuah layer.
    
    Usage:
        ctx = NativePrecisionContext(NativePrecisionMode.BASE_FP16)
        with ctx:
            output = layer(input)
    """
    
    def __init__(self, mode: NativePrecisionMode):
        self.mode = mode
        self._autocast_ctx = None
    
    def __enter__(self):
        if self.mode == NativePrecisionMode.BASE_FP16:
            self._autocast_ctx = torch.cuda.amp.autocast(dtype=torch.float16)
            self._autocast_ctx.__enter__()
        elif self.mode == NativePrecisionMode.BASE_TF32:
            # TF32 diaktifkan via global flag, tidak perlu context
            pass
        elif self.mode == NativePrecisionMode.LOW_FP8:
            # FP8 ditangani secara manual per-operation, bukan via context
            pass
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._autocast_ctx is not None:
            self._autocast_ctx.__exit__(exc_type, exc_val, exc_tb)
            self._autocast_ctx = None
        return False


# ============================================================
# Utility: Enable TF32 Globally
# ============================================================
def enable_tf32() -> None:
    """
    Aktifkan TF32 untuk semua matmul dan convolution operations.
    
    TF32 menggunakan 10-bit mantissa (vs FP32 23-bit) tapi
    menjaga exponent range FP32. Pada Ampere+ GPUs, ini memberikan
    ~2-3x speedup untuk matmul tanpa perubahan kode.
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def disable_tf32() -> None:
    """Disable TF32 (kembali ke FP32 penuh)."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
