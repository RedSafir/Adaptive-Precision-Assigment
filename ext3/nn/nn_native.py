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
from ext3.core import EModl, EFuncMgr
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
    'flatten',
    'permute',
    'reshape',
    'view',
]

# functional wrappers to keep graph tracking alive
flatten: Callable = EFuncMgr.gen(torch.flatten)
permute: Callable = EFuncMgr.gen(torch.Tensor.permute)
reshape: Callable = EFuncMgr.gen(torch.Tensor.reshape)
view: Callable = EFuncMgr.gen(torch.Tensor.view)

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
# Custom Autograd Functions for Hardware-Accelerated FP8
# ============================================================

class FP8LinearFunction(torch.autograd.Function):
    """Custom autograd untuk operasi Linear FP8 menggunakan torch._scaled_mm."""
    
    @staticmethod
    def forward(ctx, x, weight, bias, scale_x, scale_w):
        # Save inputs for backward (done in high precision for stability)
        ctx.save_for_backward(x, weight, bias)
        
        # Sourcing from delayed scaling manager
        inv_scale_x = 1.0 / scale_x
        inv_scale_w = 1.0 / scale_w
        
        # Cast input & weight to FP8
        x_fp8 = (x * scale_x).clamp(-FP8Config.E4M3_MAX, FP8Config.E4M3_MAX).to(torch.float8_e4m3fn)
        w_fp8 = (weight * scale_w).clamp(-FP8Config.E4M3_MAX, FP8Config.E4M3_MAX).to(torch.float8_e4m3fn)
        
        # Fallback to CPU-based float matmul if running on CPU
        if not x.is_cuda:
            x_dequant = x_fp8.to(x.dtype) * inv_scale_x
            w_dequant = w_fp8.to(weight.dtype) * inv_scale_w
            out = x_dequant.matmul(w_dequant.t())
            if bias is not None:
                out = out + bias
            return out
            
        device = x.device
        scale_x_tensor = torch.tensor([inv_scale_x], device=device, dtype=torch.float32)
        scale_w_tensor = torch.tensor([inv_scale_w], device=device, dtype=torch.float32)
        
        try:
            out = torch._scaled_mm(
                x_fp8,
                w_fp8.t(),
                scale_a=scale_x_tensor,
                scale_b=scale_w_tensor,
                out_dtype=x.dtype
            )
            if isinstance(out, tuple):
                out = out[0]
        except Exception as e:
            # Fallback to simulated FP8 on older/unsupported CUDA devices
            x_dequant = x_fp8.to(x.dtype) * inv_scale_x
            w_dequant = w_fp8.to(weight.dtype) * inv_scale_w
            out = x_dequant.matmul(w_dequant.t())
            
        if bias is not None:
            out = out + bias
            
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        
        grad_x = None
        grad_weight = None
        grad_bias = None
        
        # Squeeze/cast inputs and grad_output to the same dtype (target_dtype)
        # to prevent type mismatch errors during backward pass.
        target_dtype = x.dtype
        grad_output_c = grad_output.to(target_dtype)
        weight_c = weight.to(target_dtype)
        x_c = x.to(target_dtype)
        
        # High-precision backward pass via standard PyTorch operators
        if ctx.needs_input_grad[0]:
            grad_x = grad_output_c.matmul(weight_c).to(x.dtype)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output_c.t().matmul(x_c).to(weight.dtype)
        if ctx.needs_input_grad[2] and bias is not None:
            grad_bias = grad_output_c.sum(dim=0).to(bias.dtype)
            
        return grad_x, grad_weight, grad_bias, None, None


class FP8Conv2dFunction(torch.autograd.Function):
    """Custom autograd untuk operasi Conv2d FP8 menggunakan im2col + torch._scaled_mm."""
    
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, dilation, groups, scale_x, scale_w):
        ctx.save_for_backward(x, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        
        batch_size, in_channels, in_h, in_w = x.shape
        out_channels, _, kernel_h, kernel_w = weight.shape
        
        # Calculate spatial output dimensions
        out_h = (in_h + 2 * padding[0] - dilation[0] * (kernel_h - 1) - 1) // stride[0] + 1
        out_w = (in_w + 2 * padding[1] - dilation[1] * (kernel_w - 1) - 1) // stride[1] + 1
        
        # Fallback to standard high-precision CPU execution
        if not x.is_cuda:
            return F.conv2d(x, weight, bias, stride, padding, dilation, groups)
            
        # 1. Spatial unfolding (im2col) to convert 4D input to 2D
        x_unfold = F.unfold(
            x, 
            kernel_size=(kernel_h, kernel_w), 
            dilation=dilation, 
            padding=padding, 
            stride=stride
        )
        
        # Transpose and reshape to 2D matrix: (batch_size * L, C * kh * kw)
        x_cols = x_unfold.transpose(1, 2).reshape(-1, in_channels * kernel_h * kernel_w)
        
        # Reshape weight to 2D matrix: (out_channels, C * kh * kw)
        w_mat = weight.reshape(out_channels, -1)
        
        # Sourcing from delayed scaling manager
        inv_scale_x = 1.0 / scale_x
        inv_scale_w = 1.0 / scale_w
        
        # Cast input & weight to FP8
        x_cols_fp8 = (x_cols * scale_x).clamp(-FP8Config.E4M3_MAX, FP8Config.E4M3_MAX).to(torch.float8_e4m3fn)
        w_mat_fp8 = (w_mat * scale_w).clamp(-FP8Config.E4M3_MAX, FP8Config.E4M3_MAX).to(torch.float8_e4m3fn)
        
        device = x.device
        scale_x_tensor = torch.tensor([inv_scale_x], device=device, dtype=torch.float32)
        scale_w_tensor = torch.tensor([inv_scale_w], device=device, dtype=torch.float32)
        
        try:
            out_mat = torch._scaled_mm(
                x_cols_fp8,
                w_mat_fp8.t(),
                scale_a=scale_x_tensor,
                scale_b=scale_w_tensor,
                out_dtype=x.dtype
            )
            if isinstance(out_mat, tuple):
                out_mat = out_mat[0]
        except Exception as e:
            # Fallback for non-supported GPUs (simulate FP8)
            x_cols_dequant = x_cols_fp8.to(x.dtype) * inv_scale_x
            w_mat_dequant = w_mat_fp8.to(weight.dtype) * inv_scale_w
            out_mat = x_cols_dequant.matmul(w_mat_dequant.t())
            
        # Reshape back to 4D spatial tensor
        out_tensor = out_mat.reshape(batch_size, out_h * out_w, out_channels)
        out_tensor = out_tensor.transpose(1, 2).reshape(batch_size, out_channels, out_h, out_w)
        
        if bias is not None:
            out_tensor = out_tensor + bias.view(1, -1, 1, 1)
            
        return out_tensor

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        groups = ctx.groups
        
        grad_x = None
        grad_weight = None
        grad_bias = None
        
        # Squeeze/cast inputs and grad_output to the same dtype (target_dtype)
        # to prevent type mismatch errors during backward pass.
        target_dtype = x.dtype
        grad_output_c = grad_output.to(target_dtype)
        weight_c = weight.to(target_dtype)
        x_c = x.to(target_dtype)
        
        # Explicit stable gradient calculations using PyTorch built-in gradient utilities
        if ctx.needs_input_grad[0]:
            grad_x = torch.nn.grad.conv2d_input(
                x_c.shape, weight_c, grad_output_c,
                stride=stride, padding=padding, dilation=dilation, groups=groups
            )
            grad_x = grad_x.to(x.dtype)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(
                x_c, weight_c.shape, grad_output_c,
                stride=stride, padding=padding, dilation=dilation, groups=groups
            )
            grad_weight = grad_weight.to(weight.dtype)
        if ctx.needs_input_grad[2] and bias is not None:
            grad_bias = grad_output_c.sum(dim=(0, 2, 3)).to(bias.dtype)
            
        return grad_x, grad_weight, grad_bias, None, None, None, None, None, None


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
        """
        native_mode_obj = None
        if hasattr(self, 'info_ts') and hasattr(self.info_ts[Ttype.Y], 'dtype') and len(self.info_ts[Ttype.Y].dtype) > 0:
            target_dtype = self.info_ts[Ttype.Y].dtype[0]
            if hasattr(target_dtype, 'to_native'):
                native_mode_obj = target_dtype.to_native().mode_name
                 
        mode = native_mode_obj if native_mode_obj is not None else (self._native_mode.value if self._native_mode else None)
        
        if mode is None or mode == NativePrecisionMode.BASE_TF32.value:
            return original_forward_fn(*args, **kwargs)
        
        elif mode == NativePrecisionMode.BASE_FP16.value:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                return original_forward_fn(*args, **kwargs)
        
        elif mode == NativePrecisionMode.LOW_FP8.value:
            return self._fp8_forward(original_forward_fn, *args, **kwargs)
        
        else:
            return original_forward_fn(*args, **kwargs)
    
    def _fp8_forward(
        self, 
        original_forward_fn: Callable,
        *args, 
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass menggunakan hardware-accelerated FP8.
        """
        layer_uid = self._ensure_layer_uid()
        manager = get_fp8_manager()
        
        # Ambil input tensor
        if len(args) > 0:
            x = args[0]
        else:
            x = kwargs.get('input', kwargs.get('x', None))
            if x is None:
                return original_forward_fn(*args, **kwargs)
        
        if not isinstance(x, torch.Tensor):
            return original_forward_fn(*args, **kwargs)
            
        # Check alignment requirements for FP8 Tensor Cores (Kelipatan 16)
        if isinstance(self, nn.Conv2d):
            kh, kw = self.kernel_size
            k_val = self.in_channels * kh * kw
            if k_val % 16 != 0 or self.out_channels % 16 != 0:
                return original_forward_fn(*args, **kwargs)
        elif isinstance(self, nn.Linear):
            if self.in_features % 16 != 0 or self.out_features % 16 != 0:
                return original_forward_fn(*args, **kwargs)
        else:
            return original_forward_fn(*args, **kwargs)
            
        # --- Update Amax and compute delayed scaling factors ---
        fwd_uid = f"{layer_uid}.fwd_input"
        scale_input = manager.compute_scale(fwd_uid, FP8Config.FWD_DTYPE)
        manager.update_amax(x, fwd_uid, FP8Config.FWD_DTYPE)
        
        # Inject overflow/underflow RATIO into undovr for Pasn Tracking (Ttype.Y = activation)
        if hasattr(self, 'info_ts') and hasattr(self.info_ts[Ttype.Y], 'undovr') and len(self.info_ts[Ttype.Y].undovr) > 0:
            with torch.no_grad():
                numel = max(x.numel(), 1)
                overflow_ratio = (x.abs() > FP8Config.E4M3_MAX).sum().float() / numel
                underflow_ratio = torch.tensor(0.0, device=x.device)  # Underflow tracking placeholder
                self.info_ts[Ttype.Y].undovr[0] = torch.tensor([underflow_ratio.item(), overflow_ratio.item()], device=x.device)
                
        # --- Weight FP8 Casting (jika module punya weight) ---
        scale_weight = 1.0
        if hasattr(self, 'weight') and self.weight is not None:
            wt_uid = f"{layer_uid}.fwd_weight"
            scale_weight = manager.compute_scale(wt_uid, FP8Config.FWD_DTYPE)
            manager.update_amax(self.weight.data, wt_uid, FP8Config.FWD_DTYPE)
            
            # Inject overflow/underflow RATIO into undovr for Pasn Tracking (Ttype.P = parameter)
            if hasattr(self, 'info_ts') and hasattr(self.info_ts[Ttype.P], 'undovr') and len(self.info_ts[Ttype.P].undovr) > 0:
                with torch.no_grad():
                    for i, param in enumerate(self.parameters()):
                        if i < len(self.info_ts[Ttype.P].undovr):
                            p_numel = max(param.data.numel(), 1)
                            p_overflow_ratio = (param.data.abs() > FP8Config.E4M3_MAX).sum().float() / p_numel
                            self.info_ts[Ttype.P].undovr[i] = torch.tensor([0.0, p_overflow_ratio.item()], device=param.device)

        # Dispatch forward to custom autograd functions
        if isinstance(self, nn.Conv2d):
            return FP8Conv2dFunction.apply(
                x, self.weight, self.bias, 
                self.stride, self.padding, self.dilation, self.groups,
                scale_input, scale_weight
            )
        elif isinstance(self, nn.Linear):
            # Flatten to 2D
            orig_shape = x.shape
            if len(orig_shape) > 2:
                x_2d = x.reshape(-1, orig_shape[-1])
            else:
                x_2d = x
            out_2d = FP8LinearFunction.apply(
                x_2d, self.weight, self.bias,
                scale_input, scale_weight
            )
            if len(orig_shape) > 2:
                return out_2d.reshape(orig_shape[:-1] + (self.out_features,))
            return out_2d
        else:
            return original_forward_fn(*args, **kwargs)


# ============================================================
# Native Precision Layer Wrappers
# ============================================================

class NativeConv2d(NativePrecisionMixin, nn.Conv2d, EModl):
    """
    Conv2d dengan native hardware precision support.
    """
    
    def __init__(self, *args, **kwargs):
        native_mode = kwargs.pop('native_mode', None)
        super(NativeConv2d, self).__init__(*args, **kwargs)
        self._layer_uid = f"NativeConv2d_{id(self)}"
        if native_mode is not None:
            self.set_native_precision(native_mode)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._native_forward_wrapper(
            self._conv_forward_native, x
        )
        return out
    
    def _conv_forward_native(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups
        )


class NativeLinear(NativePrecisionMixin, nn.Linear, EModl):
    """
    Linear layer dengan native hardware precision support.
    """
    
    def __init__(self, *args, **kwargs):
        native_mode = kwargs.pop('native_mode', None)
        super(NativeLinear, self).__init__(*args, **kwargs)
        self._layer_uid = f"NativeLinear_{id(self)}"
        if native_mode is not None:
            self.set_native_precision(native_mode)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._native_forward_wrapper(
            self._linear_forward_native, x
        )
        return out
    
    def _linear_forward_native(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


# ============================================================
# Non-compute-heavy layers
# ============================================================

class NativeBatchNorm2d(nn.BatchNorm2d, EModl):
    """BatchNorm2d — selalu beroperasi di FP32 (best practice)."""
    pass


class NativeReLU(nn.ReLU, EModl):
    """ReLU — operasi element-wise."""
    pass


class NativeMaxPool2d(nn.MaxPool2d, EModl):
    """MaxPool2d — operasi comparison."""
    pass


class NativeAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d, EModl):
    """AdaptiveAvgPool2d — averaging."""
    pass


class NativeDropout(nn.Dropout, EModl):
    """Dropout — mask operation."""
    pass
