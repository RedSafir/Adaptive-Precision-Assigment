from ext3.typing import *
from .ttype      import Ttype

import torch, re
import numpy as np

# ============================================================
# qtorch3 Import: Optional (backward compatibility)
# ============================================================
# qtorch3 adalah simulator bit-truncation. Dengan native precision,
# qtorch3 tidak lagi diperlukan. Import-nya dijaga untuk backward compat.
_QTORCH3_AVAILABLE = False
try:
    import qtorch3, qtorch3.quant  # type: ignore
    _QTORCH3_AVAILABLE = True
except ImportError:
    qtorch3 = None  # type: ignore
    pass

__all__ = [ 'Dtype', 'DtypeRndModl', 'FP32', 'BF16', 'FP16', 'INT', 'NativeDtype' ]

# ============================================================
# NativeDtype — Mapping ke PyTorch native dtypes
# ============================================================
class NativeDtype:
    """
    Representasi dtype yang langsung map ke torch.dtype native.
    
    Digunakan oleh native precision path (TF32/FP16/FP8)
    sebagai pengganti Dtype class yang bergantung pada qtorch3.
    
    Attributes:
        mode_name: "base_tf32", "base_fp16", atau "low_fp8"
        torch_dtype: torch.dtype yang sesuai
        torch_dtype_grad: torch.dtype untuk gradien (bisa berbeda untuk FP8)
    """
    
    _NATIVE_MAP = {
        'base_tf32': {
            'compute': torch.float32,
            'grad': torch.float32,
            'label': 'TF32 (FP32 storage, TF32 matmul)',
        },
        'base_fp16': {
            'compute': torch.float16,
            'grad': torch.float32,  # Gradien di FP32 untuk stabilitas
            'label': 'FP16 (native AMP)',
        },
    }
    
    def __init__(self, mode_name: str):
        self.mode_name = mode_name
        
        if mode_name in self._NATIVE_MAP:
            info = self._NATIVE_MAP[mode_name]
            self.torch_dtype = info['compute']
            self.torch_dtype_grad = info['grad']
            self.label = info['label']
        elif mode_name == 'low_fp8':
            # FP8 dtypes — check availability
            try:
                self.torch_dtype = torch.float8_e4m3fn
                self.torch_dtype_grad = torch.float8_e5m2
                self.label = 'FP8 (E4M3 fwd, E5M2 grad)'
            except AttributeError:
                # Fallback jika PyTorch < 2.1
                self.torch_dtype = torch.float16
                self.torch_dtype_grad = torch.float32
                self.label = 'FP8→FP16 fallback'
        else:
            raise ValueError(f"Unknown native mode: {mode_name}")
    
    def get_numbit(self) -> int:
        """Return jumlah bit untuk dtype ini."""
        bit_map = {
            torch.float32: 32,
            torch.float16: 16,
            torch.bfloat16: 16,
        }
        # FP8 dtypes
        try:
            bit_map[torch.float8_e4m3fn] = 8
            bit_map[torch.float8_e5m2] = 8
        except AttributeError:
            pass
        return bit_map.get(self.torch_dtype, 32)
    
    def __repr__(self) -> str:
        return f"NativeDtype({self.mode_name}: {self.label})"
    
    def __eq__(self, other) -> bool:
        if isinstance(other, NativeDtype):
            return self.mode_name == other.mode_name
        return False
    
    def __hash__(self) -> int:
        return hash(self.mode_name)


#=======#
# Dtype # <--- type of data (in a tensor) = FP32 | BF16 | ...
#=======#
class Dtype():
    #-----#
    # obj #
    #-----#
    #
    # For e \in {0,1}^exp and f \in {0,1}^man,
    #
    #   [[ (e,f) ]]_(exp, man, exp_bias)
    #   = [[ (e,f) ]]_(exp, man, 0) * 2^(-exp_bias) 
    #   = 2^(e-(2^(exp-1)-1)) * 1.f * 2^(-exp_bias).
    #

    # core.
    exp_bias : Opt[int] # None ==> dynamic exp_bias.
    
    # aux.
    maxval_lg   : float # aux. (max_val of fmt) = 2^(2^(fmt.exp-1)) * (2-2^(-fmt.man)).
    cur_exp_bias: Opt[int]   # aux. exp_bias of self that was used in rounding most recently.
    undfl_thrs: float # aux. for exp_bias=0.
    ovrfl_thrs: float # aux. for exp_bias=0.

    def __init__(self, exp: int, man: int, exp_bias: Opt[int]=0) -> None:
        # core.
        if _QTORCH3_AVAILABLE:
            self.fmt = qtorch3.FloatingPoint(exp=exp, man=man)
        else:
            # Lightweight substitute jika qtorch3 tidak tersedia
            class _FmtStub:
                def __init__(self, exp, man):
                    self.exp = exp
                    self.man = man
            self.fmt = _FmtStub(exp, man)  # type: ignore
        self.exp_bias = exp_bias
        # aux.
        self.maxval_lg = (2**(exp-1)) #+ np.log2(2-2**(-man))
        self.cur_exp_bias = exp_bias
        self.undfl_thrs = self.get_underflow_thrs(0)
        self.ovrfl_thrs = self.get_overflow_thrs(0)

    # clone Dtype obj.
    def clone(self) -> 'Dtype':
        # NOTE: cur_exp_bias is inited for the cloned obj (important if exp_bias=None).
        return Dtype(self.fmt.exp, self.fmt.man, self.exp_bias)

    # to use Dtype as a key in dict.
    def __hash__(self) -> int:
        return hash((self.fmt.exp, self.fmt.man, self.exp_bias))
        
    # ==, <.
    def __eq__(self, other) -> bool:
        if isinstance(other, Dtype):
            return (self.fmt.exp  == other.fmt.exp and
                    self.fmt.man  == other.fmt.man and
                    self.exp_bias == other.exp_bias)
        return False

    def __lt__(self, other) -> bool:
        if isinstance(other, Dtype):
            return (self.fmt.exp < other.fmt.exp and
                    self.fmt.man < other.fmt.man)
        raise ValueError

    # numbit.
    def get_numbit(self) -> int:
        return 1 + self.fmt.exp + self.fmt.man

    # ----- Native Dtype Conversion -----
    def to_native(self) -> NativeDtype:
        """
        Konversi Dtype (simulasi) ke NativeDtype (hardware).
        
        Mapping heuristik:
        - FP32 (8,23) → base_tf32
        - FP16 (5,10) → base_fp16
        - BF16 (8,7)  → base_fp16 (closest native equivalent)
        - ≤8 bit total → low_fp8
        - Lainnya → base_fp16
        """
        total_bits = self.get_numbit()
        if self == FP32:
            return NativeDtype('base_tf32')
        elif self == FP16:
            return NativeDtype('base_fp16')
        elif self == BF16:
            return NativeDtype('base_fp16')
        elif total_bits <= 9:  # 8-bit or less → FP8
            return NativeDtype('low_fp8')
        else:
            return NativeDtype('base_fp16')

    # round modl.
    def get_rndmd(self) -> 'DtypeRndModl':
        # NOTE: this func ignores self.exp_bias.
        # round to self.mt in fwd pass; nop in bwd pass (so all infs are kept).
        if _QTORCH3_AVAILABLE:
            rndmd = qtorch3.quant.QuantizerCustom(fwd_num=self.fmt)
            rndmd.dtype = self # type: ignore
            return rndmd # type: ignore
        else:
            # Tanpa qtorch3: return no-op rounding module
            rndmd = _NoOpRndModl()
            rndmd.dtype = self  # type: ignore
            return rndmd  # type: ignore
    
    def get_underflow_thrs(self, exp_bias: int) -> float:
        # NOTE: this func ignores self.exp_bias.
        # return val such that in dtype=(self, exp_bias),
        # - x in [0  , val] ==> x rounds to zero.
        # - x in [val, inf] ==> x rounds to non-zero.
        exp, man = self.fmt.exp, self.fmt.man
        return 2**(-2**(exp-1)+2 - exp_bias) * (2**(-man-1))

    def get_overflow_thrs(self, exp_bias: int) -> float:
        # NOTE: this func ignores self.exp_bias.
        # return val such that in dtype=(self, exp_bias),
        # - x in [val, inf] ==> x rounds to inf.
        # - x in [0  , val] ==> x rounds to non-inf.
        exp, man = self.fmt.exp, self.fmt.man
        if self.fmt.exp >= 8:
            exp = 8
            return 2**(2**(exp-1)-1 - exp_bias) * (2-2**(-man-1))
        else:
            return 2**(2**(exp-1)   - exp_bias) * (2-2**(-man-1))

    # Dtype -> str.
    def __repr__(self) -> str:
        # special.
        if self == FP32: return 'FP32'
        if self == BF16: return 'BF16'
        if self == FP16: return 'FP16'
        if self == INT : return 'INT'

        # general.
        if   self.exp_bias is None: exp_bias_str = f'd'
        elif self.exp_bias >= 0:    exp_bias_str = f'{self.exp_bias}'
        else:                       exp_bias_str = f'n{abs(self.exp_bias)}'

        if 0 <= self.fmt.exp <= 9 and 0 <= self.fmt.man <= 9 and len(exp_bias_str) == 1:
            return f'FP_{self.fmt.exp}{self.fmt.man}{exp_bias_str}'
        else:
            return f'FP_{self.fmt.exp}_{self.fmt.man}_{exp_bias_str}'

    #-------#
    # class #
    #-------#
    # str -> Dtype.
    @staticmethod
    def from_str(v: str) -> 'Dtype':
        # special.
        if v == 'FP32': return FP32
        if v == 'BF16': return BF16
        if v == 'FP16': return FP16
        if v == 'INT' : return INT
        # general.
        m = re.compile('FP_(\\d+)_(\\d+)_(d|(n|)(\\d+))$').match(v)
        if m is not None:
            g = m.groups()
            exp, man = int(g[0]), int(g[1])
            if g[2] == 'd': exp_bias = None
            else:           exp_bias = (-1 if g[3] == 'n' else 1) * int(g[4])
            return Dtype(exp, man, exp_bias)
        raise ValueError

    # helper for round_*.
    def _get_exp_bias(self, t: TS, reset_cur_exp_bias: bool) -> int:
        # set: exp_bias.
        exp_bias: int
        if   (reset_cur_exp_bias is True  and self.exp_bias     is not None):
            exp_bias = self.exp_bias
        elif (reset_cur_exp_bias is False and self.cur_exp_bias is not None):
            exp_bias = self.cur_exp_bias
        else:
            # reset_cur_exp_bias=True  ==> set *everytime* if dtype uses dynamic exp_bias.
            # reset_cur_exp_bias=False ==> set *only* if dtype's dynamic exp_bias is not yet set.
            t_abs_max = max( t.abs().max().item(), np.ldexp(1, -100) )
            exp_bias = int(np.floor( self.maxval_lg - np.log2(t_abs_max) ))
            
        # set: self.cur_exp_bias.
        self.cur_exp_bias = exp_bias
        return exp_bias

    # round with dtype.
    @staticmethod
    def round_dtype(t: Opt[TS], dtype: 'Dtype', allow_inf: bool, emodl, ttype, i,
                    reset_cur_exp_bias: bool=True) -> Opt[TS]:
        # nop if t = None, non-fp data, INT, or FP32.
        if (t is None) or (not t.is_floating_point()) or (dtype == INT) or (dtype == FP32):
            return t

        # ---- Native precision bypass ----
        # Jika qtorch3 tidak tersedia, bypass simulasi sepenuhnya
        # dan return tensor as-is (native precision ditangani di layer level)
        if not _QTORCH3_AVAILABLE:
            return t

        # set: t, exp_bias, dtype.cur_exp_bias.
        exp_bias = dtype._get_exp_bias(t, reset_cur_exp_bias)

        # calc: {under,over}flow ratio.
        undovr_ratio = qtorch3.quant.ratio_abs_leq_geq_thrs(t, 
                                                            thrs_l=dtype.undfl_thrs * (2**-exp_bias),
                                                            thrs_g=dtype.ovrfl_thrs * (2**-exp_bias),)
        emodl.info_ts[ttype].undovr[i] = undovr_ratio

        # round: t.
        res = qtorch3.quant.float_quantize_custom(t, exp=dtype.fmt.exp, man=dtype.fmt.man,
                                                  exp_bias_pow=2**exp_bias, allow_inf=allow_inf)
        return res

    # round with rndmd.
    @staticmethod
    def round_rndmd(t: Opt[TS], rndmd: 'DtypeRndModl', allow_inf: bool, emodl, ttype, i,
                    reset_cur_exp_bias: bool=True) -> Opt[TS]:
        # nop if t = None, non-fp data, INT, or FP32.
        if (t is None) or (not t.is_floating_point()) or (rndmd.dtype == INT) or (rndmd.dtype == FP32):
            return t

        # ---- Native precision bypass ----
        if not _QTORCH3_AVAILABLE:
            return t

        # set: t, exp_bias, rndmd.dtype.cur_exp_bias.
        exp_bias = rndmd.dtype._get_exp_bias(t, reset_cur_exp_bias)

        # calc: {under,over}flow ratio.
        undovr_ratio = qtorch3.quant.ratio_abs_leq_geq_thrs(t, 
                                                            thrs_l=rndmd.dtype.undfl_thrs * (2**-exp_bias),
                                                            thrs_g=rndmd.dtype.ovrfl_thrs * (2**-exp_bias),)
        emodl.info_ts[ttype].undovr[i] = undovr_ratio

        # round: t.
        res = rndmd(t, exp_bias_pow=2**exp_bias, allow_inf=allow_inf)
        return res

#==============#
# DtypeRndModl #
#==============#
class DtypeRndModl(torch.nn.Module):
    dtype: Dtype


class _NoOpRndModl(DtypeRndModl):
    """
    No-op rounding module — digunakan saat qtorch3 tidak tersedia.
    Mengembalikan tensor input tanpa modifikasi.
    """
    def forward(self, x, exp_bias_pow=1.0, allow_inf=True):
        return x


#=======#
# const #
#=======#
FP32 = Dtype(8, 23, 0)
BF16 = Dtype(8,  7, 0)
FP16 = Dtype(5, 10, 0)
INT  = Dtype(1,  1, 0)

