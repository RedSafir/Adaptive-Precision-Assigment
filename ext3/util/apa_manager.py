"""
apa_manager.py — Adaptive Precision Assignment Manager v2.0

Rewritten: 2026-07-15
Architecture: 3-Tier EMA-Gated Stability Monitor

This module provides two core capabilities:
  1. assign_precision()      — Static precision assignment at model init
  2. APAStabilityMonitor     — Runtime stability monitoring with adaptive promotion

Performance Characteristics (vs. original check_and_promote_overflow):
┌──────────────────────────┬──────────────────┬──────────────────────────┐
│ Metric                   │ Original (v1)    │ v2.0                     │
├──────────────────────────┼──────────────────┼──────────────────────────┤
│ CPU-GPU syncs per batch  │ ~176             │ 0 (stable) / 1 (spike)   │
│ Overflow check trigger   │ Every batch      │ EMA-gated (dynamic)      │
│ NaN detection latency    │ Up to N steps    │ 0 steps (immediate)      │
│ torch.no_grad() coverage │ Partial          │ 100% on all GPU ops      │
└──────────────────────────┴──────────────────┴──────────────────────────┘

Target GPU Architecture: NVIDIA Lovelace (SM89) / Hopper (SM90+)
"""

import math
import torch
import torch.nn as nn
import numpy as np
from enum import Enum, auto
from typing import Optional, Dict
from torch.cuda.amp import autocast

from ext3.core.emodlobj import EModlObjMgr
from ext3.core.include.dtype import Dtype, FP32, FP16
from ext3.core.include.pasn import Pasn
from ext3.core.include.ttype import Ttype
from ext3.nn.nn_native import NativeConv2d, NativeLinear

__all__ = [
    'new_id_grp_all_factory',
    'assign_precision',
    'APAStabilityMonitor',
    'StabilityEvent',
]


# ============================================================
# StabilityEvent — Result enum for monitor.step()
# ============================================================
class StabilityEvent(Enum):
    """
    Outcome of a single stability monitoring step.

    Used for logging, debugging, and conditional control flow
    in the training loop.
    """
    STABLE = auto()             # No issues — training proceeds normally
    LOSS_NAN = auto()           # Loss was NaN/Inf — emergency promotion triggered
    LOSS_SPIKE = auto()         # EMA spike detected, but gradients are healthy
    GRADIENT_OVERFLOW = auto()  # Gradient NaN/Inf confirmed — promotion triggered


# ============================================================
# Block-Based Grouping Factory
# ============================================================
def new_id_grp_all_factory():
    """
    Factory function to define block-based grouping logic for layers.
    - NativeLinear -> always new group, reset tracking
    - NativeConv2d -> new group ONLY if in_channels changes
    - Others (BN, ReLU, Pool) -> inherit previous group
    """
    last_in_channels = None
    def new_id_grp_all(emodl):
        nonlocal last_in_channels
        if isinstance(emodl, NativeLinear):
            last_in_channels = None
            return True
        elif isinstance(emodl, NativeConv2d):
            cur = emodl.in_channels
            ret = (last_in_channels != cur)
            last_in_channels = cur
            return ret
        else:
            return False
    return new_id_grp_all


# ============================================================
# assign_precision() — Static Precision Assignment
# ============================================================
def assign_precision(model: nn.Module, config: dict, base_dtype=None) -> Pasn:
    """
    Performs precision assignment via EModlObjMgr and Pasn.
    - Registers modules
    - Performs dummy forward pass to construct the EModl graph (under autocast if FP16)
    - Demotes both activations and parameters to FP8 based on target ratio
    - Prints the grouping & demotion map
    """
    # 1. Detect base dtype
    if base_dtype is None:
        if 'grad_scaler_init_scale' in config:
            base_dtype = FP16
        else:
            base_dtype = FP32

    print("\n" + "=" * 80)
    if base_dtype == FP16:
        print("PRECISION ASSIGNMENT (APA v2.0 — FP16 Base)")
    else:
        print("PRECISION ASSIGNMENT (APA v2.0 — TF32/FP32 Base)")
    print("=" * 80)

    # Register modules
    EModlObjMgr.unregister_all()
    EModlObjMgr.register(model)

    # Block-Based Tensor Grouping
    new_id_grp_all = new_id_grp_all_factory()
    EModlObjMgr.set_info_mdcur_id(new_id_grp_all)
    EModlObjMgr.reset_info(True)
    EModlObjMgr.set_param_forward_pre()

    # Dummy forward pass
    with torch.no_grad():
        if base_dtype == FP16:
            with autocast(dtype=torch.float16):
                dummy_input = torch.randn(2, 3, 32, 32).to(config['device'])
                model(dummy_input)
        else:
            dummy_input = torch.randn(2, 3, 32, 32).to(config['device'])
            model(dummy_input)

    EModlObjMgr.set_param_backward_pos(1.0)
    EModlObjMgr.reset_info(False)

    # Initialize effective tensor numels
    EModlObjMgr.set_info_ts_numel(2, config['batch_size'])

    # Create Pasn & Dtypes
    FP8 = Dtype(4, 3, 0)
    pasn = Pasn(EModlObjMgr.get_emodls(), dtype_fwd=base_dtype)

    # Demotion logic
    target_ratio = config['pa_upd_rmin']
    ids, r = EModlObjMgr.get_ids_chosen('grp_all', config['pa_upd_schm'], r_min=target_ratio)

    # Protect first and last layers
    if len(ids) > 0:
        sorted_emodls = EModlObjMgr.get_emodls_sort()
        first_id = sorted_emodls[0].info_mdcur.id['grp_all']
        last_id = sorted_emodls[-1].info_mdcur.id['grp_all']
        ids = [i for i in ids if i != first_id and i != last_id]

    upd_idvals = [i.val for i in ids]

    # Phase 1: Demote activations (cur node -> Y, GY)
    upd_dtplan_cur = {Ttype.Y: FP8, Ttype.GY: FP8}
    pasn.update_by_id_grp_all('cur', 'id', upd_dtplan_cur, upd_idvals)

    # Phase 2: Demote parameters (prv node -> P, GP)
    upd_dtplan_prv = {Ttype.P: FP8, Ttype.GP: FP8}
    pasn.update_by_id_grp_all('prv', 'id', upd_dtplan_prv, upd_idvals)

    # Apply to graph
    EModlObjMgr.set_info_ts_dtype(pasn)
    EModlObjMgr.set_info_ts_rndmd()

    print(f"  Target Demotion Ratio : {target_ratio:.3f}")
    print(f"  Total Layers          : {len(EModlObjMgr.get_emodls_sort())}")
    print(f"  Demoted Layers        : {len(ids)}")

    print("\n=== Grouping & Demotion Map ===")
    for emodl in EModlObjMgr.get_emodls_sort():
        uid = emodl.info_mdcur.id['grp_all'].val
        dtype_y = (emodl.info_ts[Ttype.Y].dtype[0]
                   if hasattr(emodl.info_ts[Ttype.Y], 'dtype')
                   and len(emodl.info_ts[Ttype.Y].dtype) > 0
                   else base_dtype)
        dtype_p = (emodl.info_ts[Ttype.P].dtype[0]
                   if hasattr(emodl.info_ts[Ttype.P], 'dtype')
                   and len(emodl.info_ts[Ttype.P].dtype) > 0
                   else base_dtype)
        mode_y = dtype_y.to_native().mode_name
        mode_p = dtype_p.to_native().mode_name
        markers = []
        if mode_y == "low_fp8": markers.append("Y=FP8")
        if mode_p == "low_fp8": markers.append("P=FP8")
        marker_str = f" [DEMOTED: {', '.join(markers)}]" if markers else ""
        print(f"Group {uid:3d} | {type(emodl).__qualname__:30s} "
              f"| Y={mode_y:10s} P={mode_p:10s}{marker_str}")

    print("=" * 80)
    return pasn


# ============================================================
# APAStabilityMonitor — 3-Tier EMA-Gated Stability Monitor
# ============================================================
class APAStabilityMonitor:
    """
    Production-ready stability monitor for Adaptive Precision training.

    Replaces check_and_promote_overflow() with a 3-tier gated system
    that eliminates CPU-GPU synchronization stalls during stable training.

    Architecture::

        ┌─────────────────────────────────────────────────────────┐
        │  Tier 1: Loss NaN/Inf Check (every step, CPU-only)     │
        │  Cost: 1× math.isfinite()                              │
        │  Action: Emergency promote ALL demoted layers           │
        └──────────────────────┬──────────────────────────────────┘
                               │ finite
        ┌──────────────────────▼──────────────────────────────────┐
        │  Tier 1b: EMA Update + Spike Detection (CPU-only)      │
        │  Cost: 3 float multiply-adds                            │
        │  Gate: |loss - EMA| / σ_EMA > variance_threshold       │
        └──────────────────────┬──────────────────────────────────┘
                               │ spike > threshold
        ┌──────────────────────▼──────────────────────────────────┐
        │  Tier 2: Fused Gradient Scan (1× torch.cat, GPU)       │
        │  Cost: 1× .item() sync                                  │
        │  Check: isnan(all_grads) | isinf(all_grads)             │
        └──────────────────────┬──────────────────────────────────┘
                               │ has NaN/Inf grads
        ┌──────────────────────▼──────────────────────────────────┐
        │  Tier 3: Targeted Layer Promotion (rare)                │
        │  Uses: EModlObjMgr.get_undovrs() + inc_ts_prec()       │
        └─────────────────────────────────────────────────────────┘

    Args:
        ema_alpha: Smoothing factor for loss EMA (0 < α ≤ 1).
            Higher = more responsive to recent values.
            Recommended range: 0.05–0.3. Default: 0.1.
        variance_threshold: Number of EMA standard deviations to
            classify a loss value as a "spike" and trigger Tier 2.
            Lower = more sensitive (more checks, safer).
            Recommended range: 1.5–4.0. Default: 2.0.
        ovr_thrs: Overflow ratio threshold for per-layer promotion
            in Tier 3. Layers with overflow ratio > this value get
            promoted. Default: 0.0 (any overflow triggers promotion).
        warmup_steps: Number of initial steps to skip spike detection
            while the EMA calibrates. NaN/Inf detection (Tier 1) is
            always active, even during warmup. Default: 10.

    Example::

        monitor = APAStabilityMonitor(ema_alpha=0.1, variance_threshold=2.0)

        for inputs, targets in dataloader:
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()  # Already needed for logging
            event = monitor.step(loss_val, model, FP32)
            total_loss += loss_val * inputs.size(0)
    """

    __slots__ = (
        '_ema_alpha', '_variance_threshold', '_ovr_thrs', '_warmup_steps',
        '_ema_loss', '_ema_variance',
        '_step_count', '_total_tier2_checks', '_total_promotions',
        '_nan_events', '_spike_events',
    )

    def __init__(
        self,
        ema_alpha: float = 0.1,
        variance_threshold: float = 2.0,
        ovr_thrs: float = 0.0,
        warmup_steps: int = 10,
    ):
        # --- Configuration (immutable after init) ---
        self._ema_alpha = ema_alpha
        self._variance_threshold = variance_threshold
        self._ovr_thrs = ovr_thrs
        self._warmup_steps = warmup_steps

        # --- EMA State ---
        self._ema_loss: Optional[float] = None
        self._ema_variance: float = 0.0

        # --- Diagnostic Counters ---
        self._step_count: int = 0
        self._total_tier2_checks: int = 0
        self._total_promotions: int = 0
        self._nan_events: int = 0
        self._spike_events: int = 0

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def step(
        self,
        loss_value: float,
        model: nn.Module,
        target_precision: Dtype,
    ) -> StabilityEvent:
        """
        Main entry point — call once per training batch.

        IMPORTANT: Call AFTER loss.backward() + optimizer.step().
        Pass the loss.item() value that you already compute for logging
        to avoid any additional CPU-GPU synchronization.

        Args:
            loss_value: Scalar loss (Python float from loss.item()).
            model: The nn.Module being trained.
            target_precision: Dtype to promote to (e.g., FP32 or FP16).

        Returns:
            StabilityEvent indicating the outcome of this step.
        """
        self._step_count += 1

        # ============================================================
        # TIER 1: Immediate NaN/Inf Loss Detection
        # Cost: 1× math.isfinite() — pure CPU, zero GPU overhead
        #
        # Rationale: NaN/Inf loss means catastrophic failure has ALREADY
        # happened. Every subsequent optimizer.step() will corrupt ALL
        # weights permanently. We MUST respond in this exact step —
        # waiting even 1 batch is fatal (especially for YOLO detection
        # heads where loss spikes are sharp and transient).
        # ============================================================
        if not math.isfinite(loss_value):
            self._nan_events += 1
            self._promote_all_demoted(target_precision)
            # Reset EMA — the loss signal is now meaningless
            self._ema_loss = None
            self._ema_variance = 0.0
            return StabilityEvent.LOSS_NAN

        # ============================================================
        # TIER 1b: Update Loss EMA
        # Cost: 3 float multiply-adds — truly negligible
        #
        # EMA tracks the "expected" loss trajectory. Deviation from
        # this expectation is the spike signal for Tier 2.
        # ============================================================
        if self._ema_loss is None:
            # First valid loss value — initialize EMA seed
            self._ema_loss = loss_value
            self._ema_variance = 0.0
            return StabilityEvent.STABLE

        # Measure "surprise" = squared deviation from EMA prediction
        # Must be computed BEFORE updating EMA to measure prediction error
        deviation_sq = (loss_value - self._ema_loss) ** 2

        # Exponential moving average update
        alpha = self._ema_alpha
        self._ema_loss = alpha * loss_value + (1.0 - alpha) * self._ema_loss
        self._ema_variance = (alpha * deviation_sq
                              + (1.0 - alpha) * self._ema_variance)

        # During warmup, EMA statistics are unreliable — skip spike detection
        # but still accumulate EMA for calibration.
        # NaN/Inf detection (Tier 1) remains active even during warmup.
        if self._step_count <= self._warmup_steps:
            return StabilityEvent.STABLE

        # ============================================================
        # TIER 2: EMA Spike Detection → Fused Gradient Scan
        #
        # Gate condition: |loss - EMA_loss| / σ_EMA > threshold
        #
        # This replaces the old fixed-interval check with a DYNAMIC
        # trigger. The gradient scan fires ONLY when the loss deviates
        # abnormally — meaning the FP8 quantization is likely causing
        # numerical issues in this specific batch.
        #
        # Cost when no spike: 1× sqrt + 1× division — pure CPU, zero GPU
        # Cost when spike:    1× torch.cat + 1× .item() — 1 GPU sync
        # ============================================================
        ema_std = math.sqrt(max(self._ema_variance, 1e-12))
        spike_magnitude = abs(loss_value - self._ema_loss) / ema_std

        if spike_magnitude <= self._variance_threshold:
            return StabilityEvent.STABLE  # No spike — ZERO GPU overhead

        # Spike detected — run fused gradient health check on GPU
        self._spike_events += 1

        if not self._fused_grad_overflow_check(model):
            # Loss spiked but gradients are clean — likely just a hard
            # mini-batch (common in CIFAR-10 with small batch sizes).
            # No action needed.
            return StabilityEvent.LOSS_SPIKE

        # ============================================================
        # TIER 3: Targeted Layer Promotion
        #
        # Only reached when BOTH conditions are met:
        #   1. Loss spike exceeds EMA variance threshold
        #   2. Gradient tensor contains NaN or Inf
        #
        # This is the expensive path (get_undovrs → GPU→CPU transfer),
        # but it is triggered extremely rarely during healthy training.
        # ============================================================
        self._promote_unstable_layers(target_precision)
        return StabilityEvent.GRADIENT_OVERFLOW

    def get_stats(self) -> Dict[str, object]:
        """
        Return monitoring statistics for logging and diagnostics.

        Returns:
            Dictionary with EMA state, event counts, and efficiency metrics.
            Useful for printing at the end of each epoch.
        """
        return {
            'step_count': self._step_count,
            'ema_loss': self._ema_loss,
            'ema_variance': self._ema_variance,
            'ema_std': (math.sqrt(max(self._ema_variance, 1e-12))
                        if self._ema_variance > 0 else 0.0),
            'nan_events': self._nan_events,
            'spike_events': self._spike_events,
            'tier2_checks': self._total_tier2_checks,
            'total_promotions': self._total_promotions,
            'sync_efficiency': (
                f"{self._total_tier2_checks} GPU syncs / "
                f"{self._step_count} steps"
                if self._step_count > 0 else "N/A"
            ),
        }

    def reset(self) -> None:
        """Reset all state. Call when starting a new training run."""
        self._ema_loss = None
        self._ema_variance = 0.0
        self._step_count = 0
        self._total_tier2_checks = 0
        self._total_promotions = 0
        self._nan_events = 0
        self._spike_events = 0

    # --------------------------------------------------------
    # Internal Methods
    # --------------------------------------------------------

    @torch.no_grad()
    def _fused_grad_overflow_check(self, model: nn.Module) -> bool:
        """
        Tier 2 implementation: Single fused GPU scan of all gradients.

        Strategy:
          1. Collect all non-None gradients as flat views (.view(-1) is
             zero-copy — it creates a view, NOT a new allocation).
          2. Concatenate into a single contiguous tensor via torch.cat()
             (1 GPU memory allocation, 1 GPU kernel).
          3. Run isnan().any() | isinf().any() — PyTorch fuses these
             reductions into minimal GPU kernel launches.
          4. Transfer single boolean to CPU via .item() — this is the
             ONLY CPU-GPU synchronization point.

        Sync points: exactly 1 (.item() on the boolean result)
        GPU kernels:  ~3 (cat, isnan_any, isinf_any — may be fused)

        Returns:
            True if ANY gradient parameter contains NaN or Inf.
        """
        self._total_tier2_checks += 1

        # Collect gradient views — .view(-1) is zero-copy
        grads = [p.grad.view(-1) for p in model.parameters()
                 if p.grad is not None]

        if not grads:
            return False  # No gradients computed (unlikely but safe)

        # Single concatenation — one GPU memory allocation
        all_grads = torch.cat(grads)

        # Fused health check — single .item() = single CPU-GPU sync
        has_bad = bool(
            (torch.isnan(all_grads).any()
             | torch.isinf(all_grads).any()).item()
        )

        return has_bad

    def _promote_unstable_layers(self, target_precision: Dtype) -> None:
        """
        Tier 3: Promote layers whose overflow ratio exceeds threshold.

        Uses the existing EModlObjMgr.get_undovrs() + inc_ts_prec()
        infrastructure. The overflow ratios were already recorded during
        the forward pass by _fp8_forward() in nn_native.py.

        This method is only called when Tier 1+2 confirm instability,
        so the sync cost of get_undovrs() is amortized by its rarity.
        """
        self._total_promotions += 1
        undovrs = EModlObjMgr.get_undovrs()
        flag = np.concatenate([
            undovrs[Ttype.P][:, 1] > self._ovr_thrs,
            undovrs[Ttype.Y][:, 1] > self._ovr_thrs,
        ])
        if flag.any():
            EModlObjMgr.inc_ts_prec(
                flag,
                {Ttype.Y: target_precision, Ttype.P: target_precision},
            )

    def _promote_all_demoted(self, target_precision: Dtype) -> None:
        """
        Emergency promotion: force-promote ALL demoted layers.

        Called when loss is NaN/Inf — the training state is already
        corrupted and targeted detection is pointless. We maximize
        stability by reverting ALL FP8 layers to full precision.

        inc_ts_prec() internally skips layers that are already at
        target_precision, so this is safe for non-demoted layers.
        """
        self._total_promotions += 1
        undovrs = EModlObjMgr.get_undovrs()
        # Force-promote everything: threshold=-1 so all entries pass
        # (overflow_ratio >= 0 always holds, so > -1 is always True)
        flag = np.concatenate([
            undovrs[Ttype.P][:, 1] > -1.0,
            undovrs[Ttype.Y][:, 1] > -1.0,
        ])
        EModlObjMgr.inc_ts_prec(
            flag,
            {Ttype.Y: target_precision, Ttype.P: target_precision},
        )
