"""
pallet_packer._brkga_core.dispatch — Python wrappers + JIT warmup.

  warmup_jit             pre-compile every JIT decoder (~5s, idempotent)
  decode_chromosome      dispatch a chromosome to the right decoder mode
                         (0-5 geometric, 0-4 cstr; mode 5 cstr → mode 4 cstr)
  _selector_to_mode      map a chromosome's last key to a mode index
                         (uniform or adaptive-CDF)
  decode_auto_mode       picks mode from chromosome's selector key,
                         then forwards to decode_chromosome

The dispatcher is the API contract the JIT decoders implement. The
short-circuit `cstr_active = has_constraints and ... and rfs is not None`
keeps the geometric path (BR / academic) untouched — verified
bit-identical at n=12 BR regression.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..models import Box, Pallet, PackerConfig, Placement
from ..packer import PackResult, PalletState
from ..brkga_v3_fast import precompute_box_dims
from .precompute import _NO_LIMIT
from .blocks import resolve_sku_blocks_from_chrom
# Prefer the Cython-compiled geometric decoders when the .so is present
# (built by `pip install -e .` inside the project Docker image). Falls back
# to the Numba reference for environments without a build toolchain — same
# signatures, bit-identical results. Phase 3a-d completes the geometric
# port: all four decoder entry points now have a Cython implementation.
#   Phase 3a — decode_njit_mode (modes 0/1/2)
#   Phase 3b — decode_layer_njit (mode 3)
#   Phase 3c — decode_blocks_njit_mode (mode 4)
#   Phase 3d — decode_precomputed_blocks_njit_mode (mode 5)
try:
    from .jit_decoders_geom_cy import (  # noqa: F401
        decode_njit_mode,
        decode_layer_njit,
        decode_blocks_njit_mode,
        decode_precomputed_blocks_njit_mode,
    )
except ImportError:
    from .jit_decoders_geom import (  # noqa: F401
        decode_njit_mode,
        decode_layer_njit,
        decode_blocks_njit_mode,
        decode_precomputed_blocks_njit_mode,
    )
# Phase 4 — constraint-aware decoders. All three cstr decoders are now
# Cython-LIVE; the Numba reference path remains the import fallback for
# environments without a build toolchain.
#   Phase 4b — decode_njit_mode_cstr (cstr modes 0/1/2)
#   Phase 4c — decode_blocks_njit_mode_cstr (cstr mode 4)
#   Phase 4d — decode_layer_njit_cstr (cstr mode 3)
try:
    from .jit_decoders_cstr_cy import (  # noqa: F401
        decode_njit_mode_cstr,
        decode_blocks_njit_mode_cstr,
        decode_layer_njit_cstr,
    )
except ImportError:
    from .jit_decoders_cstr import (  # noqa: F401
        decode_njit_mode_cstr,
        decode_blocks_njit_mode_cstr,
        decode_layer_njit_cstr,
    )


# Module-level flag — set to True after the first warmup_jit() call so
# subsequent calls return immediately. (Was previously a free variable
# living in brkga_v3_5.py's global scope; now scoped to dispatch module.)
_JIT_WARMED = False


def warmup_jit() -> None:
    """Pre-compile all JIT decoder modes. Saves ~1-2s on first calls.

    Without this, each new mode triggers compilation on its first invocation
    inside the BRKGA loop, causing ~400-700ms hiccups that bias eval timing.
    """
    global _JIT_WARMED
    if _JIT_WARMED:
        return
    n = 2
    bps = np.zeros(n, dtype=np.int64)
    bps[0] = 0; bps[1] = 1
    n_rots = np.array([1, 1], dtype=np.int64)
    dims = np.zeros((n, 6, 3), dtype=np.int64)
    dims[:, 0, 0] = 10
    dims[:, 0, 1] = 10
    dims[:, 0, 2] = 10
    placements = np.zeros((n, 6), dtype=np.int64)
    sku_ids = np.zeros(n, dtype=np.int64)  # both boxes same SKU
    for mode in (0, 1, 2):
        _ = decode_njit_mode(bps, n_rots, dims, 100, 100, 100, 1, placements, mode)
    _ = decode_layer_njit(bps, n_rots, dims, 100, 100, 100, 1, placements)
    _ = decode_blocks_njit_mode(bps, n_rots, dims, sku_ids, 100, 100, 100, 1, placements, 1)
    sku_best = np.array([[2, 1, 1, 0]], dtype=np.int64)
    _ = decode_precomputed_blocks_njit_mode(
        bps, n_rots, dims, sku_ids, sku_best, 100, 100, 100, 1, placements, 1)
    # Constraint-aware variant (v3.10/v3.11/v3.12)
    weights = np.zeros(n, dtype=np.float64)
    mlot = np.full(n, 1e18, dtype=np.float64)
    rfs_arr = np.zeros(n, dtype=np.int64)
    placements_cstr = np.zeros((n, 6), dtype=np.int64)
    # cog scalars: any large range with cog_active=0 → check skipped
    _CMIN, _CMAX = -1e18, 1e18
    for mode in (0, 1, 2):
        _ = decode_njit_mode_cstr(
            bps, n_rots, dims, 100, 100, 100, 1, placements_cstr, mode,
            weights, mlot, rfs_arr, 1e18, 0.0,
            0, _CMIN, _CMAX, _CMIN, _CMAX, 0.0, 0)
    _ = decode_blocks_njit_mode_cstr(
        bps, n_rots, dims, sku_ids, 100, 100, 100, 1, placements_cstr, 1,
        weights, mlot, rfs_arr, 1e18, 0.0,
        0, _CMIN, _CMAX, _CMIN, _CMAX, 0.0, 0)
    _ = decode_layer_njit_cstr(
        bps, n_rots, dims, 100, 100, 100, 1, placements_cstr,
        weights, mlot, rfs_arr, 1e18, 0.0,
        0, _CMIN, _CMAX, _CMIN, _CMAX, 0.0, 0)
    _JIT_WARMED = True


def decode_chromosome(
    chrom: np.ndarray,
    boxes: List[Box],
    pallet: Pallet,
    config: PackerConfig,
    n_rots_arr: np.ndarray,
    dims_all: np.ndarray,
    mode: int,
    max_pallets: int = 1,
    sku_id_per_box: Optional[np.ndarray] = None,
    sku_best_block: Optional[np.ndarray] = None,
    sku_top_k_blocks: Optional[np.ndarray] = None,
    # v3.10/v3.12 constraint-aware path. When weights is None, the
    # geometric-only decoder runs (BR / academic). When weights + mlot +
    # rfs + pmw are provided and any constraint is finite, modes 0/1/2 use
    # decode_njit_mode_cstr; modes 3/4/5 use their own cstr variants.
    weights: Optional[np.ndarray] = None,
    mlot: Optional[np.ndarray] = None,
    rfs: Optional[np.ndarray] = None,
    pallet_max_weight: float = _NO_LIMIT,
    has_constraints: bool = False,
    support_ratio: float = 0.0,
    # v3.12 additions:
    require_centroid: int = 0,
    cog_x_min: float = -1e18, cog_x_max: float = 1e18,
    cog_y_min: float = -1e18, cog_y_max: float = 1e18,
    cog_min_load_frac: float = 0.0,
    cog_active: int = 0,
    max_overhang: float = 0.0,
) -> PackResult:
    """Decode chromosome with chosen mode.

    Modes:
      0 = DFTRC
      1 = wall-build
      2 = corner-fill
      3 = layer-build
      4 = DFTRC + dynamic composite blocks (needs sku_id_per_box)
      5 = DFTRC + pre-computed top-K blocks (chromosome picks one of K
          candidates per SKU via VBO keys; needs sku_id_per_box and
          either sku_top_k_blocks or sku_best_block)

    Constraint behavior (v3.10):
      When has_constraints=True, modes 0/1/2 enforce Pallet.max_weight +
      Box.max_load_on_top. Modes 3/4/5 fall back to constraint-aware mode 0
      because their block/layer variants aren't constraint-aware yet.
    """
    n = len(boxes)
    if n == 0:
        return PackResult(pallets=[], unpacked=[])
    bps = chrom[:n]
    order = np.argsort(bps).astype(np.int64)

    L = int(round(pallet.length))
    W = int(round(pallet.width))
    H = int(round(pallet.height))
    # Effective container dims include positive-side overhang (matches v2:
    # boxes can overhang the +x / +y pallet edges but not the -x / -y).
    L_eff = L + int(round(max_overhang)) if max_overhang > 0 else L
    W_eff = W + int(round(max_overhang)) if max_overhang > 0 else W

    placements_out = np.zeros((n, 6), dtype=np.int64)
    # Constraint-aware dispatch (v3.11/v3.12): every mode has a constraint-
    # aware variant when has_constraints=True. Mode 5 delegates to mode 4
    # cstr because the precomputed top-K block selection has no value under
    # constraints (top-K was pre-enumerated assuming no weight/load limits).
    cstr_active = (has_constraints and weights is not None
                   and mlot is not None and rfs is not None)
    if mode == 3:
        if cstr_active:
            n_bins = decode_layer_njit_cstr(
                order, n_rots_arr, dims_all, L_eff, W_eff, H,
                max_pallets, placements_out,
                weights, mlot, rfs, pallet_max_weight, support_ratio,
                require_centroid,
                cog_x_min, cog_x_max, cog_y_min, cog_y_max,
                cog_min_load_frac, cog_active,
            )
        else:
            n_bins = decode_layer_njit(
                order, n_rots_arr, dims_all, L_eff, W_eff, H,
                max_pallets, placements_out,
            )
    elif mode == 4:
        if sku_id_per_box is None:
            # Without SKU info, fall back to mode 0 (or its cstr variant).
            if cstr_active:
                n_bins = decode_njit_mode_cstr(
                    order, n_rots_arr, dims_all, L_eff, W_eff, H,
                    max_pallets, placements_out, 0,
                    weights, mlot, rfs, pallet_max_weight, support_ratio,
                    require_centroid,
                    cog_x_min, cog_x_max, cog_y_min, cog_y_max,
                    cog_min_load_frac, cog_active,
                )
            else:
                n_bins = decode_njit_mode(
                    order, n_rots_arr, dims_all, L_eff, W_eff, H,
                    max_pallets, placements_out, 0,
                )
        else:
            n_skus = int(sku_id_per_box.max()) + 1
            if cstr_active:
                n_bins = decode_blocks_njit_mode_cstr(
                    order, n_rots_arr, dims_all, sku_id_per_box,
                    L_eff, W_eff, H, max_pallets, placements_out, n_skus,
                    weights, mlot, rfs, pallet_max_weight, support_ratio,
                    require_centroid,
                    cog_x_min, cog_x_max, cog_y_min, cog_y_max,
                    cog_min_load_frac, cog_active,
                )
            else:
                n_bins = decode_blocks_njit_mode(
                    order, n_rots_arr, dims_all, sku_id_per_box,
                    L_eff, W_eff, H, max_pallets, placements_out, n_skus,
                )
    elif mode == 5:
        if sku_id_per_box is None:
            # Without SKU info, fall back to mode 0 (or its cstr variant).
            if cstr_active:
                n_bins = decode_njit_mode_cstr(
                    order, n_rots_arr, dims_all, L_eff, W_eff, H,
                    max_pallets, placements_out, 0,
                    weights, mlot, rfs, pallet_max_weight, support_ratio,
                    require_centroid,
                    cog_x_min, cog_x_max, cog_y_min, cog_y_max,
                    cog_min_load_frac, cog_active,
                )
            else:
                n_bins = decode_njit_mode(
                    order, n_rots_arr, dims_all, L_eff, W_eff, H,
                    max_pallets, placements_out, 0,
                )
        elif cstr_active:
            # Under constraints, delegate to mode 4 cstr — the precomputed
            # top-K block selection has no meaningful interaction with
            # weight/load constraints (top-K was pre-enumerated for the
            # unconstrained packing). Dynamic block extension is equivalent
            # or better.
            n_skus = int(sku_id_per_box.max()) + 1
            n_bins = decode_blocks_njit_mode_cstr(
                order, n_rots_arr, dims_all, sku_id_per_box,
                L_eff, W_eff, H, max_pallets, placements_out, n_skus,
                weights, mlot, rfs, pallet_max_weight, support_ratio,
                require_centroid,
                cog_x_min, cog_x_max, cog_y_min, cog_y_max,
                cog_min_load_frac, cog_active,
            )
        else:
            n_skus = int(sku_id_per_box.max()) + 1
            # Geometric-only path (no constraints): no overhang either,
            # so use L, W directly.
            if sku_top_k_blocks is not None:
                chosen = resolve_sku_blocks_from_chrom(
                    chrom, n, n_skus, sku_top_k_blocks)
                n_bins = decode_precomputed_blocks_njit_mode(
                    order, n_rots_arr, dims_all, sku_id_per_box, chosen,
                    L, W, H, max_pallets, placements_out, n_skus,
                )
            elif sku_best_block is not None:
                n_bins = decode_precomputed_blocks_njit_mode(
                    order, n_rots_arr, dims_all, sku_id_per_box,
                    sku_best_block,
                    L, W, H, max_pallets, placements_out, n_skus,
                )
            else:
                n_bins = decode_blocks_njit_mode(
                    order, n_rots_arr, dims_all, sku_id_per_box,
                    L, W, H, max_pallets, placements_out, n_skus,
                )
    else:
        # Modes 0/1/2: geometric or constraint-aware.
        if cstr_active:
            n_bins = decode_njit_mode_cstr(
                order, n_rots_arr, dims_all, L_eff, W_eff, H,
                max_pallets, placements_out, mode,
                weights, mlot, rfs, pallet_max_weight, support_ratio,
                require_centroid,
                cog_x_min, cog_x_max, cog_y_min, cog_y_max,
                cog_min_load_frac, cog_active,
            )
        else:
            n_bins = decode_njit_mode(
                order, n_rots_arr, dims_all, L_eff, W_eff, H,
                max_pallets, placements_out, mode,
            )

    pallets_state: List[PalletState] = []
    for b in range(int(n_bins)):
        pallets_state.append(PalletState(pallet, f"P{b + 1:03d}", config))
    unpacked: List[Box] = []
    for i in range(n):
        box_idx = int(order[i])
        box = boxes[box_idx]
        if placements_out[i, 5] == 0:
            unpacked.append(box)
            continue
        b = int(placements_out[i, 0])
        r = int(placements_out[i, 1])
        x = float(placements_out[i, 2])
        y = float(placements_out[i, 3])
        z = float(placements_out[i, 4])
        rotation = box.allowed_rotations[r]
        pl = Placement(box=box, rotation=rotation, x=x, y=y, z=z)
        pallets_state[b].placements.append(pl)
        pallets_state[b].total_weight += box.weight
    return PackResult(pallets=pallets_state, unpacked=unpacked)


def _selector_to_mode(selector: float, n_modes: int,
                      mode_cdf: Optional[np.ndarray] = None) -> int:
    """Map selector key in [0,1) to mode index.

    Uniform when mode_cdf is None. With mode_cdf (cumulative weights of
    length n_modes summing to 1.0), inverse-CDF lookup biases the mapping
    toward modes with higher weight. This is the adaptive selector.
    """
    if mode_cdf is None:
        return min(n_modes - 1, int(selector * n_modes))
    # mode_cdf[m] = prob(mode <= m); find smallest m with selector < cdf[m]
    m = int(np.searchsorted(mode_cdf, selector, side='right'))
    return min(m, n_modes - 1)


def decode_auto_mode(
    chrom: np.ndarray,
    boxes: List[Box],
    pallet: Pallet,
    config: PackerConfig,
    n_rots_arr: np.ndarray,
    dims_all: np.ndarray,
    max_pallets: int = 1,
    n_modes: int = 5,
    sku_id_per_box: Optional[np.ndarray] = None,
    sku_best_block: Optional[np.ndarray] = None,
    sku_top_k_blocks: Optional[np.ndarray] = None,
    mode_cdf: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    mlot: Optional[np.ndarray] = None,
    rfs: Optional[np.ndarray] = None,
    pallet_max_weight: float = _NO_LIMIT,
    has_constraints: bool = False,
    support_ratio: float = 0.0,
    require_centroid: int = 0,
    cog_x_min: float = -1e18, cog_x_max: float = 1e18,
    cog_y_min: float = -1e18, cog_y_max: float = 1e18,
    cog_min_load_frac: float = 0.0,
    cog_active: int = 0,
    max_overhang: float = 0.0,
) -> PackResult:
    """Use last key in chromosome to pick decoder mode (0..n_modes-1).
    Mode 5 (if n_modes >= 6) uses pre-computed top-K blocks per SKU.

    If mode_cdf is provided, it remaps the uniform selector key via
    inverse-CDF so modes with higher weight get more chromosome share —
    the adaptive mode selector (v3.9).

    If has_constraints=True, the constraint-aware decode path runs (v3.10
    + v3.11 + v3.12). support_ratio > 0 enforces partial-support fraction;
    require_centroid + per-box rfs add the v3.12 stability checks; the
    cog_* params add the v3.12 CoG envelope; max_overhang enlarges the
    effective pallet by allowing positive-edge overhang.
    """
    n = len(boxes)
    if len(chrom) >= 2 * n + 1:
        selector = float(chrom[-1])
        mode = _selector_to_mode(selector, n_modes, mode_cdf)
    else:
        mode = 0
    return decode_chromosome(chrom, boxes, pallet, config,
                             n_rots_arr, dims_all, mode, max_pallets,
                             sku_id_per_box=sku_id_per_box,
                             sku_best_block=sku_best_block,
                             sku_top_k_blocks=sku_top_k_blocks,
                             weights=weights, mlot=mlot,
                             pallet_max_weight=pallet_max_weight,
                             has_constraints=has_constraints,
                             support_ratio=support_ratio,
                             rfs=rfs,
                             require_centroid=require_centroid,
                             cog_x_min=cog_x_min, cog_x_max=cog_x_max,
                             cog_y_min=cog_y_min, cog_y_max=cog_y_max,
                             cog_min_load_frac=cog_min_load_frac,
                             cog_active=cog_active,
                             max_overhang=max_overhang)


# ============================================================================
# Adaptive mode selector (v3.9): probe each mode on a small seed set,
# bias the selector keyspace toward better-performing modes.
# ============================================================================
