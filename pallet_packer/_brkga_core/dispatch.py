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
    # Hardening round 2: raw deck dims for the floor deck-contact check
    # (F17) — passed only when overhang inflates L/W, else the 0 sentinel
    # keeps the decoders' legacy unconditional-floor-support path — and the
    # transitive load-commit flag (F19).
    pallet_l = L if max_overhang > 0 else 0
    pallet_w = W if max_overhang > 0 else 0
    transitive = 1 if getattr(config, "transitive_load_bearing", False) else 0

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
                pallet_l, pallet_w, transitive,
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
                    pallet_l, pallet_w, transitive,
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
                    pallet_l, pallet_w, transitive,
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
                    pallet_l, pallet_w, transitive,
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
                pallet_l, pallet_w, transitive,
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
                pallet_l, pallet_w, transitive,
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


# ============================================================================
# Phase 5c: batch population fitness — parallel decode + vectorised fitness.
#
# Replaces the per-chromosome decode loop inside the BRKGA evolution step.
# Each generation evaluates pop_size individuals; with the per-chromosome
# decode call, this is the dominant runtime. The batch path:
#   1. Groups chromosomes by their auto-selected mode (or single mode).
#   2. Calls the corresponding decode_batch_* entry (one per mode) which
#      dispatches the inner nogil cdef loops across cores via prange.
#   3. Computes fitness from the raw placements_out arrays via vectorised
#      numpy ops — no PackResult construction per chromosome.
#
# PackResult conversion is deferred to the new-best path in the driver
# (see driver.py); for individuals that don't improve on the current best,
# we never pay for object allocation or placement-list construction.
# ============================================================================

# Re-import the batch entry points (try Cython first, fall back is a no-op
# because there's no Numba-equivalent — batch decode is Cython-only).
try:
    from .jit_decoders_geom_cy import (  # noqa: F401
        decode_batch_njit_mode,
        decode_batch_layer_njit,
        decode_batch_blocks_njit_mode,
        decode_batch_precomputed_blocks_njit_mode,
        decode_batch_precomputed_blocks_per_chrom,
    )
    from .jit_decoders_cstr_cy import (  # noqa: F401
        decode_batch_njit_mode_cstr,
        decode_batch_blocks_njit_mode_cstr,
        decode_batch_layer_njit_cstr,
    )
    _BATCH_AVAILABLE = True
except ImportError:
    _BATCH_AVAILABLE = False


def _compute_fitness_pallet1_batch(
    placements_out_all: np.ndarray,   # (pop_size, n, 6)
    orders: np.ndarray,                # (pop_size, n)
    dims_all: np.ndarray,              # (n_boxes, n_rots_max, 3)
    pallet: Pallet,
    realism=None,
) -> np.ndarray:
    """Vectorised _fitness_pallet1 over a batch of decode results.

    fits[i] = 1 - vol_on_pallet0 / pallet_capacity  (lower is better),
    plus the epsilon-scaled realism term when a RealismContext is given
    (realism=None keeps the historical value bit-identical).
    """
    cap = float(pallet.length * pallet.width * pallet.height)
    if cap <= 0.0:
        return np.ones(placements_out_all.shape[0], dtype=np.float64)
    placed = placements_out_all[:, :, 5] == 1
    on_pallet_0 = placements_out_all[:, :, 0] == 0
    mask = placed & on_pallet_0
    rot_idx = placements_out_all[:, :, 1]
    # Fancy-index dims_all: (pop_size, n, 3)
    selected_dims = dims_all[orders, rot_idx]
    vols = selected_dims[:, :, 0] * selected_dims[:, :, 1] * selected_dims[:, :, 2]
    used = (vols * mask).sum(axis=1).astype(np.float64)
    fits = 1.0 - used / cap
    if realism is not None:
        from .realism import realism_batch
        fits = fits + realism.eps * realism_batch(
            placements_out_all, orders, realism)
    return fits


def decode_population_fitness(
    population: np.ndarray,            # (pop_size, chrom_size)
    boxes: List[Box],
    pallet: Pallet,
    config: PackerConfig,
    n_rots_arr: np.ndarray,
    dims_all: np.ndarray,
    max_pallets: int = 1,
    *,
    use_multi_decoder: bool = False,
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
    realism=None,
) -> np.ndarray:
    """Compute the BRKGA fitness array for an entire population in parallel.

    Returns: fits — np.ndarray shape (pop_size,) of float64.

    Math is bit-identical to:
        for i in range(pop_size):
            res = decode_chromosome(population[i], ...)
            fits[i] = _fitness_pallet1(res, pallet, realism=realism)
    (up to summation-order float noise well under the 1e-9 acceptance band
    when realism is active; exactly identical when realism=None).
    """
    pop_size = population.shape[0]
    n = len(boxes)
    if pop_size == 0 or n == 0:
        return np.array([], dtype=np.float64)

    # Effective container dims include positive-side overhang (mirrors
    # decode_chromosome). Constraint path uses L_eff/W_eff; geometric
    # path uses raw L/W when has_constraints=False.
    L = int(round(pallet.length))
    W = int(round(pallet.width))
    H = int(round(pallet.height))
    L_eff = L + int(round(max_overhang)) if max_overhang > 0 else L
    W_eff = W + int(round(max_overhang)) if max_overhang > 0 else W
    # F17 deck check / F19 transitive load — mirrors decode_chromosome.
    pallet_l = L if max_overhang > 0 else 0
    pallet_w = W if max_overhang > 0 else 0
    transitive = 1 if getattr(config, "transitive_load_bearing", False) else 0

    cstr_active = (has_constraints and weights is not None
                   and mlot is not None and rfs is not None)

    # Per-chromosome BPS argsort for fitness computation. The batch entry
    # also argsorts internally; redoing here is cheap and keeps the fitness
    # pure-numpy. (~1ms for pop_size=80, n=80.)
    bps = np.ascontiguousarray(population)[:, :n]
    orders = np.argsort(bps, axis=1).astype(np.int64)

    # Determine each chromosome's mode (uniform 0 when not multi-decoder).
    if use_multi_decoder and population.shape[1] >= 2 * n + 1:
        selectors = population[:, -1]
        if mode_cdf is None:
            modes = np.minimum(n_modes - 1,
                               (selectors * n_modes).astype(np.int64))
        else:
            # Inverse-CDF lookup, one per chromosome.
            modes = np.minimum(n_modes - 1,
                               np.searchsorted(mode_cdf, selectors, side='right'))
    else:
        modes = np.zeros(pop_size, dtype=np.int64)

    # Allocate the per-chromosome output buffers up front. All mode groups
    # write into different rows of the same buffer — no contention.
    placements_out_all = np.zeros((pop_size, n, 6), dtype=np.int64)
    n_bins_out_all = np.zeros(pop_size, dtype=np.int64)

    n_skus = (int(sku_id_per_box.max()) + 1
              if sku_id_per_box is not None else 0)

    # Group chromosomes by mode + dispatch.
    for m in np.unique(modes):
        idx = np.where(modes == m)[0]
        pop_m = population[idx]
        po_m = np.zeros((len(idx), n, 6), dtype=np.int64)
        nb_m = np.zeros(len(idx), dtype=np.int64)
        _decode_mode_batch(
            pop_m, n_rots_arr, dims_all, L_eff, W_eff, H, max_pallets,
            int(m), po_m, nb_m,
            cstr_active=cstr_active, has_constraints=has_constraints,
            sku_id_per_box=sku_id_per_box,
            sku_best_block=sku_best_block, sku_top_k_blocks=sku_top_k_blocks,
            weights=weights, mlot=mlot, rfs=rfs,
            pallet_max_weight=pallet_max_weight,
            support_ratio=support_ratio,
            require_centroid=require_centroid,
            cog_x_min=cog_x_min, cog_x_max=cog_x_max,
            cog_y_min=cog_y_min, cog_y_max=cog_y_max,
            cog_min_load_frac=cog_min_load_frac, cog_active=cog_active,
            L=L, W=W, n_skus=n_skus,
            pallet_l=pallet_l, pallet_w=pallet_w, transitive=transitive,
        )
        placements_out_all[idx] = po_m
        n_bins_out_all[idx] = nb_m

    return _compute_fitness_pallet1_batch(
        placements_out_all, orders, dims_all, pallet, realism=realism)


def _decode_mode_batch(
    population, n_rots_arr, dims_all,
    L_eff, W_eff, H, max_pallets, mode,
    placements_out_all, n_bins_out,
    *,
    cstr_active, has_constraints,
    sku_id_per_box, sku_best_block, sku_top_k_blocks,
    weights, mlot, rfs,
    pallet_max_weight, support_ratio, require_centroid,
    cog_x_min, cog_x_max, cog_y_min, cog_y_max,
    cog_min_load_frac, cog_active,
    L, W, n_skus,
    pallet_l=0, pallet_w=0, transitive=0,
):
    """Dispatch a homogeneous (single-mode) sub-population to the right
    batch entry. Mirrors the mode-branching in decode_chromosome but
    operates on populations. pallet_l/pallet_w/transitive: see
    decode_chromosome (F17 deck check / F19 transitive load).
    """
    if mode == 3:
        if cstr_active:
            decode_batch_layer_njit_cstr(
                population, n_rots_arr, dims_all, L_eff, W_eff, H, max_pallets,
                weights, mlot, rfs, pallet_max_weight, support_ratio,
                require_centroid,
                cog_x_min, cog_x_max, cog_y_min, cog_y_max,
                cog_min_load_frac, cog_active,
                placements_out_all, n_bins_out,
                pallet_l, pallet_w, transitive,
            )
        else:
            decode_batch_layer_njit(
                population, n_rots_arr, dims_all, L_eff, W_eff, H, max_pallets,
                placements_out_all, n_bins_out,
            )
    elif mode == 4 or mode == 5:
        # Mode 5 under constraints delegates to cstr mode 4 batch — matches
        # the per-chromosome dispatch in decode_chromosome.
        if cstr_active and sku_id_per_box is not None:
            decode_batch_blocks_njit_mode_cstr(
                population, n_rots_arr, dims_all, sku_id_per_box,
                L_eff, W_eff, H, max_pallets,
                weights, mlot, rfs, pallet_max_weight, support_ratio,
                require_centroid,
                cog_x_min, cog_x_max, cog_y_min, cog_y_max,
                cog_min_load_frac, cog_active,
                placements_out_all, n_bins_out, n_skus,
                pallet_l, pallet_w, transitive,
            )
        elif cstr_active and sku_id_per_box is None:
            decode_batch_njit_mode_cstr(
                population, n_rots_arr, dims_all, L_eff, W_eff, H, max_pallets,
                0,  # fall back to mode 0 when no SKU info
                weights, mlot, rfs, pallet_max_weight, support_ratio,
                require_centroid,
                cog_x_min, cog_x_max, cog_y_min, cog_y_max,
                cog_min_load_frac, cog_active,
                placements_out_all, n_bins_out,
                pallet_l, pallet_w, transitive,
            )
        elif sku_id_per_box is None:
            decode_batch_njit_mode(
                population, n_rots_arr, dims_all, L, W, H, max_pallets,
                0, placements_out_all, n_bins_out,
            )
        elif mode == 5 and sku_top_k_blocks is not None:
            # Per-chromosome top-K resolution (v3.8): each chromosome picks
            # its own (k, l, m, rot) per SKU from top_k_blocks via chrom keys.
            # Resolve outside nogil (numpy/Python), then run the batch entry
            # that takes the per-chrom chosen array.
            n_boxes = n_rots_arr.shape[0]
            chosen_per_chrom = np.zeros(
                (population.shape[0], n_skus, 4), dtype=np.int64)
            for ci in range(population.shape[0]):
                chosen_per_chrom[ci] = resolve_sku_blocks_from_chrom(
                    population[ci], n_boxes, n_skus, sku_top_k_blocks)
            decode_batch_precomputed_blocks_per_chrom(
                population, n_rots_arr, dims_all, sku_id_per_box,
                chosen_per_chrom,
                L, W, H, max_pallets, placements_out_all, n_bins_out, n_skus,
            )
        elif mode == 5 and sku_best_block is not None:
            decode_batch_precomputed_blocks_njit_mode(
                population, n_rots_arr, dims_all, sku_id_per_box, sku_best_block,
                L, W, H, max_pallets, placements_out_all, n_bins_out, n_skus,
            )
        else:
            # Mode 4 or mode 5 with top-K (top-K per-chrom block resolution
            # would need a special path — for now, fall back to dynamic blocks
            # which is what mode 4 does anyway). Geometric path uses L, W.
            decode_batch_blocks_njit_mode(
                population, n_rots_arr, dims_all, sku_id_per_box,
                L, W, H, max_pallets, placements_out_all, n_bins_out, n_skus,
            )
    else:
        # Modes 0/1/2.
        if cstr_active:
            decode_batch_njit_mode_cstr(
                population, n_rots_arr, dims_all, L_eff, W_eff, H, max_pallets,
                mode,
                weights, mlot, rfs, pallet_max_weight, support_ratio,
                require_centroid,
                cog_x_min, cog_x_max, cog_y_min, cog_y_max,
                cog_min_load_frac, cog_active,
                placements_out_all, n_bins_out,
                pallet_l, pallet_w, transitive,
            )
        else:
            decode_batch_njit_mode(
                population, n_rots_arr, dims_all, L, W, H, max_pallets,
                mode, placements_out_all, n_bins_out,
            )
