"""
pallet_packer._brkga_core.jit_constraints — JIT feasibility + bookkeeping
helpers used by every constraint-aware decoder.

  _check_load_on_top_njit       support_ratio + load-bearing + centroid +
                                per-box requires_full_support
  _check_cog_envelope_njit      pallet CoG envelope (gated by min_load_frac)
  _apply_load_contribution_njit update placement_top_loads after commit
  _apply_cog_contribution_njit  update pallet_sum_xw / pallet_sum_yw

Sentinel constant `_NO_LIMIT` (1e18) represents "no cap" without using
infinity (keeps JIT comparisons cheap and nan-safe).
"""
from __future__ import annotations

import numpy as np
from numba import njit


# Sentinel: "no limit" stored as a large finite number so JIT comparisons
# work without nan/inf-aware paths. Mirrors precompute._NO_LIMIT.
_NO_LIMIT = 1e18


@njit(cache=True, fastmath=True)
def _check_load_on_top_njit(
    placements_out: np.ndarray,
    dims_all: np.ndarray,
    bps_order: np.ndarray,
    mlot: np.ndarray,
    placement_top_loads: np.ndarray,
    n_placed: int,
    cand_pallet: int,
    cand_x: int, cand_y: int, cand_z: int,
    cand_dx: int, cand_dy: int, cand_dz: int,
    cand_weight: float,
    support_ratio: float,
    require_centroid: int = 0,
    require_full_support: int = 0,
) -> bool:
    """Returns True if placing candidate would not violate any supporter's
    max_load_on_top AND the placement's support is geometrically valid.

    Geometric checks (when cand_z > 0):
      - Total contact area / footprint >= effective_support_ratio
        (= 1.0 if require_full_support else support_ratio)
      - When require_centroid: footprint centroid (cand_x + cand_dx/2,
        cand_y + cand_dy/2) must lie within at least one supporter's
        XY footprint
    Load check:
      - Distributes candidate weight across supporters in proportion to
        contact area; rejects if any supporter's running top-load + share
        exceeds its mlot.
    """
    if cand_z <= 0:
        return True
    # Centroid coordinates (integer * 2 to keep math exact).
    cx2 = 2 * cand_x + cand_dx
    cy2 = 2 * cand_y + cand_dy
    centroid_supported = 0
    # Pass 1: total contact area across supporters on this pallet.
    total_area = 0.0
    for i in range(n_placed):
        if placements_out[i, 5] == 0:
            continue
        if placements_out[i, 0] != cand_pallet:
            continue
        sup_box = bps_order[i]
        sup_rot = placements_out[i, 1]
        sup_dz = dims_all[sup_box, sup_rot, 2]
        sup_z = placements_out[i, 4]
        if sup_z + sup_dz != cand_z:
            continue
        sup_dx = dims_all[sup_box, sup_rot, 0]
        sup_dy = dims_all[sup_box, sup_rot, 1]
        sup_x = placements_out[i, 2]
        sup_y = placements_out[i, 3]
        x_lo = cand_x if cand_x > sup_x else sup_x
        x_hi = cand_x + cand_dx if cand_x + cand_dx < sup_x + sup_dx else sup_x + sup_dx
        if x_hi <= x_lo:
            continue
        y_lo = cand_y if cand_y > sup_y else sup_y
        y_hi = cand_y + cand_dy if cand_y + cand_dy < sup_y + sup_dy else sup_y + sup_dy
        if y_hi <= y_lo:
            continue
        total_area += float((x_hi - x_lo) * (y_hi - y_lo))
        if centroid_supported == 0 and require_centroid != 0:
            if (cx2 >= 2 * sup_x and cx2 <= 2 * (sup_x + sup_dx)
                    and cy2 >= 2 * sup_y and cy2 <= 2 * (sup_y + sup_dy)):
                centroid_supported = 1
    if total_area <= 0.0:
        return False
    eff_sr = 1.0 if require_full_support != 0 else support_ratio
    if eff_sr > 0.0:
        footprint = float(cand_dx * cand_dy)
        if footprint > 0 and total_area / footprint < eff_sr - 1e-6:
            return False
    if require_centroid != 0 and centroid_supported == 0:
        return False
    # Pass 2: each supporter's accumulated top-load must not exceed mlot.
    for i in range(n_placed):
        if placements_out[i, 5] == 0:
            continue
        if placements_out[i, 0] != cand_pallet:
            continue
        sup_box = bps_order[i]
        sup_rot = placements_out[i, 1]
        sup_dz = dims_all[sup_box, sup_rot, 2]
        sup_z = placements_out[i, 4]
        if sup_z + sup_dz != cand_z:
            continue
        sup_dx = dims_all[sup_box, sup_rot, 0]
        sup_dy = dims_all[sup_box, sup_rot, 1]
        sup_x = placements_out[i, 2]
        sup_y = placements_out[i, 3]
        x_lo = cand_x if cand_x > sup_x else sup_x
        x_hi = cand_x + cand_dx if cand_x + cand_dx < sup_x + sup_dx else sup_x + sup_dx
        if x_hi <= x_lo:
            continue
        y_lo = cand_y if cand_y > sup_y else sup_y
        y_hi = cand_y + cand_dy if cand_y + cand_dy < sup_y + sup_dy else sup_y + sup_dy
        if y_hi <= y_lo:
            continue
        area = float((x_hi - x_lo) * (y_hi - y_lo))
        share = cand_weight * (area / total_area)
        if placement_top_loads[i] + share > mlot[sup_box] + 1e-6:
            return False
    return True


@njit(cache=True, fastmath=True)
def _check_cog_envelope_njit(
    cand_x: int, cand_y: int,
    cand_dx: int, cand_dy: int,
    cand_weight: float,
    cand_pallet: int,
    pallet_weights: np.ndarray,
    pallet_sum_xw: np.ndarray,
    pallet_sum_yw: np.ndarray,
    pallet_max_weight: float,
    cog_x_min: float, cog_x_max: float,
    cog_y_min: float, cog_y_max: float,
    cog_min_load_frac: float,
) -> bool:
    """Returns True if adding candidate keeps the pallet CoG inside the
    envelope after placement. Mirrors v2 packer.py:_cog_ok.

    The check only kicks in once the pallet is at least
    cog_min_load_frac of pallet_max_weight (avoids rejecting early
    placements that haven't accumulated enough weight to balance).
    """
    new_total = pallet_weights[cand_pallet] + cand_weight
    if new_total <= 0.0:
        return True
    if pallet_max_weight < _NO_LIMIT:
        if new_total < cog_min_load_frac * pallet_max_weight:
            return True
    cand_cx = float(cand_x) + 0.5 * float(cand_dx)
    cand_cy = float(cand_y) + 0.5 * float(cand_dy)
    new_sum_xw = pallet_sum_xw[cand_pallet] + cand_weight * cand_cx
    new_sum_yw = pallet_sum_yw[cand_pallet] + cand_weight * cand_cy
    cx = new_sum_xw / new_total
    cy = new_sum_yw / new_total
    if cx < cog_x_min - 1e-6 or cx > cog_x_max + 1e-6:
        return False
    if cy < cog_y_min - 1e-6 or cy > cog_y_max + 1e-6:
        return False
    return True


@njit(cache=True, fastmath=True)
def _apply_cog_contribution_njit(
    cand_x: int, cand_y: int,
    cand_dx: int, cand_dy: int,
    cand_weight: float,
    cand_pallet: int,
    pallet_sum_xw: np.ndarray,
    pallet_sum_yw: np.ndarray,
) -> None:
    """Update per-pallet weighted-position sums after committing a placement."""
    if cand_weight <= 0.0:
        return
    pallet_sum_xw[cand_pallet] += cand_weight * (float(cand_x) + 0.5 * float(cand_dx))
    pallet_sum_yw[cand_pallet] += cand_weight * (float(cand_y) + 0.5 * float(cand_dy))


@njit(cache=True, fastmath=True)
def _apply_load_contribution_njit(
    placements_out: np.ndarray,
    dims_all: np.ndarray,
    bps_order: np.ndarray,
    placement_top_loads: np.ndarray,
    n_placed: int,
    cand_pallet: int,
    cand_x: int, cand_y: int, cand_z: int,
    cand_dx: int, cand_dy: int, cand_dz: int,
    cand_weight: float,
) -> None:
    """Update placement_top_loads to add candidate's weight share to each
    of its supporters. Must be called AFTER _check_load_on_top_njit passes.
    """
    if cand_z <= 0 or cand_weight <= 0:
        return
    total_area = 0.0
    for i in range(n_placed):
        if placements_out[i, 5] == 0:
            continue
        if placements_out[i, 0] != cand_pallet:
            continue
        sup_box = bps_order[i]
        sup_rot = placements_out[i, 1]
        sup_dz = dims_all[sup_box, sup_rot, 2]
        sup_z = placements_out[i, 4]
        if sup_z + sup_dz != cand_z:
            continue
        sup_dx = dims_all[sup_box, sup_rot, 0]
        sup_dy = dims_all[sup_box, sup_rot, 1]
        sup_x = placements_out[i, 2]
        sup_y = placements_out[i, 3]
        x_lo = cand_x if cand_x > sup_x else sup_x
        x_hi = cand_x + cand_dx if cand_x + cand_dx < sup_x + sup_dx else sup_x + sup_dx
        if x_hi <= x_lo:
            continue
        y_lo = cand_y if cand_y > sup_y else sup_y
        y_hi = cand_y + cand_dy if cand_y + cand_dy < sup_y + sup_dy else sup_y + sup_dy
        if y_hi <= y_lo:
            continue
        total_area += float((x_hi - x_lo) * (y_hi - y_lo))
    if total_area <= 0.0:
        return
    for i in range(n_placed):
        if placements_out[i, 5] == 0:
            continue
        if placements_out[i, 0] != cand_pallet:
            continue
        sup_box = bps_order[i]
        sup_rot = placements_out[i, 1]
        sup_dz = dims_all[sup_box, sup_rot, 2]
        sup_z = placements_out[i, 4]
        if sup_z + sup_dz != cand_z:
            continue
        sup_dx = dims_all[sup_box, sup_rot, 0]
        sup_dy = dims_all[sup_box, sup_rot, 1]
        sup_x = placements_out[i, 2]
        sup_y = placements_out[i, 3]
        x_lo = cand_x if cand_x > sup_x else sup_x
        x_hi = cand_x + cand_dx if cand_x + cand_dx < sup_x + sup_dx else sup_x + sup_dx
        if x_hi <= x_lo:
            continue
        y_lo = cand_y if cand_y > sup_y else sup_y
        y_hi = cand_y + cand_dy if cand_y + cand_dy < sup_y + sup_dy else sup_y + sup_dy
        if y_hi <= y_lo:
            continue
        area = float((x_hi - x_lo) * (y_hi - y_lo))
        placement_top_loads[i] += cand_weight * (area / total_area)
