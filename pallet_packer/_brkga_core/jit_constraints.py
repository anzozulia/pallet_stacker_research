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
    require_centroid: int,
    require_full_support: int,
    pallet_l: int,
    pallet_w: int,
    pending_loads: np.ndarray,
) -> bool:
    """Returns True if placing candidate would not violate any supporter's
    max_load_on_top AND the placement's support is geometrically valid.

    pending_loads (round 3, F20): per-row load contributions ACCEPTED but
    not yet committed — the block decoder's sibling-column overlay. All
    other callers pass an all-zero array (same behavior as before). The
    load compare reads placement_top_loads[i] + pending_loads[i], so a
    block's columns are checked against the running aggregate instead of
    the stale pre-block state.

    Geometric checks (when cand_z > 0):
      - Total contact area / footprint >= effective_support_ratio
        (= 1.0 if require_full_support else support_ratio)
      - When require_centroid: footprint centroid (cand_x + cand_dx/2,
        cand_y + cand_dy/2) must lie within at least one supporter's
        XY footprint
    Floor check (cand_z <= 0, only when pallet_l > 0):
      - pallet_l/pallet_w are the RAW deck dims, passed by dispatch only
        when pallet overhang is active (else the 0 sentinel keeps the
        historical unconditional-True path, bit-identical). The box must
        rest ON the deck: deck-contact area / footprint >= the effective
        support ratio — without this, overhang lets floor boxes sit fully
        off the deck, floating in air (F17). No centroid-over-deck on
        purpose: the v2 engine never applies the centroid rule at floor
        level.
    Load check:
      - Distributes candidate weight across supporters in proportion to
        contact area; rejects if any supporter's running top-load + share
        exceeds its mlot.
    """
    if cand_z <= 0:
        if pallet_l <= 0:
            return True
        x_hi = cand_x + cand_dx if cand_x + cand_dx < pallet_l else pallet_l
        y_hi = cand_y + cand_dy if cand_y + cand_dy < pallet_w else pallet_w
        if x_hi <= cand_x or y_hi <= cand_y:
            return False
        contact = float((x_hi - cand_x) * (y_hi - cand_y))
        eff_sr0 = 1.0 if require_full_support != 0 else support_ratio
        if eff_sr0 > 0.0:
            footprint0 = float(cand_dx * cand_dy)
            if footprint0 > 0 and contact / footprint0 < eff_sr0 - 1e-6:
                return False
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
        # Scale-aware tolerance (round 3, F25): the old absolute 1e-6 is
        # below one double ulp for limits >= ~4.5e9 (contract allows 1e12),
        # making accept/reject flip on accumulation-order noise.
        lim = mlot[sup_box]
        eps = 1e-9 * lim
        if eps < 1e-6:
            eps = 1e-6
        if placement_top_loads[i] + pending_loads[i] + share > lim + eps:
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


@njit(cache=True, fastmath=True)
def _check_load_transitive_njit(
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
    tl_inc: np.ndarray,
    tl_touched: np.ndarray,
    pending_loads: np.ndarray,
) -> bool:
    """Transitive load CHECK (PackerConfig.transitive_load_bearing): a
    dry-run of _apply_load_contribution_transitive_njit that rejects if any
    box in the downward chain would exceed its max_load_on_top.
    pending_loads: the block sibling-column overlay (F20) — all-zero from
    every non-block caller; see _check_load_on_top_njit.

    Needed because the check-direct / commit-transitive split (v2's model)
    never rejects a FRESH pure column: each new box's direct supporter
    carries only its immediate rider, so a 10-high stack of individually
    legal links still crushes the bottom box (finding F19). The dry-run
    walks the same flow the commit would push and tests every affected row.
    tl_inc / tl_touched: caller scratch, all-zero in/out (shared with the
    apply — never both in flight).
    """
    if cand_z <= 0 or cand_weight <= 0:
        return True
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
        return True     # no supporters: the geometric check handles this
    n_touched = 0
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
        if share <= 0.0:
            continue
        if tl_inc[i] == 0.0:
            tl_touched[n_touched] = i
            n_touched += 1
        tl_inc[i] += share
    ok = True
    n_done = 0
    while n_done < n_touched:
        best_k = n_done
        best_row = tl_touched[n_done]
        best_z = placements_out[best_row, 4]
        for k in range(n_done + 1, n_touched):
            row_k = tl_touched[k]
            z_k = placements_out[row_k, 4]
            if z_k > best_z or (z_k == best_z and row_k < best_row):
                best_k = k
                best_row = row_k
                best_z = z_k
        tl_touched[best_k] = tl_touched[n_done]
        tl_touched[n_done] = best_row
        n_done += 1
        inc = tl_inc[best_row]
        lim = mlot[bps_order[best_row]]
        eps = 1e-9 * lim         # scale-aware tolerance (round 3, F25)
        if eps < 1e-6:
            eps = 1e-6
        if ok and (placement_top_loads[best_row] + pending_loads[best_row]
                   + inc > lim + eps):
            ok = False        # keep walking only to zero the scratch
        tl_inc[best_row] = 0.0
        if not ok:
            continue
        row_z = placements_out[best_row, 4]
        if row_z <= 0 or inc <= 0.0:
            continue
        r_box = bps_order[best_row]
        r_rot = placements_out[best_row, 1]
        r_x = placements_out[best_row, 2]
        r_y = placements_out[best_row, 3]
        r_dx = dims_all[r_box, r_rot, 0]
        r_dy = dims_all[r_box, r_rot, 1]
        r_pallet = placements_out[best_row, 0]
        tarea = 0.0
        for i in range(n_placed):
            if placements_out[i, 5] == 0:
                continue
            if placements_out[i, 0] != r_pallet:
                continue
            sup_box = bps_order[i]
            sup_rot = placements_out[i, 1]
            sup_dz = dims_all[sup_box, sup_rot, 2]
            sup_z = placements_out[i, 4]
            if sup_z + sup_dz != row_z:
                continue
            sup_dx = dims_all[sup_box, sup_rot, 0]
            sup_dy = dims_all[sup_box, sup_rot, 1]
            sup_x = placements_out[i, 2]
            sup_y = placements_out[i, 3]
            x_lo = r_x if r_x > sup_x else sup_x
            x_hi = r_x + r_dx if r_x + r_dx < sup_x + sup_dx else sup_x + sup_dx
            if x_hi <= x_lo:
                continue
            y_lo = r_y if r_y > sup_y else sup_y
            y_hi = r_y + r_dy if r_y + r_dy < sup_y + sup_dy else sup_y + sup_dy
            if y_hi <= y_lo:
                continue
            tarea += float((x_hi - x_lo) * (y_hi - y_lo))
        if tarea <= 0.0:
            continue
        for i in range(n_placed):
            if placements_out[i, 5] == 0:
                continue
            if placements_out[i, 0] != r_pallet:
                continue
            sup_box = bps_order[i]
            sup_rot = placements_out[i, 1]
            sup_dz = dims_all[sup_box, sup_rot, 2]
            sup_z = placements_out[i, 4]
            if sup_z + sup_dz != row_z:
                continue
            sup_dx = dims_all[sup_box, sup_rot, 0]
            sup_dy = dims_all[sup_box, sup_rot, 1]
            sup_x = placements_out[i, 2]
            sup_y = placements_out[i, 3]
            x_lo = r_x if r_x > sup_x else sup_x
            x_hi = r_x + r_dx if r_x + r_dx < sup_x + sup_dx else sup_x + sup_dx
            if x_hi <= x_lo:
                continue
            y_lo = r_y if r_y > sup_y else sup_y
            y_hi = r_y + r_dy if r_y + r_dy < sup_y + sup_dy else sup_y + sup_dy
            if y_hi <= y_lo:
                continue
            area = float((x_hi - x_lo) * (y_hi - y_lo))
            w = inc * (area / tarea)
            if w <= 0.0:
                continue
            if tl_inc[i] == 0.0:
                tl_touched[n_touched] = i
                n_touched += 1
            tl_inc[i] += w
    return ok


@njit(cache=True, fastmath=True)
def _apply_load_contribution_transitive_njit(
    placements_out: np.ndarray,
    dims_all: np.ndarray,
    bps_order: np.ndarray,
    placement_top_loads: np.ndarray,
    n_placed: int,
    cand_pallet: int,
    cand_x: int, cand_y: int, cand_z: int,
    cand_dx: int, cand_dy: int, cand_dz: int,
    cand_weight: float,
    tl_inc: np.ndarray,
    tl_touched: np.ndarray,
) -> None:
    """Transitive variant of _apply_load_contribution_njit
    (PackerConfig.transitive_load_bearing): the candidate's weight flows
    through its direct supporters and on down every support chain to the
    floor, each hop split by contact-area fraction — mirroring the v2
    engine's _commit + _propagate_load. The CHECK stays direct
    (check-direct / commit-transitive is exactly v2's model), so
    _check_load_on_top_njit is unchanged and compares each direct share
    against the now-transitively-accumulated running loads.

    tl_inc (float64[n]) / tl_touched (int64[n]) are caller-owned scratch;
    tl_inc must be all-zero on entry and is restored to all-zero on exit
    (only touched entries are written). Rows are processed highest
    bottom-z first, ties by lowest row index — supporters are always
    strictly lower, so every row is finalised before its own increment is
    pushed down; this ordering is the bit-identity contract with the
    Cython twin.
    """
    if cand_z <= 0 or cand_weight <= 0:
        return
    # Seed pass 1: total direct-supporter contact area (identical scan to
    # the direct apply).
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
    # Seed pass 2: credit each direct supporter and queue it.
    n_touched = 0
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
        if share <= 0.0:
            continue           # float underflow: nothing to propagate
        placement_top_loads[i] += share
        if tl_inc[i] == 0.0:
            tl_touched[n_touched] = i
            n_touched += 1
        tl_inc[i] += share
    # Worklist: push each queued row's increment down to ITS supporters.
    # Max-z-first guarantees single-visit: a processed row's z is >= every
    # later row's z, and new additions sit strictly below the row being
    # processed, so weight never flows into an already-processed row.
    n_done = 0
    while n_done < n_touched:
        best_k = n_done
        best_row = tl_touched[n_done]
        best_z = placements_out[best_row, 4]
        for k in range(n_done + 1, n_touched):
            row_k = tl_touched[k]
            z_k = placements_out[row_k, 4]
            if z_k > best_z or (z_k == best_z and row_k < best_row):
                best_k = k
                best_row = row_k
                best_z = z_k
        tl_touched[best_k] = tl_touched[n_done]
        tl_touched[n_done] = best_row
        n_done += 1
        inc = tl_inc[best_row]
        tl_inc[best_row] = 0.0
        row_z = placements_out[best_row, 4]
        if row_z <= 0 or inc <= 0.0:
            continue                       # floor row: the flow terminates
        r_box = bps_order[best_row]
        r_rot = placements_out[best_row, 1]
        r_x = placements_out[best_row, 2]
        r_y = placements_out[best_row, 3]
        r_dx = dims_all[r_box, r_rot, 0]
        r_dy = dims_all[r_box, r_rot, 1]
        r_pallet = placements_out[best_row, 0]
        tarea = 0.0
        for i in range(n_placed):
            if placements_out[i, 5] == 0:
                continue
            if placements_out[i, 0] != r_pallet:
                continue
            sup_box = bps_order[i]
            sup_rot = placements_out[i, 1]
            sup_dz = dims_all[sup_box, sup_rot, 2]
            sup_z = placements_out[i, 4]
            if sup_z + sup_dz != row_z:
                continue
            sup_dx = dims_all[sup_box, sup_rot, 0]
            sup_dy = dims_all[sup_box, sup_rot, 1]
            sup_x = placements_out[i, 2]
            sup_y = placements_out[i, 3]
            x_lo = r_x if r_x > sup_x else sup_x
            x_hi = r_x + r_dx if r_x + r_dx < sup_x + sup_dx else sup_x + sup_dx
            if x_hi <= x_lo:
                continue
            y_lo = r_y if r_y > sup_y else sup_y
            y_hi = r_y + r_dy if r_y + r_dy < sup_y + sup_dy else sup_y + sup_dy
            if y_hi <= y_lo:
                continue
            tarea += float((x_hi - x_lo) * (y_hi - y_lo))
        if tarea <= 0.0:
            continue
        for i in range(n_placed):
            if placements_out[i, 5] == 0:
                continue
            if placements_out[i, 0] != r_pallet:
                continue
            sup_box = bps_order[i]
            sup_rot = placements_out[i, 1]
            sup_dz = dims_all[sup_box, sup_rot, 2]
            sup_z = placements_out[i, 4]
            if sup_z + sup_dz != row_z:
                continue
            sup_dx = dims_all[sup_box, sup_rot, 0]
            sup_dy = dims_all[sup_box, sup_rot, 1]
            sup_x = placements_out[i, 2]
            sup_y = placements_out[i, 3]
            x_lo = r_x if r_x > sup_x else sup_x
            x_hi = r_x + r_dx if r_x + r_dx < sup_x + sup_dx else sup_x + sup_dx
            if x_hi <= x_lo:
                continue
            y_lo = r_y if r_y > sup_y else sup_y
            y_hi = r_y + r_dy if r_y + r_dy < sup_y + sup_dy else sup_y + sup_dy
            if y_hi <= y_lo:
                continue
            area = float((x_hi - x_lo) * (y_hi - y_lo))
            w = inc * (area / tarea)
            if w <= 0.0:
                continue       # float underflow: nothing to propagate
            placement_top_loads[i] += w
            if tl_inc[i] == 0.0:
                tl_touched[n_touched] = i
                n_touched += 1
            tl_inc[i] += w


@njit(cache=True, fastmath=True)
def _rider_inflow_njit(
    placements_out: np.ndarray,
    dims_all: np.ndarray,
    bps_order: np.ndarray,
    weights: np.ndarray,
    placement_top_loads: np.ndarray,
    n_placed: int,
    cand_pallet: int,
    cand_x: int, cand_y: int, cand_z: int,
    cand_dx: int, cand_dy: int, cand_dz: int,
    transitive: int,
) -> float:
    """Load the candidate would INHERIT by becoming a NEW supporter of
    already-placed boxes (round 3, F21 "under-fill").

    A box placed with its top plane exactly at an existing box's bottom,
    overlapping it in XY, physically takes a contact-share of that rider's
    outflow. The downward checks never see this, so the caller must (a)
    reject when the inherited load exceeds the candidate's own
    max_load_on_top, (b) under the transitive model flow
    cand_weight + inherited downward instead of cand_weight, and (c) book
    the inherited load onto the candidate's own row at commit.

    Per rider R: T_old = R's existing supporter contact area, a = contact
    with the candidate, share = out_R * a / (T_old + a) where out_R is R's
    weight plus (transitive model only) the load already booked on R. Old
    supporters are deliberately NOT debited — strictly conservative, no
    negative propagation, no float dust. Zero-weight riders contribute
    nothing (keeps weightless workloads bit-identical).
    """
    cand_top = cand_z + cand_dz
    inherited = 0.0
    for i in range(n_placed):
        if placements_out[i, 5] == 0:
            continue
        if placements_out[i, 0] != cand_pallet:
            continue
        if placements_out[i, 4] != cand_top:
            continue
        r_box = bps_order[i]
        r_rot = placements_out[i, 1]
        r_dx = dims_all[r_box, r_rot, 0]
        r_dy = dims_all[r_box, r_rot, 1]
        r_x = placements_out[i, 2]
        r_y = placements_out[i, 3]
        x_lo = cand_x if cand_x > r_x else r_x
        x_hi = cand_x + cand_dx if cand_x + cand_dx < r_x + r_dx else r_x + r_dx
        if x_hi <= x_lo:
            continue
        y_lo = cand_y if cand_y > r_y else r_y
        y_hi = cand_y + cand_dy if cand_y + cand_dy < r_y + r_dy else r_y + r_dy
        if y_hi <= y_lo:
            continue
        a = float((x_hi - x_lo) * (y_hi - y_lo))
        out_r = weights[r_box]
        if transitive != 0:
            out_r += placement_top_loads[i]
        if out_r <= 0.0:
            continue
        # T_old: the rider's existing supporter contact (the candidate is
        # not committed yet, so it is not in this scan).
        t_old = 0.0
        for j in range(n_placed):
            if j == i:
                continue
            if placements_out[j, 5] == 0:
                continue
            if placements_out[j, 0] != cand_pallet:
                continue
            s_box = bps_order[j]
            s_rot = placements_out[j, 1]
            s_dz = dims_all[s_box, s_rot, 2]
            s_z = placements_out[j, 4]
            if s_z + s_dz != cand_top:
                continue
            s_dx = dims_all[s_box, s_rot, 0]
            s_dy = dims_all[s_box, s_rot, 1]
            s_x = placements_out[j, 2]
            s_y = placements_out[j, 3]
            sx_lo = r_x if r_x > s_x else s_x
            sx_hi = r_x + r_dx if r_x + r_dx < s_x + s_dx else s_x + s_dx
            if sx_hi <= sx_lo:
                continue
            sy_lo = r_y if r_y > s_y else s_y
            sy_hi = r_y + r_dy if r_y + r_dy < s_y + s_dy else s_y + s_dy
            if sy_hi <= sy_lo:
                continue
            t_old += float((sx_hi - sx_lo) * (sy_hi - sy_lo))
        inherited += out_r * (a / (t_old + a))
    return inherited
