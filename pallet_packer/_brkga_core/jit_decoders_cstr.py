"""
pallet_packer._brkga_core.jit_decoders_cstr — constraint-aware JIT decoders.

Variants of the geometric decoders that enforce v3.10–v3.12 physical
constraints:
  - Pallet.max_weight  (per-pallet running weight cap)
  - Box.max_load_on_top  (direct-supporter load distribution)
  - support_ratio + per-box requires_full_support
  - require_centroid_supported  (footprint centroid over a supporter)
  - CoG envelope  (per-pallet weighted centroid)

  decode_njit_mode_cstr           cstr mode 0/1/2 (DFTRC/wall/corner)
  decode_blocks_njit_mode_cstr    cstr mode 4 (dynamic blocks)
  decode_layer_njit_cstr          cstr mode 3 (Bischoff-Ratcliff layers)

  _max_block_under_constraints_njit   shrink (k,l,m) under pallet+stack caps
  _commit_block_placements_njit       assign block positions + seed top-loads

Constraint mode 5 (top-K precomputed blocks) delegates to
decode_blocks_njit_mode_cstr at the dispatcher level — the precomputed
top-K selection has no value once weight/load caps apply.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from ..brkga_v3_fast import MAX_EMS, find_best_dftrc_njit, commit_ems_njit
from .jit_primitives import (
    find_best_wall_njit,
    find_best_corner_njit,
    find_best_in_slab_njit,
)
from .jit_decoders_geom import find_best_block_at_pos_njit
from .jit_constraints import (
    _check_load_on_top_njit,
    _check_load_transitive_njit,
    _rider_inflow_njit,
    _check_cog_envelope_njit,
    _apply_load_contribution_njit,
    _apply_load_contribution_transitive_njit,
    _apply_cog_contribution_njit,
)


@njit(cache=True, fastmath=True)
def decode_njit_mode_cstr(
    bps_order: np.ndarray,
    n_rots_per_box: np.ndarray,
    dims_all: np.ndarray,
    L: int, W: int, H: int,
    max_pallets: int,
    placements_out: np.ndarray,
    mode: int,  # 0=DFTRC, 1=wall, 2=corner
    weights: np.ndarray,
    mlot: np.ndarray,
    rfs: np.ndarray,
    pallet_max_weight: float,
    support_ratio: float,
    require_centroid: int,
    cog_x_min: float, cog_x_max: float,
    cog_y_min: float, cog_y_max: float,
    cog_min_load_frac: float,
    cog_active: int,
    pallet_l: int = 0,
    pallet_w: int = 0,
    transitive: int = 0,
) -> int:
    """Constraint-aware variant of decode_njit_mode (v3.10/v3.12).

    Enforces (v3.10): Pallet.max_weight, Box.max_load_on_top, support_ratio.
    Enforces (v3.12): per-box requires_full_support, centroid-supported,
    pallet CoG envelope. Caller passes L, W as L_eff, W_eff already inflated
    by max_overhang when applicable. Rejected candidate → try next bin.
    Hardening round 2: pallet_l/pallet_w are the RAW deck dims (0 sentinel
    unless overhang is active) for the floor deck-contact check (F17);
    transitive selects the transitive load commit (F19).
    """
    n = bps_order.shape[0]
    MAX_BINS = max_pallets if max_pallets > 0 else 32
    bin_emss = np.zeros((MAX_BINS, MAX_EMS, 2, 3), dtype=np.int64)
    bin_ems_count = np.zeros(MAX_BINS, dtype=np.int64)
    n_bins = 0
    scratch = np.zeros((MAX_EMS, 2, 3), dtype=np.int64)
    pallet_weights = np.zeros(MAX_BINS, dtype=np.float64)
    placement_top_loads = np.zeros(n, dtype=np.float64)
    tl_inc = np.zeros(n, dtype=np.float64)
    tl_touched = np.zeros(n, dtype=np.int64)
    # Round 3 (F20): sibling-column overlay for the check functions —
    # all-zero here (this decoder has no multi-commit paths); required by
    # the shared check signatures.
    blk_inc = np.zeros(n, dtype=np.float64)
    pallet_sum_xw = np.zeros(MAX_BINS, dtype=np.float64)
    pallet_sum_yw = np.zeros(MAX_BINS, dtype=np.float64)

    for i in range(n):
        box_idx = bps_order[i]
        n_rots = n_rots_per_box[box_idx]
        cand_weight = weights[box_idx]
        placed = False

        for b in range(n_bins):
            # Pre-check: pallet weight cap.
            if pallet_weights[b] + cand_weight > pallet_max_weight + 1e-6:
                continue
            # Find best (rot, x, y, z) for this bin.
            bin_best_rot = -1
            bin_best_score = -1
            bin_best_min = 1 << 62
            bin_best_x = 0
            bin_best_y = 0
            bin_best_z = 0
            for r in range(n_rots):
                dx = dims_all[box_idx, r, 0]
                dy = dims_all[box_idx, r, 1]
                dz = dims_all[box_idx, r, 2]
                if mode == 0:
                    idx, x, y, z = find_best_dftrc_njit(
                        bin_emss[b], bin_ems_count[b], dx, dy, dz, L, W, H)
                    if idx < 0:
                        continue
                    sc = ((L - x - dx) * (L - x - dx)
                          + (W - y - dy) * (W - y - dy)
                          + (H - z - dz) * (H - z - dz))
                    if sc > bin_best_score:
                        bin_best_score = sc
                        bin_best_rot = r
                        bin_best_x, bin_best_y, bin_best_z = x, y, z
                elif mode == 1:
                    idx, x, y, z = find_best_wall_njit(
                        bin_emss[b], bin_ems_count[b], dx, dy, dz, L, W, H)
                    if idx < 0:
                        continue
                    yz = ((W - y - dy) * (W - y - dy)
                          + (H - z - dz) * (H - z - dz))
                    BIG = (W + H) * (W + H) + 1
                    cand = x * BIG - yz
                    if cand < bin_best_min:
                        bin_best_min = cand
                        bin_best_rot = r
                        bin_best_x, bin_best_y, bin_best_z = x, y, z
                else:
                    idx, x, y, z = find_best_corner_njit(
                        bin_emss[b], bin_ems_count[b], dx, dy, dz, L, W, H)
                    if idx < 0:
                        continue
                    cand = x + y + z
                    if cand < bin_best_min:
                        bin_best_min = cand
                        bin_best_rot = r
                        bin_best_x, bin_best_y, bin_best_z = x, y, z

            if bin_best_rot < 0:
                continue  # no geometric fit in this bin
            dx = dims_all[box_idx, bin_best_rot, 0]
            dy = dims_all[box_idx, bin_best_rot, 1]
            dz = dims_all[box_idx, bin_best_rot, 2]
            # Load-on-top + support-ratio + centroid + per-box rfs check
            if not _check_load_on_top_njit(
                    placements_out, dims_all, bps_order, mlot,
                    placement_top_loads, n,
                    b, bin_best_x, bin_best_y, bin_best_z,
                    dx, dy, dz, cand_weight, support_ratio,
                    require_centroid, rfs[box_idx],
                    pallet_l, pallet_w, blk_inc):
                continue  # try next bin
            # Under-fill check (round 3, F21): if the candidate's top plane
            # meets an existing box's bottom, it becomes a NEW supporter and
            # inherits a contact-share of that rider's load. Reject when the
            # candidate can't carry it; otherwise it flows down with the
            # candidate's own weight (transitive) and is booked at commit.
            inherited = _rider_inflow_njit(
                placements_out, dims_all, bps_order, weights,
                placement_top_loads, n,
                b, bin_best_x, bin_best_y, bin_best_z,
                dx, dy, dz, transitive)
            if inherited > 0.0:
                lim_c = mlot[box_idx]
                eps_c = 1e-9 * lim_c
                if eps_c < 1e-6:
                    eps_c = 1e-6
                if inherited > lim_c + eps_c:
                    continue  # try next bin
            # Transitive load check (F19): the direct check never rejects a
            # fresh column — dry-run the downward flow too.
            if transitive != 0 and not _check_load_transitive_njit(
                    placements_out, dims_all, bps_order, mlot,
                    placement_top_loads, n,
                    b, bin_best_x, bin_best_y, bin_best_z,
                    dx, dy, dz, cand_weight + inherited,
                    tl_inc, tl_touched, blk_inc):
                continue  # try next bin
            # CoG envelope check (only when active).
            if cog_active != 0:
                if not _check_cog_envelope_njit(
                        bin_best_x, bin_best_y, dx, dy,
                        cand_weight, b,
                        pallet_weights, pallet_sum_xw, pallet_sum_yw,
                        pallet_max_weight,
                        cog_x_min, cog_x_max, cog_y_min, cog_y_max,
                        cog_min_load_frac):
                    continue
            # Commit
            new_count = commit_ems_njit(
                bin_emss[b], bin_ems_count[b],
                bin_best_x, bin_best_y, bin_best_z,
                bin_best_x + dx, bin_best_y + dy, bin_best_z + dz,
                scratch,
            )
            for j in range(new_count):
                bin_emss[b, j, 0, 0] = scratch[j, 0, 0]
                bin_emss[b, j, 0, 1] = scratch[j, 0, 1]
                bin_emss[b, j, 0, 2] = scratch[j, 0, 2]
                bin_emss[b, j, 1, 0] = scratch[j, 1, 0]
                bin_emss[b, j, 1, 1] = scratch[j, 1, 1]
                bin_emss[b, j, 1, 2] = scratch[j, 1, 2]
            bin_ems_count[b] = new_count
            placements_out[i, 0] = b
            placements_out[i, 1] = bin_best_rot
            placements_out[i, 2] = bin_best_x
            placements_out[i, 3] = bin_best_y
            placements_out[i, 4] = bin_best_z
            placements_out[i, 5] = 1
            pallet_weights[b] += cand_weight
            # F21: book the inherited rider load on the candidate's own row
            # (row i was just committed; its counter starts at 0).
            if inherited > 0.0:
                placement_top_loads[i] += inherited
            if transitive != 0:
                _apply_load_contribution_transitive_njit(
                    placements_out, dims_all, bps_order,
                    placement_top_loads, n,
                    b, bin_best_x, bin_best_y, bin_best_z,
                    dx, dy, dz, cand_weight + inherited,
                    tl_inc, tl_touched)
            else:
                _apply_load_contribution_njit(
                    placements_out, dims_all, bps_order,
                    placement_top_loads, n,
                    b, bin_best_x, bin_best_y, bin_best_z,
                    dx, dy, dz, cand_weight)
            _apply_cog_contribution_njit(
                bin_best_x, bin_best_y, dx, dy,
                cand_weight, b, pallet_sum_xw, pallet_sum_yw)
            placed = True
            break

        if not placed:
            if n_bins >= MAX_BINS:
                placements_out[i, 5] = 0
                continue
            # Open new bin and try to place there.
            if cand_weight > pallet_max_weight + 1e-6:
                # Box alone exceeds pallet cap — cannot pack at all.
                placements_out[i, 5] = 0
                continue
            bin_emss[n_bins, 0, 0, 0] = 0
            bin_emss[n_bins, 0, 0, 1] = 0
            bin_emss[n_bins, 0, 0, 2] = 0
            bin_emss[n_bins, 0, 1, 0] = L
            bin_emss[n_bins, 0, 1, 1] = W
            bin_emss[n_bins, 0, 1, 2] = H
            bin_ems_count[n_bins] = 1
            best_rot_n = -1
            best_score_n = -1
            best_min_n = 1 << 62
            best_x_n = 0
            best_y_n = 0
            best_z_n = 0
            for r in range(n_rots):
                dx = dims_all[box_idx, r, 0]
                dy = dims_all[box_idx, r, 1]
                dz = dims_all[box_idx, r, 2]
                if mode == 0:
                    idx, x, y, z = find_best_dftrc_njit(
                        bin_emss[n_bins], bin_ems_count[n_bins], dx, dy, dz, L, W, H)
                    if idx < 0:
                        continue
                    sc = ((L - x - dx) * (L - x - dx)
                          + (W - y - dy) * (W - y - dy)
                          + (H - z - dz) * (H - z - dz))
                    if sc > best_score_n:
                        best_score_n = sc
                        best_rot_n = r
                        best_x_n, best_y_n, best_z_n = x, y, z
                elif mode == 1:
                    idx, x, y, z = find_best_wall_njit(
                        bin_emss[n_bins], bin_ems_count[n_bins], dx, dy, dz, L, W, H)
                    if idx < 0:
                        continue
                    yz = ((W - y - dy) * (W - y - dy)
                          + (H - z - dz) * (H - z - dz))
                    BIG = (W + H) * (W + H) + 1
                    cand = x * BIG - yz
                    if cand < best_min_n:
                        best_min_n = cand
                        best_rot_n = r
                        best_x_n, best_y_n, best_z_n = x, y, z
                else:
                    idx, x, y, z = find_best_corner_njit(
                        bin_emss[n_bins], bin_ems_count[n_bins], dx, dy, dz, L, W, H)
                    if idx < 0:
                        continue
                    cand = x + y + z
                    if cand < best_min_n:
                        best_min_n = cand
                        best_rot_n = r
                        best_x_n, best_y_n, best_z_n = x, y, z
            if best_rot_n < 0:
                placements_out[i, 5] = 0
                continue
            # First placement in a new bin is always on the floor (z=0),
            # so no load-on-top check needed — but still respect cap.
            r = best_rot_n
            dx = dims_all[box_idx, r, 0]
            dy = dims_all[box_idx, r, 1]
            dz = dims_all[box_idx, r, 2]
            # Deck-contact check (F17): with overhang, L/W are inflated and
            # even a floor placement can sit off the raw deck.
            if pallet_l > 0:
                if not _check_load_on_top_njit(
                        placements_out, dims_all, bps_order, mlot,
                        placement_top_loads, n,
                        n_bins, best_x_n, best_y_n, best_z_n,
                        dx, dy, dz, cand_weight, support_ratio,
                        require_centroid, rfs[box_idx],
                        pallet_l, pallet_w, blk_inc):
                    placements_out[i, 5] = 0
                    continue
            new_count = commit_ems_njit(
                bin_emss[n_bins], bin_ems_count[n_bins],
                best_x_n, best_y_n, best_z_n,
                best_x_n + dx, best_y_n + dy, best_z_n + dz,
                scratch,
            )
            for j in range(new_count):
                bin_emss[n_bins, j, 0, 0] = scratch[j, 0, 0]
                bin_emss[n_bins, j, 0, 1] = scratch[j, 0, 1]
                bin_emss[n_bins, j, 0, 2] = scratch[j, 0, 2]
                bin_emss[n_bins, j, 1, 0] = scratch[j, 1, 0]
                bin_emss[n_bins, j, 1, 1] = scratch[j, 1, 1]
                bin_emss[n_bins, j, 1, 2] = scratch[j, 1, 2]
            bin_ems_count[n_bins] = new_count
            placements_out[i, 0] = n_bins
            placements_out[i, 1] = best_rot_n
            placements_out[i, 2] = best_x_n
            placements_out[i, 3] = best_y_n
            placements_out[i, 4] = best_z_n
            placements_out[i, 5] = 1
            pallet_weights[n_bins] += cand_weight
            # First box at z=0; no top-load update needed.
            _apply_cog_contribution_njit(
                best_x_n, best_y_n, dx, dy,
                cand_weight, n_bins, pallet_sum_xw, pallet_sum_yw)
            n_bins += 1
    return n_bins


# ============================================================================
# Constraint-aware block decoders (v3.11): mode 4 (dynamic blocks) and
# mode 5 (precomputed top-K blocks) with full constraint enforcement.
# ============================================================================

@njit(cache=True, fastmath=True)
def _commit_block_placements_njit(
    placements_out: np.ndarray,
    placed: np.ndarray,
    bps_order: np.ndarray,
    sku_id_per_box: np.ndarray,
    placement_top_loads: np.ndarray,
    dims_all: np.ndarray,
    start_i: int, n: int,
    best_bin: int, best_rot: int,
    best_x: int, best_y: int, best_z: int,
    dx: int, dy: int, dz: int,
    k: int, l: int, m: int,
    my_sku: int,
    box_weight: float,
    cand_pallet: int,
) -> int:
    """Assign block positions and seed internal top-loads. Returns count placed.

    Positions tile X-fastest (k), then Y (l), then Z (m), matching the
    geometric-mode block decoder. For each box at z-layer ll (0=bottom),
    its placement_top_loads is initialized to (m - 1 - ll) * box_weight —
    the weight of same-SKU boxes stacked above WITHIN the block. This is
    necessary so any future placement landing on the block's top layer
    sees an accurate prior load (external supporters above the block).
    """
    placed_count = 0
    kk = 0
    ll = 0
    mm = 0
    for j in range(start_i, n):
        if placed_count >= k * l * m:
            break
        if placed[j] == 1:
            continue
        if sku_id_per_box[bps_order[j]] != my_sku:
            continue
        px = best_x + kk * dx
        py = best_y + ll * dy
        pz = best_z + mm * dz
        placements_out[j, 0] = best_bin
        placements_out[j, 1] = best_rot
        placements_out[j, 2] = px
        placements_out[j, 3] = py
        placements_out[j, 4] = pz
        placements_out[j, 5] = 1
        placed[j] = 1
        # Internal top-load: this box has (m - 1 - mm) box-weights resting
        # on it from the block above (same SKU, so all same weight).
        # Top layer (mm = m-1) → 0; bottom layer (mm = 0) → (m-1) * weight.
        placement_top_loads[j] = float(m - 1 - mm) * box_weight
        placed_count += 1
        kk += 1
        if kk >= k:
            kk = 0
            ll += 1
            if ll >= l:
                ll = 0
                mm += 1
    return placed_count


@njit(cache=True, fastmath=True)
def _max_block_under_constraints_njit(
    k: int, l: int, m: int,
    box_weight: float,
    box_mlot: float,
    pallet_weight_remaining: float,
) -> tuple:
    """Reduce (k, l, m) to respect pallet weight cap + internal stack limit.

    Constraints:
      - Internal stack: (m - 1) * box_weight <= box_mlot  → caps m
      - Pallet weight cap: k * l * m * box_weight <= remaining  → caps total

    Reduction strategy when pallet cap binds: shrink m first, then l, then k.
    """
    # Internal stack cap.
    if box_weight > 0:
        max_m_stack = int(box_mlot / box_weight) + 1
        if max_m_stack < 1:
            max_m_stack = 1
        if m > max_m_stack:
            m = max_m_stack
    if m < 1:
        m = 1
    # Pallet cap.
    if box_weight > 0 and pallet_weight_remaining < box_weight * k * l * m:
        max_count = int(pallet_weight_remaining / box_weight)
        if max_count < 1:
            return 0, 0, 0
        while k * l * m > max_count and m > 1:
            m -= 1
        while k * l * m > max_count and l > 1:
            l -= 1
        while k * l * m > max_count and k > 1:
            k -= 1
        if k * l * m > max_count:
            return 0, 0, 0
    return k, l, m


@njit(cache=True, fastmath=True)
def decode_blocks_njit_mode_cstr(
    bps_order: np.ndarray,
    n_rots_per_box: np.ndarray,
    dims_all: np.ndarray,
    sku_id_per_box: np.ndarray,
    L: int, W: int, H: int,
    max_pallets: int,
    placements_out: np.ndarray,
    n_skus: int,
    weights: np.ndarray,
    mlot: np.ndarray,
    rfs: np.ndarray,
    pallet_max_weight: float,
    support_ratio: float,
    require_centroid: int,
    cog_x_min: float, cog_x_max: float,
    cog_y_min: float, cog_y_max: float,
    cog_min_load_frac: float,
    cog_active: int,
    pallet_l: int = 0,
    pallet_w: int = 0,
    transitive: int = 0,
) -> int:
    """Constraint-aware dynamic-block decoder (mode 4 + v3.11/v3.12).

    Like decode_blocks_njit_mode but enforces Pallet.max_weight,
    Box.max_load_on_top (both internal-block stack height AND external
    supporter loads), support_ratio, per-box requires_full_support,
    centroid-supported, and pallet CoG envelope (v3.12). When a candidate
    block doesn't satisfy constraints, falls back to (1, 1, 1) at the
    same DFTRC position; if that fails, tries the next bin.
    """
    n = bps_order.shape[0]
    MAX_BINS = max_pallets if max_pallets > 0 else 32
    bin_emss = np.zeros((MAX_BINS, MAX_EMS, 2, 3), dtype=np.int64)
    bin_ems_count = np.zeros(MAX_BINS, dtype=np.int64)
    n_bins = 0
    scratch = np.zeros((MAX_EMS, 2, 3), dtype=np.int64)
    placed = np.zeros(n, dtype=np.int64)
    pallet_weights = np.zeros(MAX_BINS, dtype=np.float64)
    placement_top_loads = np.zeros(n, dtype=np.float64)
    tl_inc = np.zeros(n, dtype=np.float64)
    tl_touched = np.zeros(n, dtype=np.int64)
    # Round 3 (F20): the sibling-column overlay. During Phase 2c each
    # ACCEPTED column's external contribution accumulates here so later
    # sibling columns are checked against the running aggregate instead of
    # the stale pre-block state (pre-fix a k*l block could jointly crush a
    # shared supporter that every per-column check individually passed).
    # All-zero outside Phase 2c.
    blk_inc = np.zeros(n, dtype=np.float64)
    pallet_sum_xw = np.zeros(MAX_BINS, dtype=np.float64)
    pallet_sum_yw = np.zeros(MAX_BINS, dtype=np.float64)

    sku_remaining = np.zeros(n_skus, dtype=np.int64)
    for i in range(n):
        sku_remaining[sku_id_per_box[bps_order[i]]] += 1

    for i in range(n):
        if placed[i] == 1:
            continue
        box_idx = bps_order[i]
        my_sku = sku_id_per_box[box_idx]
        n_rots = n_rots_per_box[box_idx]
        max_count = sku_remaining[my_sku]
        box_weight = weights[box_idx]
        box_mlot = mlot[box_idx]

        # If a single box alone exceeds pallet cap, can't pack at all.
        if box_weight > pallet_max_weight + 1e-6:
            placements_out[i, 5] = 0
            placed[i] = 1
            sku_remaining[my_sku] -= 1
            continue

        block_committed = False
        for b in range(n_bins):
            # Pre-check: pallet weight cap for at least a single box.
            if pallet_weights[b] + box_weight > pallet_max_weight + 1e-6:
                continue
            # Phase 1: DFTRC place single box.
            best_rot = -1
            best_score = -1
            best_x = 0
            best_y = 0
            best_z = 0
            for r in range(n_rots):
                dx = dims_all[box_idx, r, 0]
                dy = dims_all[box_idx, r, 1]
                dz = dims_all[box_idx, r, 2]
                idx, x, y, z = find_best_dftrc_njit(
                    bin_emss[b], bin_ems_count[b], dx, dy, dz, L, W, H)
                if idx < 0:
                    continue
                sc = ((L - x - dx) * (L - x - dx)
                      + (W - y - dy) * (W - y - dy)
                      + (H - z - dz) * (H - z - dz))
                if sc > best_score:
                    best_score = sc
                    best_rot = r
                    best_x, best_y, best_z = x, y, z
            if best_rot < 0:
                continue  # no geometric fit in this bin
            dx = dims_all[box_idx, best_rot, 0]
            dy = dims_all[box_idx, best_rot, 1]
            dz = dims_all[box_idx, best_rot, 2]
            # Phase 2: find geometric max block.
            k, l, m = find_best_block_at_pos_njit(
                bin_emss[b], bin_ems_count[b],
                best_x, best_y, best_z, dx, dy, dz, max_count,
            )
            # Phase 2b: reduce (k, l, m) under pallet-cap + stack-limit.
            cap_remain = pallet_max_weight - pallet_weights[b]
            k, l, m = _max_block_under_constraints_njit(
                k, l, m, box_weight, box_mlot, cap_remain)
            if k * l * m < 1:
                continue  # can't even fit a single box weight-wise
            # Phase 2c-pre (round 3, F21): a block column placed directly
            # beneath an existing box's bottom plane would become its NEW
            # supporter and inherit load the block bookkeeping cannot
            # carry. Blocks with riders on any column top are shrunk to a
            # single box (1x1x1), which gets exact rider semantics below.
            rider_free = True
            for bx_ll in range(l):
                for bx_kk in range(k):
                    if _rider_inflow_njit(
                            placements_out, dims_all, bps_order, weights,
                            placement_top_loads, n,
                            b, best_x + bx_kk * dx, best_y + bx_ll * dy,
                            best_z, dx, dy, m * dz, transitive) > 0.0:
                        rider_free = False
                        break
                if not rider_free:
                    break
            inherited_blk = 0.0
            if not rider_free:
                k, l, m = 1, 1, 1
                inherited_blk = _rider_inflow_njit(
                    placements_out, dims_all, bps_order, weights,
                    placement_top_loads, n,
                    b, best_x, best_y, best_z, dx, dy, dz, transitive)
                if inherited_blk > 0.0:
                    lim_c = mlot[box_idx]
                    eps_c = 1e-9 * lim_c
                    if eps_c < 1e-6:
                        eps_c = 1e-6
                    if inherited_blk > lim_c + eps_c:
                        continue  # next bin
            # Phase 2c: PER-bottom-box support + load check. The validator
            # checks every box individually, so a block-aggregate check is
            # wrong (a 3x1 bottom layer can average >= support_ratio while a
            # corner box sits at 0.4 -> the validator floats it). Check each of
            # the k*l bottom boxes at its own footprint. Mirrors the Cython
            # port's _decode_cstr_blocks_loop (kept bit-equivalent on purpose).
            # Round 3 (F20): checks run against placement_top_loads PLUS the
            # blk_inc overlay, and each ACCEPTED column's contribution is
            # accumulated into the overlay before the next column is checked
            # — sibling columns can no longer jointly crush a shared
            # supporter that each individual check passed.
            bottom_ok = True
            for bx_ll in range(l):
                for bx_kk in range(k):
                    if not _check_load_on_top_njit(
                            placements_out, dims_all, bps_order, mlot,
                            placement_top_loads, n,
                            b, best_x + bx_kk * dx, best_y + bx_ll * dy, best_z,
                            dx, dy, dz, box_weight, support_ratio,
                            require_centroid, rfs[box_idx],
                            pallet_l, pallet_w, blk_inc):
                        bottom_ok = False
                        break
                    # Transitive (F19): each bottom box relays its whole
                    # column (m*w) to the external chain below (plus the
                    # inherited rider load when shrunk to a single box).
                    if transitive != 0 and not _check_load_transitive_njit(
                            placements_out, dims_all, bps_order, mlot,
                            placement_top_loads, n,
                            b, best_x + bx_kk * dx, best_y + bx_ll * dy, best_z,
                            dx, dy, dz, float(m) * box_weight + inherited_blk,
                            tl_inc, tl_touched, blk_inc):
                        bottom_ok = False
                        break
                    # Accumulate the accepted column's external contribution
                    # into the overlay (writes blk_inc, not the real loads).
                    if transitive != 0:
                        _apply_load_contribution_transitive_njit(
                            placements_out, dims_all, bps_order,
                            blk_inc, n,
                            b, best_x + bx_kk * dx, best_y + bx_ll * dy,
                            best_z, dx, dy, dz,
                            float(m) * box_weight + inherited_blk,
                            tl_inc, tl_touched)
                    else:
                        _apply_load_contribution_njit(
                            placements_out, dims_all, bps_order,
                            blk_inc, n,
                            b, best_x + bx_kk * dx, best_y + bx_ll * dy,
                            best_z, dx, dy, dz, box_weight)
                if not bottom_ok:
                    break
            # Wipe the overlay (touched rows are not tracked — O(n) is fine,
            # Phase 2c runs once per block attempt).
            for zz in range(n):
                blk_inc[zz] = 0.0
            if not bottom_ok:
                # Try shrinking block (l, k → 1) before giving up on bin.
                k, l = 1, 1
                if _rider_inflow_njit(
                        placements_out, dims_all, bps_order, weights,
                        placement_top_loads, n,
                        b, best_x, best_y, best_z,
                        dx, dy, m * dz, transitive) > 0.0:
                    # Rider on the shrunk column's top: only the exact
                    # single-box path (m == 1, handled above) may carry
                    # inherited load — give up on this bin.
                    continue  # next bin
                if not _check_load_on_top_njit(
                        placements_out, dims_all, bps_order, mlot,
                        placement_top_loads, n,
                        b, best_x, best_y, best_z,
                        dx, dy, dz, box_weight, support_ratio,
                        require_centroid, rfs[box_idx],
                        pallet_l, pallet_w, blk_inc):
                    continue  # next bin
                if transitive != 0 and not _check_load_transitive_njit(
                        placements_out, dims_all, bps_order, mlot,
                        placement_top_loads, n,
                        b, best_x, best_y, best_z,
                        dx, dy, dz, float(m) * box_weight,
                        tl_inc, tl_touched, blk_inc):
                    continue  # next bin
            # Phase 2d: CoG envelope check (block treated as point mass at
            # the bottom-layer footprint centroid; total weight = k*l*m*w).
            if cog_active != 0:
                block_total_w = float(k * l * m) * box_weight
                if not _check_cog_envelope_njit(
                        best_x, best_y, k * dx, l * dy,
                        block_total_w, b,
                        pallet_weights, pallet_sum_xw, pallet_sum_yw,
                        pallet_max_weight,
                        cog_x_min, cog_x_max, cog_y_min, cog_y_max,
                        cog_min_load_frac):
                    continue
            # Phase 3: commit block.
            placed_so_far = _commit_block_placements_njit(
                placements_out, placed, bps_order, sku_id_per_box,
                placement_top_loads, dims_all, i, n,
                b, best_rot, best_x, best_y, best_z,
                dx, dy, dz, k, l, m, my_sku, box_weight, b)
            sku_remaining[my_sku] -= placed_so_far
            pallet_weights[b] += float(placed_so_far) * box_weight
            # F21: book the inherited rider load on the committed box's own
            # row (only ever nonzero for the shrunk 1x1x1 case, whose sole
            # committed row is i). Booked even at best_z == 0 — a floor box
            # under a rider carries the load with nothing below to flow to.
            if inherited_blk > 0.0:
                placement_top_loads[i] += inherited_blk
            # Apply load contribution from bottom layer to external supporters.
            # Each bottom-layer box contributes box_weight via its own footprint.
            # Iterate the same-SKU bottom-layer placements just committed.
            if best_z > 0:
                kk = 0
                ll = 0
                applied = 0
                for jj in range(i, n):
                    if applied >= k * l:
                        break
                    if placed[jj] != 1:
                        continue
                    if placements_out[jj, 0] != b:
                        continue
                    if placements_out[jj, 4] != best_z:
                        continue
                    if sku_id_per_box[bps_order[jj]] != my_sku:
                        continue
                    if transitive != 0:
                        # The bottom box carries its whole column (the
                        # block-internal seeding is already transitive), so
                        # the FULL column weight flows through it to the
                        # external supporters and on down (F19), plus any
                        # inherited rider load (F21; 1x1x1 only).
                        _apply_load_contribution_transitive_njit(
                            placements_out, dims_all, bps_order,
                            placement_top_loads, n,
                            b, placements_out[jj, 2], placements_out[jj, 3],
                            best_z, dx, dy, dz,
                            float(m) * box_weight + inherited_blk,
                            tl_inc, tl_touched)
                    else:
                        _apply_load_contribution_njit(
                            placements_out, dims_all, bps_order,
                            placement_top_loads, n,
                            b, placements_out[jj, 2], placements_out[jj, 3],
                            best_z, dx, dy, dz, box_weight)
                    applied += 1
            # Update CoG with each box in the block as a separate point mass
            # (using its actual position rather than collapsing to bottom-layer
            # centroid — keeps CoG tracking precise for the upper layers too).
            if box_weight > 0:
                kk2 = 0
                ll2 = 0
                mm2 = 0
                blk_count = k * l * m
                applied2 = 0
                for jj in range(i, n):
                    if applied2 >= blk_count:
                        break
                    if placed[jj] != 1:
                        continue
                    if placements_out[jj, 0] != b:
                        continue
                    if sku_id_per_box[bps_order[jj]] != my_sku:
                        continue
                    # Confirm this is one of the just-committed block boxes.
                    px = best_x + kk2 * dx
                    py = best_y + ll2 * dy
                    pz = best_z + mm2 * dz
                    if (placements_out[jj, 2] != px
                            or placements_out[jj, 3] != py
                            or placements_out[jj, 4] != pz):
                        continue
                    _apply_cog_contribution_njit(
                        px, py, dx, dy, box_weight, b,
                        pallet_sum_xw, pallet_sum_yw)
                    applied2 += 1
                    kk2 += 1
                    if kk2 >= k:
                        kk2 = 0
                        ll2 += 1
                        if ll2 >= l:
                            ll2 = 0
                            mm2 += 1
            # Single EMS commit for the entire block region.
            new_count = commit_ems_njit(
                bin_emss[b], bin_ems_count[b],
                best_x, best_y, best_z,
                best_x + k * dx, best_y + l * dy, best_z + m * dz,
                scratch,
            )
            for j in range(new_count):
                bin_emss[b, j, 0, 0] = scratch[j, 0, 0]
                bin_emss[b, j, 0, 1] = scratch[j, 0, 1]
                bin_emss[b, j, 0, 2] = scratch[j, 0, 2]
                bin_emss[b, j, 1, 0] = scratch[j, 1, 0]
                bin_emss[b, j, 1, 1] = scratch[j, 1, 1]
                bin_emss[b, j, 1, 2] = scratch[j, 1, 2]
            bin_ems_count[b] = new_count
            block_committed = True
            break

        if block_committed:
            continue

        # No existing bin accepted — open a new bin.
        if n_bins >= MAX_BINS:
            placements_out[i, 5] = 0
            placed[i] = 1
            sku_remaining[my_sku] -= 1
            continue
        bin_emss[n_bins, 0, 0, 0] = 0
        bin_emss[n_bins, 0, 0, 1] = 0
        bin_emss[n_bins, 0, 0, 2] = 0
        bin_emss[n_bins, 0, 1, 0] = L
        bin_emss[n_bins, 0, 1, 1] = W
        bin_emss[n_bins, 0, 1, 2] = H
        bin_ems_count[n_bins] = 1
        best_rot_n = -1
        best_score_n = -1
        best_x_n = 0
        best_y_n = 0
        best_z_n = 0
        for r in range(n_rots):
            dx = dims_all[box_idx, r, 0]
            dy = dims_all[box_idx, r, 1]
            dz = dims_all[box_idx, r, 2]
            idx, x, y, z = find_best_dftrc_njit(
                bin_emss[n_bins], bin_ems_count[n_bins], dx, dy, dz, L, W, H)
            if idx < 0:
                continue
            sc = ((L - x - dx) * (L - x - dx)
                  + (W - y - dy) * (W - y - dy)
                  + (H - z - dz) * (H - z - dz))
            if sc > best_score_n:
                best_score_n = sc
                best_rot_n = r
                best_x_n, best_y_n, best_z_n = x, y, z
        if best_rot_n < 0:
            placements_out[i, 5] = 0
            placed[i] = 1
            sku_remaining[my_sku] -= 1
            continue
        # First placement at z=0 is trivially supported; only pallet weight
        # cap matters (already checked above).
        dx = dims_all[box_idx, best_rot_n, 0]
        dy = dims_all[box_idx, best_rot_n, 1]
        dz = dims_all[box_idx, best_rot_n, 2]
        k, l, m = find_best_block_at_pos_njit(
            bin_emss[n_bins], bin_ems_count[n_bins],
            best_x_n, best_y_n, best_z_n, dx, dy, dz, max_count,
        )
        k, l, m = _max_block_under_constraints_njit(
            k, l, m, box_weight, box_mlot, pallet_max_weight)
        if k * l * m < 1:
            placements_out[i, 5] = 0
            placed[i] = 1
            sku_remaining[my_sku] -= 1
            continue
        # Deck-contact check (F17): under overhang the bottom layer can
        # extend past the raw deck; every bottom box needs deck contact.
        # Mirror Phase 2c's fallback: shrink to 1x1 before giving up.
        if pallet_l > 0:
            deck_ok = True
            for bx_ll in range(l):
                for bx_kk in range(k):
                    if not _check_load_on_top_njit(
                            placements_out, dims_all, bps_order, mlot,
                            placement_top_loads, n,
                            n_bins, best_x_n + bx_kk * dx,
                            best_y_n + bx_ll * dy, best_z_n,
                            dx, dy, dz, box_weight, support_ratio,
                            require_centroid, rfs[box_idx],
                            pallet_l, pallet_w, blk_inc):
                        deck_ok = False
                        break
                if not deck_ok:
                    break
            if not deck_ok:
                k, l = 1, 1
                if not _check_load_on_top_njit(
                        placements_out, dims_all, bps_order, mlot,
                        placement_top_loads, n,
                        n_bins, best_x_n, best_y_n, best_z_n,
                        dx, dy, dz, box_weight, support_ratio,
                        require_centroid, rfs[box_idx],
                        pallet_l, pallet_w, blk_inc):
                    placements_out[i, 5] = 0
                    placed[i] = 1
                    sku_remaining[my_sku] -= 1
                    continue
        # Floor placement (z=0), no support/load check needed for bottom.
        placed_so_far = _commit_block_placements_njit(
            placements_out, placed, bps_order, sku_id_per_box,
            placement_top_loads, dims_all, i, n,
            n_bins, best_rot_n, best_x_n, best_y_n, best_z_n,
            dx, dy, dz, k, l, m, my_sku, box_weight, n_bins)
        sku_remaining[my_sku] -= placed_so_far
        pallet_weights[n_bins] += float(placed_so_far) * box_weight
        # CoG contribution for each box in the new-bin block.
        if box_weight > 0:
            kk2 = 0
            ll2 = 0
            mm2 = 0
            blk_count = k * l * m
            applied2 = 0
            for jj in range(i, n):
                if applied2 >= blk_count:
                    break
                if placed[jj] != 1:
                    continue
                if placements_out[jj, 0] != n_bins:
                    continue
                if sku_id_per_box[bps_order[jj]] != my_sku:
                    continue
                px = best_x_n + kk2 * dx
                py = best_y_n + ll2 * dy
                pz = best_z_n + mm2 * dz
                if (placements_out[jj, 2] != px
                        or placements_out[jj, 3] != py
                        or placements_out[jj, 4] != pz):
                    continue
                _apply_cog_contribution_njit(
                    px, py, dx, dy, box_weight, n_bins,
                    pallet_sum_xw, pallet_sum_yw)
                applied2 += 1
                kk2 += 1
                if kk2 >= k:
                    kk2 = 0
                    ll2 += 1
                    if ll2 >= l:
                        ll2 = 0
                        mm2 += 1
        new_count = commit_ems_njit(
            bin_emss[n_bins], bin_ems_count[n_bins],
            best_x_n, best_y_n, best_z_n,
            best_x_n + k * dx, best_y_n + l * dy, best_z_n + m * dz,
            scratch,
        )
        for j in range(new_count):
            bin_emss[n_bins, j, 0, 0] = scratch[j, 0, 0]
            bin_emss[n_bins, j, 0, 1] = scratch[j, 0, 1]
            bin_emss[n_bins, j, 0, 2] = scratch[j, 0, 2]
            bin_emss[n_bins, j, 1, 0] = scratch[j, 1, 0]
            bin_emss[n_bins, j, 1, 1] = scratch[j, 1, 1]
            bin_emss[n_bins, j, 1, 2] = scratch[j, 1, 2]
        bin_ems_count[n_bins] = new_count
        n_bins += 1

    return n_bins


# ============================================================================
# Constraint-aware layer-build decoder (mode 3, v3.11)
# ============================================================================

@njit(cache=True, fastmath=True)
def decode_layer_njit_cstr(
    bps_order: np.ndarray,
    n_rots_per_box: np.ndarray,
    dims_all: np.ndarray,
    L: int, W: int, H: int,
    max_pallets: int,
    placements_out: np.ndarray,
    weights: np.ndarray,
    mlot: np.ndarray,
    rfs: np.ndarray,
    pallet_max_weight: float,
    support_ratio: float,
    require_centroid: int,
    cog_x_min: float, cog_x_max: float,
    cog_y_min: float, cog_y_max: float,
    cog_min_load_frac: float,
    cog_active: int,
    pallet_l: int = 0,
    pallet_w: int = 0,
    transitive: int = 0,
) -> int:
    """Constraint-aware Bischoff-Ratcliff layer-build (mode 3 + v3.11/v3.12).

    Same slab structure as decode_layer_njit but per-placement checks:
      - Pallet weight cap
      - Box.max_load_on_top via _check_load_on_top_njit
      - support_ratio for non-floor placements
    """
    n = bps_order.shape[0]
    MAX_BINS = max_pallets if max_pallets > 0 else 32
    bin_emss = np.zeros((MAX_BINS, MAX_EMS, 2, 3), dtype=np.int64)
    bin_ems_count = np.zeros(MAX_BINS, dtype=np.int64)
    bin_slab_min_x = np.zeros(MAX_BINS, dtype=np.int64)
    bin_slab_max_x = np.zeros(MAX_BINS, dtype=np.int64)
    n_bins = 0
    scratch = np.zeros((MAX_EMS, 2, 3), dtype=np.int64)
    pallet_weights = np.zeros(MAX_BINS, dtype=np.float64)
    placement_top_loads = np.zeros(n, dtype=np.float64)
    tl_inc = np.zeros(n, dtype=np.float64)
    tl_touched = np.zeros(n, dtype=np.int64)
    # Round 3 (F20): all-zero overlay for the shared check signatures
    # (this decoder commits per box — no multi-commit staleness).
    blk_inc = np.zeros(n, dtype=np.float64)
    pallet_sum_xw = np.zeros(MAX_BINS, dtype=np.float64)
    pallet_sum_yw = np.zeros(MAX_BINS, dtype=np.float64)

    for i in range(n):
        box_idx = bps_order[i]
        n_rots = n_rots_per_box[box_idx]
        cand_weight = weights[box_idx]
        placed = False

        for b in range(n_bins):
            # Pre-check pallet weight cap.
            if pallet_weights[b] + cand_weight > pallet_max_weight + 1e-6:
                continue
            # Phase 1: try current slab.
            bin_best_rot = -1
            bin_best_yz = -1
            bin_best_x = 0
            bin_best_y = 0
            bin_best_z = 0
            for r in range(n_rots):
                dx = dims_all[box_idx, r, 0]
                dy = dims_all[box_idx, r, 1]
                dz = dims_all[box_idx, r, 2]
                idx, x, y, z = find_best_in_slab_njit(
                    bin_emss[b], bin_ems_count[b], dx, dy, dz, L, W, H,
                    bin_slab_min_x[b], bin_slab_max_x[b])
                if idx < 0:
                    continue
                yz = (W - y - dy) * (W - y - dy) + (H - z - dz) * (H - z - dz)
                if yz > bin_best_yz:
                    bin_best_yz = yz
                    bin_best_rot = r
                    bin_best_x, bin_best_y, bin_best_z = x, y, z
            if bin_best_rot >= 0:
                dx = dims_all[box_idx, bin_best_rot, 0]
                dy = dims_all[box_idx, bin_best_rot, 1]
                dz = dims_all[box_idx, bin_best_rot, 2]
                cog_ok = True
                if cog_active != 0:
                    cog_ok = _check_cog_envelope_njit(
                        bin_best_x, bin_best_y, dx, dy,
                        cand_weight, b,
                        pallet_weights, pallet_sum_xw, pallet_sum_yw,
                        pallet_max_weight,
                        cog_x_min, cog_x_max, cog_y_min, cog_y_max,
                        cog_min_load_frac)
                # F21 (round 3): load inherited by under-filling beneath
                # already-placed boxes; see _rider_inflow_njit.
                inherited = _rider_inflow_njit(
                    placements_out, dims_all, bps_order, weights,
                    placement_top_loads, n,
                    b, bin_best_x, bin_best_y, bin_best_z,
                    dx, dy, dz, transitive)
                if cog_ok and inherited > 0.0:
                    lim_c = mlot[box_idx]
                    eps_c = 1e-9 * lim_c
                    if eps_c < 1e-6:
                        eps_c = 1e-6
                    if inherited > lim_c + eps_c:
                        cog_ok = False
                if cog_ok and transitive != 0:
                    cog_ok = _check_load_transitive_njit(
                        placements_out, dims_all, bps_order, mlot,
                        placement_top_loads, n,
                        b, bin_best_x, bin_best_y, bin_best_z,
                        dx, dy, dz, cand_weight + inherited,
                        tl_inc, tl_touched, blk_inc)
                if cog_ok and _check_load_on_top_njit(
                        placements_out, dims_all, bps_order, mlot,
                        placement_top_loads, n,
                        b, bin_best_x, bin_best_y, bin_best_z,
                        dx, dy, dz, cand_weight, support_ratio,
                        require_centroid, rfs[box_idx],
                        pallet_l, pallet_w, blk_inc):
                    new_count = commit_ems_njit(
                        bin_emss[b], bin_ems_count[b],
                        bin_best_x, bin_best_y, bin_best_z,
                        bin_best_x + dx, bin_best_y + dy, bin_best_z + dz,
                        scratch,
                    )
                    for j in range(new_count):
                        bin_emss[b, j, 0, 0] = scratch[j, 0, 0]
                        bin_emss[b, j, 0, 1] = scratch[j, 0, 1]
                        bin_emss[b, j, 0, 2] = scratch[j, 0, 2]
                        bin_emss[b, j, 1, 0] = scratch[j, 1, 0]
                        bin_emss[b, j, 1, 1] = scratch[j, 1, 1]
                        bin_emss[b, j, 1, 2] = scratch[j, 1, 2]
                    bin_ems_count[b] = new_count
                    placements_out[i, 0] = b
                    placements_out[i, 1] = bin_best_rot
                    placements_out[i, 2] = bin_best_x
                    placements_out[i, 3] = bin_best_y
                    placements_out[i, 4] = bin_best_z
                    placements_out[i, 5] = 1
                    pallet_weights[b] += cand_weight
                    if inherited > 0.0:              # F21: book on own row
                        placement_top_loads[i] += inherited
                    if transitive != 0:
                        _apply_load_contribution_transitive_njit(
                            placements_out, dims_all, bps_order,
                            placement_top_loads, n,
                            b, bin_best_x, bin_best_y, bin_best_z,
                            dx, dy, dz, cand_weight + inherited,
                            tl_inc, tl_touched)
                    else:
                        _apply_load_contribution_njit(
                            placements_out, dims_all, bps_order,
                            placement_top_loads, n,
                            b, bin_best_x, bin_best_y, bin_best_z,
                            dx, dy, dz, cand_weight)
                    _apply_cog_contribution_njit(
                        bin_best_x, bin_best_y, dx, dy,
                        cand_weight, b, pallet_sum_xw, pallet_sum_yw)
                    placed = True
                    break

            # Phase 2: open a new slab at slab_max_x in this bin.
            # Try all rotations starting at x = slab_max_x.
            new_slab_start = bin_slab_max_x[b]
            new_best_rot = -1
            new_best_yz = -1
            new_best_y = 0
            new_best_z = 0
            new_best_dx = 0
            for r in range(n_rots):
                dx = dims_all[box_idx, r, 0]
                dy = dims_all[box_idx, r, 1]
                dz = dims_all[box_idx, r, 2]
                if new_slab_start + dx > L:
                    continue
                # Find an EMS that starts at x >= new_slab_start with sufficient
                # extent. Slab placement is at (new_slab_start, smallest valid
                # y, smallest valid z) for any matching EMS — pick the YZ-DFTRC
                # best.
                idx, x, y, z = find_best_in_slab_njit(
                    bin_emss[b], bin_ems_count[b], dx, dy, dz, L, W, H,
                    new_slab_start, new_slab_start + dx)
                if idx < 0:
                    continue
                yz = (W - y - dy) * (W - y - dy) + (H - z - dz) * (H - z - dz)
                if yz > new_best_yz:
                    new_best_yz = yz
                    new_best_rot = r
                    new_best_y, new_best_z = y, z
                    new_best_dx = dx
            if new_best_rot >= 0:
                r = new_best_rot
                dx = dims_all[box_idx, r, 0]
                dy = dims_all[box_idx, r, 1]
                dz = dims_all[box_idx, r, 2]
                cog_ok = True
                if cog_active != 0:
                    cog_ok = _check_cog_envelope_njit(
                        new_slab_start, new_best_y, dx, dy,
                        cand_weight, b,
                        pallet_weights, pallet_sum_xw, pallet_sum_yw,
                        pallet_max_weight,
                        cog_x_min, cog_x_max, cog_y_min, cog_y_max,
                        cog_min_load_frac)
                # F21 (round 3): under-fill inherited load; see Phase 1.
                inherited = _rider_inflow_njit(
                    placements_out, dims_all, bps_order, weights,
                    placement_top_loads, n,
                    b, new_slab_start, new_best_y, new_best_z,
                    dx, dy, dz, transitive)
                if cog_ok and inherited > 0.0:
                    lim_c = mlot[box_idx]
                    eps_c = 1e-9 * lim_c
                    if eps_c < 1e-6:
                        eps_c = 1e-6
                    if inherited > lim_c + eps_c:
                        cog_ok = False
                if cog_ok and transitive != 0:
                    cog_ok = _check_load_transitive_njit(
                        placements_out, dims_all, bps_order, mlot,
                        placement_top_loads, n,
                        b, new_slab_start, new_best_y, new_best_z,
                        dx, dy, dz, cand_weight + inherited,
                        tl_inc, tl_touched, blk_inc)
                if cog_ok and _check_load_on_top_njit(
                        placements_out, dims_all, bps_order, mlot,
                        placement_top_loads, n,
                        b, new_slab_start, new_best_y, new_best_z,
                        dx, dy, dz, cand_weight, support_ratio,
                        require_centroid, rfs[box_idx],
                        pallet_l, pallet_w, blk_inc):
                    new_count = commit_ems_njit(
                        bin_emss[b], bin_ems_count[b],
                        new_slab_start, new_best_y, new_best_z,
                        new_slab_start + dx, new_best_y + dy, new_best_z + dz,
                        scratch,
                    )
                    for j in range(new_count):
                        bin_emss[b, j, 0, 0] = scratch[j, 0, 0]
                        bin_emss[b, j, 0, 1] = scratch[j, 0, 1]
                        bin_emss[b, j, 0, 2] = scratch[j, 0, 2]
                        bin_emss[b, j, 1, 0] = scratch[j, 1, 0]
                        bin_emss[b, j, 1, 1] = scratch[j, 1, 1]
                        bin_emss[b, j, 1, 2] = scratch[j, 1, 2]
                    bin_ems_count[b] = new_count
                    placements_out[i, 0] = b
                    placements_out[i, 1] = r
                    placements_out[i, 2] = new_slab_start
                    placements_out[i, 3] = new_best_y
                    placements_out[i, 4] = new_best_z
                    placements_out[i, 5] = 1
                    pallet_weights[b] += cand_weight
                    bin_slab_min_x[b] = new_slab_start
                    bin_slab_max_x[b] = new_slab_start + dx
                    if inherited > 0.0:              # F21: book on own row
                        placement_top_loads[i] += inherited
                    if transitive != 0:
                        _apply_load_contribution_transitive_njit(
                            placements_out, dims_all, bps_order,
                            placement_top_loads, n,
                            b, new_slab_start, new_best_y, new_best_z,
                            dx, dy, dz, cand_weight + inherited,
                            tl_inc, tl_touched)
                    else:
                        _apply_load_contribution_njit(
                            placements_out, dims_all, bps_order,
                            placement_top_loads, n,
                            b, new_slab_start, new_best_y, new_best_z,
                            dx, dy, dz, cand_weight)
                    _apply_cog_contribution_njit(
                        new_slab_start, new_best_y, dx, dy,
                        cand_weight, b, pallet_sum_xw, pallet_sum_yw)
                    placed = True
                    break

        if placed:
            continue

        # No fit in any existing bin — open a new bin (first box is seed).
        if n_bins >= MAX_BINS:
            placements_out[i, 5] = 0
            continue
        if cand_weight > pallet_max_weight + 1e-6:
            placements_out[i, 5] = 0
            continue
        bin_emss[n_bins, 0, 0, 0] = 0
        bin_emss[n_bins, 0, 0, 1] = 0
        bin_emss[n_bins, 0, 0, 2] = 0
        bin_emss[n_bins, 0, 1, 0] = L
        bin_emss[n_bins, 0, 1, 1] = W
        bin_emss[n_bins, 0, 1, 2] = H
        bin_ems_count[n_bins] = 1
        seed_rot = -1
        seed_yz = -1
        seed_y = 0
        seed_z = 0
        for r in range(n_rots):
            dx = dims_all[box_idx, r, 0]
            dy = dims_all[box_idx, r, 1]
            dz = dims_all[box_idx, r, 2]
            if dx > L or dy > W or dz > H:
                continue
            yz = (W - dy) * (W - dy) + (H - dz) * (H - dz)
            if yz > seed_yz:
                seed_yz = yz
                seed_rot = r
                seed_y = 0
                seed_z = 0
        if seed_rot < 0:
            placements_out[i, 5] = 0
            continue
        dx = dims_all[box_idx, seed_rot, 0]
        dy = dims_all[box_idx, seed_rot, 1]
        dz = dims_all[box_idx, seed_rot, 2]
        # Deck-contact check (F17): the seed sits at the origin, so this
        # only bites when the box itself outspans the raw deck under
        # overhang-inflated L/W.
        if pallet_l > 0:
            if not _check_load_on_top_njit(
                    placements_out, dims_all, bps_order, mlot,
                    placement_top_loads, n,
                    n_bins, 0, seed_y, seed_z,
                    dx, dy, dz, cand_weight, support_ratio,
                    require_centroid, rfs[box_idx],
                    pallet_l, pallet_w, blk_inc):
                placements_out[i, 5] = 0
                continue
        new_count = commit_ems_njit(
            bin_emss[n_bins], bin_ems_count[n_bins],
            0, seed_y, seed_z, dx, seed_y + dy, seed_z + dz, scratch,
        )
        for j in range(new_count):
            bin_emss[n_bins, j, 0, 0] = scratch[j, 0, 0]
            bin_emss[n_bins, j, 0, 1] = scratch[j, 0, 1]
            bin_emss[n_bins, j, 0, 2] = scratch[j, 0, 2]
            bin_emss[n_bins, j, 1, 0] = scratch[j, 1, 0]
            bin_emss[n_bins, j, 1, 1] = scratch[j, 1, 1]
            bin_emss[n_bins, j, 1, 2] = scratch[j, 1, 2]
        bin_ems_count[n_bins] = new_count
        placements_out[i, 0] = n_bins
        placements_out[i, 1] = seed_rot
        placements_out[i, 2] = 0
        placements_out[i, 3] = seed_y
        placements_out[i, 4] = seed_z
        placements_out[i, 5] = 1
        bin_slab_min_x[n_bins] = 0
        bin_slab_max_x[n_bins] = dx
        pallet_weights[n_bins] += cand_weight
        # First box at z=0 — no top-load update needed.
        _apply_cog_contribution_njit(
            0, seed_y, dx, dy, cand_weight, n_bins,
            pallet_sum_xw, pallet_sum_yw)
        n_bins += 1

    return n_bins


# ============================================================================
# Python wrappers
# ============================================================================

_JIT_WARMED = False

