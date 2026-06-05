"""
pallet_packer._brkga_core.jit_decoders_geom — pure geometric JIT decoders.

  decode_layer_njit                  mode 3 — Bischoff-Ratcliff layers
  find_best_block_at_pos_njit        helper for mode 4 block extension
  decode_blocks_njit_mode            mode 4 — dynamic composite blocks
  decode_precomputed_blocks_njit_mode mode 5 — Bischoff 2002 top-K blocks
  decode_njit_mode                   modes 0/1/2 — DFTRC/wall/corner

These ignore physical constraints (weight, fragility, etc.) and run on
pure-geometric problems (BR1-7 academic benchmark). The constraint-aware
variants live in jit_decoders_cstr.py.
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


@njit(cache=True, fastmath=True)
def decode_layer_njit(
    bps_order: np.ndarray,
    n_rots_per_box: np.ndarray,
    dims_all: np.ndarray,
    L: int, W: int, H: int,
    max_pallets: int,
    placements_out: np.ndarray,
) -> int:
    """Bischoff-Ratcliff layer-building decoder (JIT).

    For each bin we maintain a current slab [slab_min_x, slab_max_x] along X.
    The slab's depth is determined by the FIRST box placed in it (the
    "seed"). Subsequent boxes in BPS order:

      Phase 1: try to place in current slab via find_best_in_slab_njit.
               Scoring is DFTRC on YZ only.
      Phase 2: if no fit, open a new slab at slab_max_x (this box becomes
               the new seed, slab_max_x advances by box.dx).
      Phase 3: if no fit in any existing bin, open a new bin.

    This produces explicit slab-structured packings ideal for homogeneous
    loads (BR1, BR3) — the literature's key advantage over greedy DFTRC.
    """
    n = bps_order.shape[0]
    MAX_BINS = max_pallets if max_pallets > 0 else 32
    bin_emss = np.zeros((MAX_BINS, MAX_EMS, 2, 3), dtype=np.int64)
    bin_ems_count = np.zeros(MAX_BINS, dtype=np.int64)
    bin_slab_min_x = np.zeros(MAX_BINS, dtype=np.int64)
    bin_slab_max_x = np.zeros(MAX_BINS, dtype=np.int64)
    n_bins = 0
    scratch = np.zeros((MAX_EMS, 2, 3), dtype=np.int64)

    for i in range(n):
        box_idx = bps_order[i]
        n_rots = n_rots_per_box[box_idx]
        placed = False

        # Phase 1: try existing slab in each bin
        best_bin = -1
        best_rot = -1
        best_x = 0
        best_y = 0
        best_z = 0
        best_yz = -1
        for b in range(n_bins):
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
                if yz > best_yz:
                    best_yz = yz
                    best_bin = b
                    best_rot = r
                    best_x, best_y, best_z = x, y, z
            if best_bin >= 0:
                break

        if best_bin >= 0:
            r = best_rot
            dx = dims_all[box_idx, r, 0]
            dy = dims_all[box_idx, r, 1]
            dz = dims_all[box_idx, r, 2]
            new_count = commit_ems_njit(
                bin_emss[best_bin], bin_ems_count[best_bin],
                best_x, best_y, best_z,
                best_x + dx, best_y + dy, best_z + dz,
                scratch,
            )
            for j in range(new_count):
                bin_emss[best_bin, j, 0, 0] = scratch[j, 0, 0]
                bin_emss[best_bin, j, 0, 1] = scratch[j, 0, 1]
                bin_emss[best_bin, j, 0, 2] = scratch[j, 0, 2]
                bin_emss[best_bin, j, 1, 0] = scratch[j, 1, 0]
                bin_emss[best_bin, j, 1, 1] = scratch[j, 1, 1]
                bin_emss[best_bin, j, 1, 2] = scratch[j, 1, 2]
            bin_ems_count[best_bin] = new_count
            placements_out[i, 0] = best_bin
            placements_out[i, 1] = best_rot
            placements_out[i, 2] = best_x
            placements_out[i, 3] = best_y
            placements_out[i, 4] = best_z
            placements_out[i, 5] = 1
            continue

        # Phase 2: open new slab in existing bin (this box becomes the seed)
        for b in range(n_bins):
            slab_start = bin_slab_max_x[b]
            if slab_start >= L:
                continue
            for r in range(n_rots):
                dx = dims_all[box_idx, r, 0]
                dy = dims_all[box_idx, r, 1]
                dz = dims_all[box_idx, r, 2]
                if slab_start + dx > L:
                    continue
                idx, x, y, z = find_best_in_slab_njit(
                    bin_emss[b], bin_ems_count[b], dx, dy, dz, L, W, H,
                    slab_start, slab_start + dx)
                if idx < 0:
                    continue
                # Commit: box becomes seed of new slab
                new_count = commit_ems_njit(
                    bin_emss[b], bin_ems_count[b],
                    x, y, z, x + dx, y + dy, z + dz, scratch,
                )
                for j in range(new_count):
                    bin_emss[b, j, 0, 0] = scratch[j, 0, 0]
                    bin_emss[b, j, 0, 1] = scratch[j, 0, 1]
                    bin_emss[b, j, 0, 2] = scratch[j, 0, 2]
                    bin_emss[b, j, 1, 0] = scratch[j, 1, 0]
                    bin_emss[b, j, 1, 1] = scratch[j, 1, 1]
                    bin_emss[b, j, 1, 2] = scratch[j, 1, 2]
                bin_ems_count[b] = new_count
                bin_slab_min_x[b] = slab_start
                bin_slab_max_x[b] = slab_start + dx
                placements_out[i, 0] = b
                placements_out[i, 1] = r
                placements_out[i, 2] = x
                placements_out[i, 3] = y
                placements_out[i, 4] = z
                placements_out[i, 5] = 1
                placed = True
                break
            if placed:
                break
        if placed:
            continue

        # Phase 3: open new bin
        if n_bins >= MAX_BINS:
            placements_out[i, 5] = 0
            continue
        bin_emss[n_bins, 0, 0, 0] = 0
        bin_emss[n_bins, 0, 0, 1] = 0
        bin_emss[n_bins, 0, 0, 2] = 0
        bin_emss[n_bins, 0, 1, 0] = L
        bin_emss[n_bins, 0, 1, 1] = W
        bin_emss[n_bins, 0, 1, 2] = H
        bin_ems_count[n_bins] = 1
        bin_slab_min_x[n_bins] = 0
        bin_slab_max_x[n_bins] = 0
        # First-fit rotation
        best_rot_n = -1
        for r in range(n_rots):
            dx = dims_all[box_idx, r, 0]
            dy = dims_all[box_idx, r, 1]
            dz = dims_all[box_idx, r, 2]
            if dx <= L and dy <= W and dz <= H:
                best_rot_n = r
                break
        if best_rot_n < 0:
            placements_out[i, 5] = 0
            continue
        r = best_rot_n
        dx = dims_all[box_idx, r, 0]
        dy = dims_all[box_idx, r, 1]
        dz = dims_all[box_idx, r, 2]
        new_count = commit_ems_njit(
            bin_emss[n_bins], bin_ems_count[n_bins],
            0, 0, 0, dx, dy, dz, scratch,
        )
        for j in range(new_count):
            bin_emss[n_bins, j, 0, 0] = scratch[j, 0, 0]
            bin_emss[n_bins, j, 0, 1] = scratch[j, 0, 1]
            bin_emss[n_bins, j, 0, 2] = scratch[j, 0, 2]
            bin_emss[n_bins, j, 1, 0] = scratch[j, 1, 0]
            bin_emss[n_bins, j, 1, 1] = scratch[j, 1, 1]
            bin_emss[n_bins, j, 1, 2] = scratch[j, 1, 2]
        bin_ems_count[n_bins] = new_count
        bin_slab_min_x[n_bins] = 0
        bin_slab_max_x[n_bins] = dx
        placements_out[i, 0] = n_bins
        placements_out[i, 1] = best_rot_n
        placements_out[i, 2] = 0
        placements_out[i, 3] = 0
        placements_out[i, 4] = 0
        placements_out[i, 5] = 1
        n_bins += 1
    return n_bins


@njit(cache=True, fastmath=True)
def find_best_block_at_pos_njit(
    emss: np.ndarray, n_ems: int,
    x: int, y: int, z: int,
    dx: int, dy: int, dz: int,
    max_count: int,
) -> tuple:
    """Find largest (k, l, m) composite block at position (x, y, z) with
    box dims (dx, dy, dz), constrained by:
      - block region must fit in some EMS that already contains the single box
      - k * l * m <= max_count (available same-SKU boxes)

    Returns (best_k, best_l, best_m). Always returns >= (1, 1, 1).

    The algorithm exploits the EMS invariant: any free region [x..x+kdx,
    y..y+ldy, z..z+mdz] must be entirely contained in some single EMS that
    also contains the box at (x, y, z). So we iterate over candidate EMSs
    and within each find the max (k, l, m) that fits AND respects max_count.
    """
    # Floor-first block shape: prefer the largest floor FOOTPRINT (k*l), then
    # the tallest stack (m) that fits within that footprint and the available
    # same-SKU count. Building complete floor layers before stacking keeps
    # under-filled pallets flat instead of corner-towered. This is box-count-
    # neutral in the dense regime: max_m does not depend on (k, l), so the full
    # footprint reaches the same maximum count as any tower whenever the box
    # count is not the binding limit. It only trades count for floor-spread
    # when boxes are scarce relative to the pallet — exactly the under-filled
    # case the user hit (10 boxes piling into a 2x5 tower on a tall pallet).
    best_k = 1
    best_l = 1
    best_m = 1
    best_fp = 1       # footprint (k*l) of the current best
    best_count = 1    # k*l*m of the current best (tiebreak within a footprint)
    for ei in range(n_ems):
        ex_min = emss[ei, 0, 0]
        ey_min = emss[ei, 0, 1]
        ez_min = emss[ei, 0, 2]
        ex_max = emss[ei, 1, 0]
        ey_max = emss[ei, 1, 1]
        ez_max = emss[ei, 1, 2]
        # Skip EMS that doesn't contain the single-box placement
        if ex_min > x or ey_min > y or ez_min > z:
            continue
        if ex_max < x + dx or ey_max < y + dy or ez_max < z + dz:
            continue
        # Max block dims within this EMS, capped by available count
        max_k = min((ex_max - x) // dx, max_count)
        max_l = (ey_max - y) // dy
        max_m = (ez_max - z) // dz
        if max_k < 1 or max_l < 1 or max_m < 1:
            continue
        for k in range(1, max_k + 1):
            if k > max_count:
                break
            for l in range(1, max_l + 1):
                fp = k * l
                if fp > max_count:
                    break
                # Tallest stack that fits this footprint without exceeding the
                # EMS height (max_m) or the available same-SKU count.
                m = max_m
                if fp * m > max_count:
                    m = max_count // fp
                count = fp * m
                if fp > best_fp or (fp == best_fp and count > best_count):
                    best_fp = fp
                    best_count = count
                    best_k = k
                    best_l = l
                    best_m = m
    return best_k, best_l, best_m


@njit(cache=True, fastmath=True)
def decode_blocks_njit_mode(
    bps_order: np.ndarray,
    n_rots_per_box: np.ndarray,
    dims_all: np.ndarray,
    sku_id_per_box: np.ndarray,
    L: int, W: int, H: int,
    max_pallets: int,
    placements_out: np.ndarray,
    n_skus: int,
) -> int:
    """Decode with composite block extension (Bischoff-Ratcliff 1995).

    Two-phase placement:
      Phase 1: Find DFTRC placement for single box (across rotations).
               This gives a position favored by DFTRC's "tuck into corner"
               heuristic, which leaves the pallet open for future blocks.
      Phase 2: At the chosen position+rotation, extend to a (k, l, m) block
               of same-SKU unplaced boxes (find_best_block_at_pos_njit
               returns max k*l*m fitting in any EMS containing the position).

    Why DFTRC-then-extend (not block-aware DFTRC): block-aware placement
    is too greedy — it picks positions maximizing one block's count, but
    those positions often waste space for subsequent blocks. DFTRC keeps
    the pallet "wall-like" globally, which works much better in practice.

    Block extension makes huge gains on homogeneous loads (BR1, BR3).
    For heterogeneous loads (BR7) it degenerates to k=l=m=1 with no
    measurable overhead.
    """
    n = bps_order.shape[0]
    MAX_BINS = max_pallets if max_pallets > 0 else 32
    bin_emss = np.zeros((MAX_BINS, MAX_EMS, 2, 3), dtype=np.int64)
    bin_ems_count = np.zeros(MAX_BINS, dtype=np.int64)
    n_bins = 0
    scratch = np.zeros((MAX_EMS, 2, 3), dtype=np.int64)
    placed = np.zeros(n, dtype=np.int64)  # 1 if box already placed (via block or single)

    # Count remaining boxes per SKU
    sku_remaining = np.zeros(n_skus, dtype=np.int64)
    for i in range(n):
        box_idx = bps_order[i]
        sku_remaining[sku_id_per_box[box_idx]] += 1

    for i in range(n):
        if placed[i] == 1:
            continue
        box_idx = bps_order[i]
        my_sku = sku_id_per_box[box_idx]
        n_rots = n_rots_per_box[box_idx]
        max_count = sku_remaining[my_sku]

        # --- Phase 1: DFTRC placement (existing bins) ---
        best_bin = -1
        best_rot = -1
        best_x = 0
        best_y = 0
        best_z = 0
        best_score = -1
        for b in range(n_bins):
            for r in range(n_rots):
                dx = dims_all[box_idx, r, 0]
                dy = dims_all[box_idx, r, 1]
                dz = dims_all[box_idx, r, 2]
                idx, x, y, z = find_best_dftrc_njit(
                    bin_emss[b], bin_ems_count[b], dx, dy, dz, L, W, H)
                if idx < 0:
                    continue
                score = ((L - x - dx) * (L - x - dx)
                         + (W - y - dy) * (W - y - dy)
                         + (H - z - dz) * (H - z - dz))
                if score > best_score:
                    best_score = score
                    best_bin = b
                    best_rot = r
                    best_x, best_y, best_z = x, y, z
            if best_bin >= 0:
                break

        # --- Phase 2: open new bin if no fit ---
        if best_bin < 0:
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
                score = ((L - x - dx) * (L - x - dx)
                         + (W - y - dy) * (W - y - dy)
                         + (H - z - dz) * (H - z - dz))
                if score > best_score_n:
                    best_score_n = score
                    best_rot_n = r
                    best_x_n, best_y_n, best_z_n = x, y, z
            if best_rot_n < 0:
                placements_out[i, 5] = 0
                placed[i] = 1
                sku_remaining[my_sku] -= 1
                continue
            best_bin = n_bins
            best_rot = best_rot_n
            best_x, best_y, best_z = best_x_n, best_y_n, best_z_n
            n_bins += 1

        # --- Phase 3: Block extension at chosen DFTRC position ---
        dx = dims_all[box_idx, best_rot, 0]
        dy = dims_all[box_idx, best_rot, 1]
        dz = dims_all[box_idx, best_rot, 2]
        k, l, m = find_best_block_at_pos_njit(
            bin_emss[best_bin], bin_ems_count[best_bin],
            best_x, best_y, best_z, dx, dy, dz, max_count,
        )
        block_count = k * l * m

        # --- Phase 4: assign positions to next 'block_count' unplaced same-SKU boxes ---
        placed_so_far = 0
        kk = 0
        ll = 0
        mm = 0
        # First box is the current i itself
        for j in range(i, n):
            if placed_so_far >= block_count:
                break
            if placed[j] == 1:
                continue
            if sku_id_per_box[bps_order[j]] != my_sku:
                continue
            # Compute position
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
            placed_so_far += 1
            # Advance position within block (X fastest, then Y, then Z)
            kk += 1
            if kk >= k:
                kk = 0
                ll += 1
                if ll >= l:
                    ll = 0
                    mm += 1
        sku_remaining[my_sku] -= placed_so_far

        # --- Phase 5: single EMS commit for the entire block region ---
        new_count = commit_ems_njit(
            bin_emss[best_bin], bin_ems_count[best_bin],
            best_x, best_y, best_z,
            best_x + k * dx, best_y + l * dy, best_z + m * dz,
            scratch,
        )
        for j in range(new_count):
            bin_emss[best_bin, j, 0, 0] = scratch[j, 0, 0]
            bin_emss[best_bin, j, 0, 1] = scratch[j, 0, 1]
            bin_emss[best_bin, j, 0, 2] = scratch[j, 0, 2]
            bin_emss[best_bin, j, 1, 0] = scratch[j, 1, 0]
            bin_emss[best_bin, j, 1, 1] = scratch[j, 1, 1]
            bin_emss[best_bin, j, 1, 2] = scratch[j, 1, 2]
        bin_ems_count[best_bin] = new_count

    return n_bins


@njit(cache=True, fastmath=True)
def decode_precomputed_blocks_njit_mode(
    bps_order: np.ndarray,
    n_rots_per_box: np.ndarray,
    dims_all: np.ndarray,
    sku_id_per_box: np.ndarray,
    sku_best_block: np.ndarray,  # (n_skus, 4) - (k, l, m, rot)
    L: int, W: int, H: int,
    max_pallets: int,
    placements_out: np.ndarray,
    n_skus: int,
) -> int:
    """Mode 5: pre-computed block decoder (Bischoff 2002 style).

    For each box in BPS order:
      1. Get pre-computed best (k, l, m, rot) for box's SKU
      2. If enough same-SKU boxes unplaced, try placing the full pre-computed
         block at DFTRC position for BLOCK dimensions
      3. If fits: place block, mark constituent boxes
      4. If block doesn't fit anywhere or not enough boxes: fall back to
         single-box DFTRC + dynamic block extension (like mode 4)

    Key difference from mode 4: prefer a SPECIFIC pre-computed block size
    rather than max-greedy. This sometimes packs better because the
    pre-computed block was chosen for the WHOLE pallet capacity rather
    than the current EMS shape.
    """
    n = bps_order.shape[0]
    MAX_BINS = max_pallets if max_pallets > 0 else 32
    bin_emss = np.zeros((MAX_BINS, MAX_EMS, 2, 3), dtype=np.int64)
    bin_ems_count = np.zeros(MAX_BINS, dtype=np.int64)
    n_bins = 0
    scratch = np.zeros((MAX_EMS, 2, 3), dtype=np.int64)
    placed = np.zeros(n, dtype=np.int64)
    sku_remaining = np.zeros(n_skus, dtype=np.int64)
    for i in range(n):
        sku_remaining[sku_id_per_box[bps_order[i]]] += 1

    for i in range(n):
        if placed[i] == 1:
            continue
        box_idx = bps_order[i]
        my_sku = sku_id_per_box[box_idx]
        max_count = sku_remaining[my_sku]
        n_rots = n_rots_per_box[box_idx]

        # --- Try pre-computed block placement ---
        k_pre = sku_best_block[my_sku, 0]
        l_pre = sku_best_block[my_sku, 1]
        m_pre = sku_best_block[my_sku, 2]
        rot_pre = sku_best_block[my_sku, 3]
        n_pre = k_pre * l_pre * m_pre

        best_bin = -1
        best_rot = -1
        best_x = 0
        best_y = 0
        best_z = 0
        best_k = 1
        best_l = 1
        best_m = 1

        if n_pre > 1 and max_count >= n_pre:
            # Try placing the full pre-computed block
            dx_pre = dims_all[box_idx, rot_pre, 0]
            dy_pre = dims_all[box_idx, rot_pre, 1]
            dz_pre = dims_all[box_idx, rot_pre, 2]
            block_dx = k_pre * dx_pre
            block_dy = l_pre * dy_pre
            block_dz = m_pre * dz_pre
            best_score = -1
            for b in range(n_bins):
                idx, x, y, z = find_best_dftrc_njit(
                    bin_emss[b], bin_ems_count[b],
                    block_dx, block_dy, block_dz, L, W, H)
                if idx < 0:
                    continue
                score = ((L - x - block_dx) * (L - x - block_dx)
                         + (W - y - block_dy) * (W - y - block_dy)
                         + (H - z - block_dz) * (H - z - block_dz))
                if score > best_score:
                    best_score = score
                    best_bin = b
                    best_rot = rot_pre
                    best_x, best_y, best_z = x, y, z
                    best_k, best_l, best_m = k_pre, l_pre, m_pre

        # --- Fallback: mode 4 style (DFTRC single-box + dynamic block extend) ---
        if best_bin < 0:
            # Single-box DFTRC across rotations and existing bins
            single_best_score = -1
            single_best_rot = -1
            single_best_x = 0
            single_best_y = 0
            single_best_z = 0
            single_best_bin = -1
            for b in range(n_bins):
                for r in range(n_rots):
                    dx = dims_all[box_idx, r, 0]
                    dy = dims_all[box_idx, r, 1]
                    dz = dims_all[box_idx, r, 2]
                    idx, x, y, z = find_best_dftrc_njit(
                        bin_emss[b], bin_ems_count[b], dx, dy, dz, L, W, H)
                    if idx < 0:
                        continue
                    score = ((L - x - dx) * (L - x - dx)
                             + (W - y - dy) * (W - y - dy)
                             + (H - z - dz) * (H - z - dz))
                    if score > single_best_score:
                        single_best_score = score
                        single_best_bin = b
                        single_best_rot = r
                        single_best_x, single_best_y, single_best_z = x, y, z
                if single_best_bin >= 0:
                    break
            if single_best_bin >= 0:
                # Dynamic block extension from single box
                dx = dims_all[box_idx, single_best_rot, 0]
                dy = dims_all[box_idx, single_best_rot, 1]
                dz = dims_all[box_idx, single_best_rot, 2]
                ek, el, em = find_best_block_at_pos_njit(
                    bin_emss[single_best_bin], bin_ems_count[single_best_bin],
                    single_best_x, single_best_y, single_best_z,
                    dx, dy, dz, max_count)
                best_bin = single_best_bin
                best_rot = single_best_rot
                best_x, best_y, best_z = single_best_x, single_best_y, single_best_z
                best_k, best_l, best_m = ek, el, em

        # --- If still no fit, open new bin ---
        if best_bin < 0:
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
            # Try pre-computed block in new bin first
            placed_in_new = False
            if n_pre > 1 and max_count >= n_pre:
                dx_pre = dims_all[box_idx, rot_pre, 0]
                dy_pre = dims_all[box_idx, rot_pre, 1]
                dz_pre = dims_all[box_idx, rot_pre, 2]
                block_dx = k_pre * dx_pre
                block_dy = l_pre * dy_pre
                block_dz = m_pre * dz_pre
                if block_dx <= L and block_dy <= W and block_dz <= H:
                    best_bin = n_bins
                    best_rot = rot_pre
                    best_x, best_y, best_z = 0, 0, 0
                    best_k, best_l, best_m = k_pre, l_pre, m_pre
                    placed_in_new = True
                    n_bins += 1
            if not placed_in_new:
                # Single-box new bin
                single_best_rot = -1
                for r in range(n_rots):
                    dx = dims_all[box_idx, r, 0]
                    dy = dims_all[box_idx, r, 1]
                    dz = dims_all[box_idx, r, 2]
                    if dx <= L and dy <= W and dz <= H:
                        single_best_rot = r
                        break
                if single_best_rot < 0:
                    placements_out[i, 5] = 0
                    placed[i] = 1
                    sku_remaining[my_sku] -= 1
                    continue
                dx = dims_all[box_idx, single_best_rot, 0]
                dy = dims_all[box_idx, single_best_rot, 1]
                dz = dims_all[box_idx, single_best_rot, 2]
                ek, el, em = find_best_block_at_pos_njit(
                    bin_emss[n_bins], 1, 0, 0, 0, dx, dy, dz, max_count)
                best_bin = n_bins
                best_rot = single_best_rot
                best_x, best_y, best_z = 0, 0, 0
                best_k, best_l, best_m = ek, el, em
                n_bins += 1

        # --- Place the block ---
        dx = dims_all[box_idx, best_rot, 0]
        dy = dims_all[box_idx, best_rot, 1]
        dz = dims_all[box_idx, best_rot, 2]
        k, l, m = best_k, best_l, best_m
        block_count = k * l * m

        placed_so_far = 0
        kk = 0; ll = 0; mm = 0
        for j in range(i, n):
            if placed_so_far >= block_count:
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
            placed_so_far += 1
            kk += 1
            if kk >= k:
                kk = 0
                ll += 1
                if ll >= l:
                    ll = 0
                    mm += 1
        sku_remaining[my_sku] -= placed_so_far

        # Commit EMS
        new_count = commit_ems_njit(
            bin_emss[best_bin], bin_ems_count[best_bin],
            best_x, best_y, best_z,
            best_x + k * dx, best_y + l * dy, best_z + m * dz, scratch)
        for j in range(new_count):
            bin_emss[best_bin, j, 0, 0] = scratch[j, 0, 0]
            bin_emss[best_bin, j, 0, 1] = scratch[j, 0, 1]
            bin_emss[best_bin, j, 0, 2] = scratch[j, 0, 2]
            bin_emss[best_bin, j, 1, 0] = scratch[j, 1, 0]
            bin_emss[best_bin, j, 1, 1] = scratch[j, 1, 1]
            bin_emss[best_bin, j, 1, 2] = scratch[j, 1, 2]
        bin_ems_count[best_bin] = new_count

    return n_bins


@njit(cache=True, fastmath=True)
def decode_njit_mode(
    bps_order: np.ndarray,
    n_rots_per_box: np.ndarray,
    dims_all: np.ndarray,
    L: int, W: int, H: int,
    max_pallets: int,
    placements_out: np.ndarray,
    mode: int,  # 0=DFTRC, 1=wall, 2=corner
) -> int:
    """Decode chromosome with selectable placement heuristic."""
    n = bps_order.shape[0]
    MAX_BINS = max_pallets if max_pallets > 0 else 32
    bin_emss = np.zeros((MAX_BINS, MAX_EMS, 2, 3), dtype=np.int64)
    bin_ems_count = np.zeros(MAX_BINS, dtype=np.int64)
    n_bins = 0
    scratch = np.zeros((MAX_EMS, 2, 3), dtype=np.int64)

    for i in range(n):
        box_idx = bps_order[i]
        n_rots = n_rots_per_box[box_idx]
        placed = False
        best_bin = -1
        best_rot = -1
        best_x = 0
        best_y = 0
        best_z = 0
        # Score depends on mode: DFTRC maximises distance, wall minimises X,
        # corner minimises sum.
        best_score = -1
        # Sentinel for wall/corner modes. wall's score range is L*(W+H)^2
        # which is much larger than corner's L+W+H, so use a huge value.
        best_minimise = 1 << 62  # ~4.6e18, larger than any valid score
        for b in range(n_bins):
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
                    if sc > best_score:
                        best_score = sc
                        best_bin = b
                        best_rot = r
                        best_x, best_y, best_z = x, y, z
                elif mode == 1:
                    idx, x, y, z = find_best_wall_njit(
                        bin_emss[b], bin_ems_count[b], dx, dy, dz, L, W, H)
                    if idx < 0:
                        continue
                    # Compose: prefer smaller x, then larger YZ-DFTRC.
                    yz = ((W - y - dy) * (W - y - dy)
                          + (H - z - dz) * (H - z - dz))
                    # Encode as a single comparable number: x * BIG + (-yz)
                    BIG = (W + H) * (W + H) + 1
                    cand = x * BIG - yz
                    if cand < best_minimise:
                        best_minimise = cand
                        best_bin = b
                        best_rot = r
                        best_x, best_y, best_z = x, y, z
                else:  # corner
                    idx, x, y, z = find_best_corner_njit(
                        bin_emss[b], bin_ems_count[b], dx, dy, dz, L, W, H)
                    if idx < 0:
                        continue
                    cand = x + y + z
                    if cand < best_minimise:
                        best_minimise = cand
                        best_bin = b
                        best_rot = r
                        best_x, best_y, best_z = x, y, z
            if best_bin >= 0:
                break

        if best_bin >= 0:
            r = best_rot
            dx = dims_all[box_idx, r, 0]
            dy = dims_all[box_idx, r, 1]
            dz = dims_all[box_idx, r, 2]
            new_count = commit_ems_njit(
                bin_emss[best_bin], bin_ems_count[best_bin],
                best_x, best_y, best_z,
                best_x + dx, best_y + dy, best_z + dz,
                scratch,
            )
            for j in range(new_count):
                bin_emss[best_bin, j, 0, 0] = scratch[j, 0, 0]
                bin_emss[best_bin, j, 0, 1] = scratch[j, 0, 1]
                bin_emss[best_bin, j, 0, 2] = scratch[j, 0, 2]
                bin_emss[best_bin, j, 1, 0] = scratch[j, 1, 0]
                bin_emss[best_bin, j, 1, 1] = scratch[j, 1, 1]
                bin_emss[best_bin, j, 1, 2] = scratch[j, 1, 2]
            bin_ems_count[best_bin] = new_count
            placements_out[i, 0] = best_bin
            placements_out[i, 1] = best_rot
            placements_out[i, 2] = best_x
            placements_out[i, 3] = best_y
            placements_out[i, 4] = best_z
            placements_out[i, 5] = 1
            placed = True

        if not placed:
            if n_bins >= MAX_BINS:
                placements_out[i, 5] = 0
                continue
            # Open new bin.
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
            r = best_rot_n
            dx = dims_all[box_idx, r, 0]
            dy = dims_all[box_idx, r, 1]
            dz = dims_all[box_idx, r, 2]
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
            n_bins += 1
    return n_bins


# ============================================================================
# Constraint-aware decoder (v3.10): DFTRC / wall / corner with
# Pallet.max_weight and Box.max_load_on_top enforcement.
# ============================================================================
