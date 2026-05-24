"""
pallet_packer.brkga_v3_5 — v3.5: hybrid BRKGA with multi-decoder, smart
init, v2 hybridization, and local search.

Key improvements over brkga_v3_fast:
1. **Multi-decoder chromosome**: the chromosome's "decoder selector" key
   picks between DFTRC and wall-build (layer-friendly) per individual.
   GR-2013 explicitly uses multiple decoders.
2. **Smart initial population**: seeds with informed orderings (vol-desc,
   sku-grouped, max-dim) plus a v2 layer-build solution converted to a
   chromosome.
3. **Local search 2-opt**: post-BRKGA polish via random BPS swaps.
4. **Hybrid wrapper**: optionally runs v2's PalletPacker.pack() and
   returns the better of {v3.5-BRKGA, v2-layer-build}.

The wall-build decoder is a JIT-compiled variant of DFTRC that prefers
placement at the smallest X coordinate (= flush against existing walls),
with DFTRC on the cross-section (Y, Z) as tiebreaker. This produces
"brick wall" packings that excel on homogeneous loads (BR1, BR3) — the
sets where v3-fast had the largest gap to BRKGA-2013.
"""
from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np
from numba import njit

from .models import Box, Pallet, PackerConfig, Placement, Rotation
from .packer import PackResult, PalletState
from .brkga_v3_fast import (
    MAX_EMS, find_best_dftrc_njit, commit_ems_njit,
    precompute_box_dims, _fitness_pallet1,
)


def precompute_box_dims_and_sku(boxes: List[Box]) -> tuple:
    """Extended precompute: also returns sku_id_per_box for block extension.

    SKU is keyed by (length, width, height, weight, allowed-rotation set).
    Boxes with same SKU can be packed as a composite block.
    """
    n = len(boxes)
    n_rots_arr, dims_all = precompute_box_dims(boxes)
    sku_id_per_box = np.zeros(n, dtype=np.int64)
    sku_to_id: dict = {}
    for i, b in enumerate(boxes):
        # Key ignores rotation flags - same SKU = same dimensions + weight
        # plus same rotation set (so rotations match in the block)
        rot_key = tuple(sorted(r.name for r in b.allowed_rotations))
        key = (round(b.length, 6), round(b.width, 6), round(b.height, 6),
               round(b.weight, 6), rot_key)
        if key not in sku_to_id:
            sku_to_id[key] = len(sku_to_id)
        sku_id_per_box[i] = sku_to_id[key]
    return n_rots_arr, dims_all, sku_id_per_box


# ============================================================================
# JIT primitives
# ============================================================================

@njit(cache=True, fastmath=True)
def find_best_wall_njit(
    emss: np.ndarray, n_ems: int,
    dx: int, dy: int, dz: int,
    L: int, W: int, H: int,
) -> tuple:
    """Wall-build placement: minimize X first, then maximize (W-y-dy)^2 +
    (H-z-dz)^2 on the cross-section. Produces brick-wall packings ideal
    for homogeneous loads.
    """
    best_idx = -1
    best_x = L + 1
    best_yz = -1
    bx = 0
    by = 0
    bz = 0
    for i in range(n_ems):
        ex_min = emss[i, 0, 0]
        ey_min = emss[i, 0, 1]
        ez_min = emss[i, 0, 2]
        ex_max = emss[i, 1, 0]
        ey_max = emss[i, 1, 1]
        ez_max = emss[i, 1, 2]
        if (ex_max - ex_min) < dx:
            continue
        if (ey_max - ey_min) < dy:
            continue
        if (ez_max - ez_min) < dz:
            continue
        x = ex_min
        y = ey_min
        z = ez_min
        yz = (W - y - dy) * (W - y - dy) + (H - z - dz) * (H - z - dz)
        if x < best_x or (x == best_x and yz > best_yz):
            best_x = x
            best_yz = yz
            best_idx = i
            bx, by, bz = x, y, z
    return best_idx, bx, by, bz


@njit(cache=True, fastmath=True)
def find_best_corner_njit(
    emss: np.ndarray, n_ems: int,
    dx: int, dy: int, dz: int,
    L: int, W: int, H: int,
) -> tuple:
    """Anti-DFTRC: place at smallest (x+y+z) corner. Useful as 3rd decoder.
    Promotes tightly-packed corner-fill behaviour different from both
    DFTRC and wall-build.
    """
    best_idx = -1
    best_sum = 3 * (L + W + H)
    bx = 0
    by = 0
    bz = 0
    for i in range(n_ems):
        ex_min = emss[i, 0, 0]
        ey_min = emss[i, 0, 1]
        ez_min = emss[i, 0, 2]
        ex_max = emss[i, 1, 0]
        ey_max = emss[i, 1, 1]
        ez_max = emss[i, 1, 2]
        if (ex_max - ex_min) < dx:
            continue
        if (ey_max - ey_min) < dy:
            continue
        if (ez_max - ez_min) < dz:
            continue
        s = ex_min + ey_min + ez_min
        if s < best_sum:
            best_sum = s
            best_idx = i
            bx, by, bz = ex_min, ey_min, ez_min
    return best_idx, bx, by, bz


@njit(cache=True, fastmath=True)
def find_best_in_slab_njit(
    emss: np.ndarray, n_ems: int,
    dx: int, dy: int, dz: int,
    L: int, W: int, H: int,
    slab_min_x: int, slab_max_x: int,
) -> tuple:
    """Find best DFTRC placement constrained to current slab.

    Slab spans X in [slab_min_x, slab_max_x]. Placement requires
    ex_min >= slab_min_x AND ex_min + dx <= slab_max_x. Scoring is
    DFTRC on YZ only (X is determined by slab structure).
    """
    best_idx = -1
    best_yz = -1
    bx = 0
    by = 0
    bz = 0
    for i in range(n_ems):
        ex_min = emss[i, 0, 0]
        ey_min = emss[i, 0, 1]
        ez_min = emss[i, 0, 2]
        ex_max = emss[i, 1, 0]
        ey_max = emss[i, 1, 1]
        ez_max = emss[i, 1, 2]
        if (ex_max - ex_min) < dx:
            continue
        if (ey_max - ey_min) < dy:
            continue
        if (ez_max - ez_min) < dz:
            continue
        if ex_min < slab_min_x:
            continue
        if ex_min + dx > slab_max_x:
            continue
        x = ex_min
        y = ey_min
        z = ez_min
        yz = (W - y - dy) * (W - y - dy) + (H - z - dz) * (H - z - dz)
        if yz > best_yz:
            best_yz = yz
            best_idx = i
            bx, by, bz = x, y, z
    return best_idx, bx, by, bz


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
    best_k = 1
    best_l = 1
    best_m = 1
    best_count = 1
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
        # Enumerate (k, l, m). The triple loop is at most max_count
        # iterations after the count guard kicks in.
        for k in range(1, max_k + 1):
            if k > max_count:
                break
            for l in range(1, max_l + 1):
                if k * l > max_count:
                    break
                for m in range(1, max_m + 1):
                    count = k * l * m
                    if count > max_count:
                        break
                    if count > best_count:
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
# Python wrappers
# ============================================================================

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
    """
    n = len(boxes)
    if n == 0:
        return PackResult(pallets=[], unpacked=[])
    bps = chrom[:n]
    order = np.argsort(bps).astype(np.int64)

    L = int(round(pallet.length))
    W = int(round(pallet.width))
    H = int(round(pallet.height))

    placements_out = np.zeros((n, 6), dtype=np.int64)
    if mode == 3:
        n_bins = decode_layer_njit(
            order, n_rots_arr, dims_all, L, W, H,
            max_pallets, placements_out,
        )
    elif mode == 4:
        if sku_id_per_box is None:
            n_bins = decode_njit_mode(
                order, n_rots_arr, dims_all, L, W, H,
                max_pallets, placements_out, 0,
            )
        else:
            n_skus = int(sku_id_per_box.max()) + 1
            n_bins = decode_blocks_njit_mode(
                order, n_rots_arr, dims_all, sku_id_per_box,
                L, W, H, max_pallets, placements_out, n_skus,
            )
    elif mode == 5:
        if sku_id_per_box is None:
            n_bins = decode_njit_mode(
                order, n_rots_arr, dims_all, L, W, H,
                max_pallets, placements_out, 0,
            )
        else:
            n_skus = int(sku_id_per_box.max()) + 1
            # Resolve per-SKU block: prefer top-K via chromosome, else
            # fixed top-1 (sku_best_block), else fall back to mode 4
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
        n_bins = decode_njit_mode(
            order, n_rots_arr, dims_all, L, W, H,
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
) -> PackResult:
    """Use last key in chromosome to pick decoder mode (0..n_modes-1).
    Mode 5 (if n_modes >= 6) uses pre-computed top-K blocks per SKU.
    """
    n = len(boxes)
    if len(chrom) >= 2 * n + 1:
        selector = chrom[-1]
        mode = min(n_modes - 1, int(selector * n_modes))
    else:
        mode = 0
    return decode_chromosome(chrom, boxes, pallet, config,
                             n_rots_arr, dims_all, mode, max_pallets,
                             sku_id_per_box=sku_id_per_box,
                             sku_best_block=sku_best_block,
                             sku_top_k_blocks=sku_top_k_blocks)


# ============================================================================
# Smart initialization
# ============================================================================

def chromosome_from_order(order: List[int], n: int, n_rots_arr: np.ndarray,
                          rotation_choices: Optional[List[int]] = None,
                          decoder_mode: Optional[int] = None,
                          n_modes: int = 5,
                          rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Build a chromosome whose argsort yields `order`.

    `order[k]` = box index that should be the (k+1)-th processed.
    `rotation_choices[i]` = which rotation index for box i (else random).
    `decoder_mode` = if specified, encoded in last key. n_modes controls
       the discretization (default 4 = DFTRC/wall/corner/layer).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    has_selector = decoder_mode is not None
    size = 2 * n + (1 if has_selector else 0)
    c = np.empty(size, dtype=np.float64)
    for pos, box_i in enumerate(order):
        c[box_i] = (pos + 0.5) / n
    for box_i in range(n):
        n_rots = int(n_rots_arr[box_i])
        if rotation_choices is not None and rotation_choices[box_i] is not None:
            r = int(rotation_choices[box_i])
            r = max(0, min(n_rots - 1, r))
            c[n + box_i] = (r + rng.random() * 0.9 + 0.05) / n_rots
        else:
            c[n + box_i] = rng.random()
    if has_selector:
        c[-1] = (decoder_mode + 0.5) / n_modes
    return c


def make_informed_chromosomes(
    boxes: List[Box], n_rots_arr: np.ndarray,
    decoder_mode: Optional[int] = None,
    n_modes: int = 5,
    seed: int = 42,
) -> List[np.ndarray]:
    """Generate informed initial chromosomes covering different orderings."""
    n = len(boxes)
    rng = np.random.default_rng(seed)
    out: List[np.ndarray] = []
    vols = np.array([b.volume for b in boxes])
    max_dims = np.array([max(b.length, b.width, b.height) for b in boxes])
    min_dims = np.array([min(b.length, b.width, b.height) for b in boxes])

    orderings = {
        'vol_desc': np.argsort(-vols).tolist(),
        'vol_asc': np.argsort(vols).tolist(),
        'max_dim_desc': np.argsort(-max_dims).tolist(),
        'min_dim_desc': np.argsort(-min_dims).tolist(),
    }
    sku_keys = [(b.length, b.width, b.height) for b in boxes]
    sku_vol = {}
    for k, b in zip(sku_keys, boxes):
        sku_vol[k] = sku_vol.get(k, 0.0) + b.volume
    sku_order = sorted(sku_vol.keys(), key=lambda k: -sku_vol[k])
    sku_to_rank = {k: i for i, k in enumerate(sku_order)}
    sku_grouped = sorted(range(n), key=lambda i: (sku_to_rank[sku_keys[i]], -vols[i]))
    orderings['sku_grouped'] = sku_grouped
    # NEW: SKU-grouped with rotation-aligned-depth (for layer-build mode)
    # Same as sku_grouped but with rotation chosen to align min-dim with X axis
    orderings['sku_grouped_alt'] = sku_grouped[::-1]  # reverse for variety

    for name, order in orderings.items():
        if decoder_mode is None:
            modes_to_try = [None]
        else:
            modes_to_try = [decoder_mode]
        for m in modes_to_try:
            c = chromosome_from_order(order, n, n_rots_arr,
                                       decoder_mode=m, n_modes=n_modes,
                                       rng=rng)
            out.append(c)
    return out


def chromosome_from_v2_result(
    v2_result: PackResult,
    boxes: List[Box],
    n_rots_arr: np.ndarray,
    decoder_mode: Optional[int] = None,
    n_modes: int = 5,
    rng: Optional[np.random.Generator] = None,
) -> Optional[np.ndarray]:
    """Convert a v2 layer-build result into a chromosome.

    BPS ordering = order boxes were placed in v2's first pallet.
    VBO = rotation index used for each box.
    Unpacked boxes go to the end of BPS in arbitrary order.
    """
    if not v2_result or not v2_result.pallets:
        return None
    if rng is None:
        rng = np.random.default_rng(7)
    n = len(boxes)
    box_to_idx = {id(b): i for i, b in enumerate(boxes)}
    # Some v2 paths reorder boxes; fall back to (l,w,h,weight) match
    # if id doesn't match.
    def find_box_idx(target: Box) -> int:
        if id(target) in box_to_idx:
            return box_to_idx[id(target)]
        for i, b in enumerate(boxes):
            if (abs(b.length - target.length) < 1e-6
                    and abs(b.width - target.width) < 1e-6
                    and abs(b.height - target.height) < 1e-6
                    and abs(b.weight - target.weight) < 1e-6):
                return i
        return -1

    seen = set()
    placed_order: List[int] = []
    rot_choices: List[Optional[int]] = [None] * n

    for pallet_state in v2_result.pallets[:1]:  # only first pallet
        for p in pallet_state.placements:
            idx = find_box_idx(p.box)
            if idx < 0 or idx in seen:
                continue
            seen.add(idx)
            placed_order.append(idx)
            # Match rotation
            for r_idx, r in enumerate(boxes[idx].allowed_rotations):
                if r == p.rotation:
                    rot_choices[idx] = r_idx
                    break
    # Append unplaced boxes at the end
    for i in range(n):
        if i not in seen:
            placed_order.append(i)

    return chromosome_from_order(placed_order, n, n_rots_arr,
                                  rotation_choices=rot_choices,
                                  decoder_mode=decoder_mode,
                                  n_modes=n_modes, rng=rng)


# ============================================================================
# Local search
# ============================================================================

def local_search_2opt(
    best_chrom: np.ndarray,
    boxes: List[Box], pallet: Pallet, config: PackerConfig,
    n_rots_arr: np.ndarray, dims_all: np.ndarray,
    *,
    time_budget_s: float = 3.0,
    multi_decoder: bool = False,
    n_modes: int = 5,
    sku_id_per_box: Optional[np.ndarray] = None,
    max_pallets: int = 1,
    seed: int = 99,
    verbose: bool = False,
) -> Tuple[PackResult, np.ndarray, int]:
    """Position-based local search on the BPS order.

    Operates on BPS POSITIONS (not raw key values). This guarantees every
    proposed move actually changes the decoded packing, unlike random key
    swaps which often produce identical sorts.

    Operators (~equally weighted):
      - Position swap: swap boxes at two positions in BPS order.
      - Segment reverse: reverse boxes in a [i, i+k] range (k=2..7).
      - Insert: take box at position i, insert at position j.
      - Rotation flip: change rotation key for one box.
      - Decoder flip (if multi-decoder): change last selector key.
    """
    n = len(boxes)
    chrom_len = len(best_chrom)
    has_selector = multi_decoder and chrom_len > 2 * n
    current = best_chrom.copy()
    if multi_decoder:
        decoder = lambda c, b, p, cfg, na, da, max_pallets=1: decode_auto_mode(
            c, b, p, cfg, na, da, max_pallets=max_pallets, n_modes=n_modes,
            sku_id_per_box=sku_id_per_box)
    else:
        decoder = lambda c, b, p, cfg, na, da, max_pallets=1: decode_chromosome(
            c, b, p, cfg, na, da, mode=0, max_pallets=max_pallets)
    res = decoder(current, boxes, pallet, config, n_rots_arr, dims_all,
                  max_pallets=max_pallets)
    best_fit = _fitness_pallet1(res, pallet)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    moves = 0
    accepts = 0

    # Snapshot the current BPS order; we'll mutate ORDER not keys.
    order = np.argsort(current[:n]).astype(np.int64)

    def encode_order(order_arr: np.ndarray) -> np.ndarray:
        """Build new chromosome from given BPS order + current VBO + selector."""
        out = current.copy()
        # Assign small key to early position, large key to late.
        for pos in range(n):
            out[order_arr[pos]] = (pos + 0.5) / n
        return out

    while time.time() - t0 < time_budget_s:
        new_order = order.copy()
        candidate = current.copy()
        move = rng.random()
        if move < 0.30:
            # Position swap
            i = int(rng.integers(0, n))
            j = int(rng.integers(0, n))
            if i == j:
                continue
            new_order[i], new_order[j] = new_order[j], new_order[i]
            candidate = encode_order(new_order)
        elif move < 0.55:
            # Segment reverse (k=2..7)
            if n < 4:
                continue
            k = int(rng.integers(2, min(8, n)))
            i = int(rng.integers(0, n - k))
            new_order[i:i + k] = new_order[i:i + k][::-1]
            candidate = encode_order(new_order)
        elif move < 0.75:
            # Insert: take box at position i, insert at position j
            i = int(rng.integers(0, n))
            j = int(rng.integers(0, n))
            if i == j:
                continue
            box_i = int(new_order[i])
            new_order = np.delete(new_order, i)
            new_order = np.insert(new_order, j if j < i else j - 1, box_i)
            candidate = encode_order(new_order)
        elif move < 0.95:
            # Rotation flip for one box (~5% of moves)
            i = int(rng.integers(0, n))
            candidate[n + i] = rng.random()
        else:
            # Decoder mode flip
            if has_selector:
                candidate[-1] = rng.random()
            else:
                continue
        cand_res = decoder(candidate, boxes, pallet, config,
                           n_rots_arr, dims_all, max_pallets=max_pallets)
        cand_fit = _fitness_pallet1(cand_res, pallet)
        moves += 1
        if cand_fit < best_fit - 1e-9:
            best_fit = cand_fit
            current = candidate
            order = np.argsort(current[:n]).astype(np.int64)
            res = cand_res
            accepts += 1
            if verbose:
                print(f"  [LS] move {moves} accept: util={(1-best_fit)*100:.2f}%")
    if verbose:
        print(f"  [LS] {accepts}/{moves} accepted, final util={(1-best_fit)*100:.2f}%")
    return res, current, accepts


# ============================================================================
# Path relinking
# ============================================================================

def path_relinking(
    chrom_a: np.ndarray,
    chrom_b: np.ndarray,
    boxes: List[Box], pallet: Pallet, config: PackerConfig,
    n_rots_arr: np.ndarray, dims_all: np.ndarray,
    *,
    multi_decoder: bool = True,
    n_modes: int = 5,
    sku_id_per_box: Optional[np.ndarray] = None,
    max_pallets: int = 1,
    max_evals: int = 100,
    verbose: bool = False,
) -> Tuple[PackResult, np.ndarray, float]:
    """Path relinking: walk from chrom_a to chrom_b by progressively copying
    chrom_b's keys into chrom_a, evaluating each intermediate.

    Returns the best intermediate found (which may be one of the endpoints
    if no improvement).
    """
    if multi_decoder:
        decoder = lambda c, b, p, cfg, na, da, max_pallets=1: decode_auto_mode(
            c, b, p, cfg, na, da, max_pallets=max_pallets, n_modes=n_modes,
            sku_id_per_box=sku_id_per_box)
    else:
        decoder = lambda c, b, p, cfg, na, da, max_pallets=1: decode_chromosome(
            c, b, p, cfg, na, da, mode=0, max_pallets=max_pallets)
    n = len(boxes)
    res_a = decoder(chrom_a, boxes, pallet, config, n_rots_arr, dims_all,
                    max_pallets=max_pallets)
    fit_a = _fitness_pallet1(res_a, pallet)
    # Identify positions where the two chromosomes differ significantly
    diffs = np.where(np.abs(chrom_a - chrom_b) > 1e-6)[0]
    if len(diffs) == 0:
        return res_a, chrom_a, fit_a
    rng = np.random.default_rng(11)
    rng.shuffle(diffs)
    # Limit steps to max_evals
    step_size = max(1, len(diffs) // max_evals)
    best = chrom_a.copy()
    best_fit = fit_a
    best_res = res_a
    current = chrom_a.copy()
    for k in range(0, len(diffs), step_size):
        # Copy this batch of diffs from chrom_b
        end = min(k + step_size, len(diffs))
        for i in range(k, end):
            current[diffs[i]] = chrom_b[diffs[i]]
        cand_res = decoder(current, boxes, pallet, config, n_rots_arr, dims_all,
                            max_pallets=max_pallets)
        cand_fit = _fitness_pallet1(cand_res, pallet)
        if cand_fit < best_fit - 1e-9:
            best_fit = cand_fit
            best = current.copy()
            best_res = cand_res
            if verbose:
                print(f"  [PR] step {k}: util={(1-best_fit)*100:.2f}%")
    return best_res, best, best_fit


# ============================================================================
# Large Neighborhood Search (LNS) polish
# ============================================================================

def lns_polish(
    best_chrom: np.ndarray,
    boxes: List[Box], pallet: Pallet, config: PackerConfig,
    n_rots_arr: np.ndarray, dims_all: np.ndarray,
    *,
    multi_decoder: bool = True,
    n_modes: int = 5,
    sku_id_per_box: Optional[np.ndarray] = None,
    max_pallets: int = 1,
    time_budget_s: float = 3.0,
    seed: int = 77,
    verbose: bool = False,
) -> Tuple[PackResult, np.ndarray, int]:
    """Large Neighborhood Search: destroy K box BPS keys, repair via decoder.

    Each iteration:
      1. Pick K random boxes (5-20% of N)
      2. Replace their BPS keys with new random values
      3. Re-decode
      4. Keep if better

    This escapes local optima that small swaps (in local_search_2opt) can't
    reach. Complementary to BRKGA's mutation operator (which is whole-
    chromosome random) and to swap-based LS (which is small-step).
    """
    n = len(boxes)
    chrom_size = len(best_chrom)
    if multi_decoder:
        decoder = lambda c, b, p, cfg, na, da, max_pallets=1: decode_auto_mode(
            c, b, p, cfg, na, da, max_pallets=max_pallets, n_modes=n_modes,
            sku_id_per_box=sku_id_per_box)
    else:
        decoder = lambda c, b, p, cfg, na, da, max_pallets=1: decode_chromosome(
            c, b, p, cfg, na, da, mode=0, max_pallets=max_pallets)

    current = best_chrom.copy()
    res = decoder(current, boxes, pallet, config, n_rots_arr, dims_all,
                  max_pallets=max_pallets)
    best_fit = _fitness_pallet1(res, pallet)
    best_res = res
    rng = np.random.default_rng(seed)
    t0 = time.time()
    iters = 0
    accepts = 0
    while time.time() - t0 < time_budget_s:
        cand = current.copy()
        # Adaptive destroy size: try mixed sizes to escape different optima
        k_destroy = int(rng.integers(max(3, n // 20), max(4, n // 4)))
        idxs = rng.choice(n, size=k_destroy, replace=False)
        for idx in idxs:
            cand[idx] = rng.random()
        cand_res = decoder(cand, boxes, pallet, config, n_rots_arr, dims_all,
                           max_pallets=max_pallets)
        cand_fit = _fitness_pallet1(cand_res, pallet)
        iters += 1
        if cand_fit < best_fit - 1e-9:
            best_fit = cand_fit
            best_res = cand_res
            current = cand
            accepts += 1
            if verbose:
                print(f"  [LNS] iter {iters} accept: util={(1-best_fit)*100:.2f}%")
    if verbose:
        print(f"  [LNS] {accepts}/{iters} accepted, final={(1-best_fit)*100:.2f}%")
    return best_res, current, accepts


# ============================================================================
# SKU-aware encoding (v3.7) — shrinks chromosome from 2N+1 to 2S+1 where
# S = number of unique SKUs. Crucial for homogeneous loads (BR1, real-world
# warehouse stock) where the same SKU appears many times. BRKGA over a
# 7-key chromosome (S=3 SKUs) converges in dozens of generations instead
# of hundreds for the 225-key box-level chromosome.
# ============================================================================

def enumerate_top_k_blocks_per_sku(
    boxes: List[Box],
    sku_id_per_box: np.ndarray,
    n_rots_arr: np.ndarray,
    dims_all: np.ndarray,
    L: int, W: int, H: int,
    k_top: int = 8,
) -> np.ndarray:
    """For each SKU, enumerate top-K candidate (k, l, m, rotation) blocks.

    Returns (n_skus, k_top, 4) int64. Each SKU has up to k_top distinct
    block shapes, sorted by box count descending (largest first).
    Singleton/missing entries padded with (1, 1, 1, 0).

    Blocks differ by COUNT (k*l*m): top-1 is biggest, top-2 is next, etc.
    Letting BRKGA pick which block per SKU avoids the "too greedy" failure
    of always using top-1.
    """
    n_skus = int(sku_id_per_box.max()) + 1 if len(sku_id_per_box) > 0 else 0
    top_k = np.zeros((n_skus, k_top, 4), dtype=np.int64)
    top_k[:, :, 0] = 1  # default (1,1,1,0)
    top_k[:, :, 1] = 1
    top_k[:, :, 2] = 1
    top_k[:, :, 3] = 0
    sku_groups = [[] for _ in range(n_skus)]
    for i in range(len(boxes)):
        sku_groups[int(sku_id_per_box[i])].append(i)
    for sku in range(n_skus):
        idxs = sku_groups[sku]
        if len(idxs) < 2:
            continue
        sample = idxs[0]
        count = len(idxs)
        n_rots = int(n_rots_arr[sample])
        # Enumerate all (k, l, m, rot) candidates
        candidates = []
        for r in range(n_rots):
            dx = int(dims_all[sample, r, 0])
            dy = int(dims_all[sample, r, 1])
            dz = int(dims_all[sample, r, 2])
            if dx <= 0 or dy <= 0 or dz <= 0:
                continue
            max_k = min(L // dx, count)
            max_l = min(W // dy, count)
            max_m = min(H // dz, count)
            for k in range(1, max_k + 1):
                if k > count:
                    break
                for l in range(1, max_l + 1):
                    if k * l > count:
                        break
                    for m in range(1, max_m + 1):
                        n = k * l * m
                        if n > count:
                            break
                        if n < 2:
                            continue
                        candidates.append((n, k, l, m, r))
        # Sort by count desc, dedupe by count (keep one per count)
        candidates.sort(key=lambda c: -c[0])
        seen_counts = set()
        unique = []
        for c in candidates:
            if c[0] in seen_counts:
                continue
            seen_counts.add(c[0])
            unique.append(c)
            if len(unique) >= k_top:
                break
        for i, (n, k, l, m, r) in enumerate(unique):
            top_k[sku, i, 0] = k
            top_k[sku, i, 1] = l
            top_k[sku, i, 2] = m
            top_k[sku, i, 3] = r
        # Fill remaining slots with the smallest available (or default)
        for i in range(len(unique), k_top):
            if unique:
                top_k[sku, i] = top_k[sku, len(unique) - 1]
    return top_k


def resolve_sku_blocks_from_chrom(
    chrom: np.ndarray,
    n_boxes: int,
    n_skus: int,
    top_k_blocks: np.ndarray,
) -> np.ndarray:
    """Pick one (k, l, m, rot) block per SKU based on chromosome.

    Uses chrom[n_boxes + sku] (first S keys of VBO portion) as selector.
    These VBO keys are unused by JIT decoders (rotation is chosen during
    placement), so repurposing them for SKU block selection is safe.

    Returns (n_skus, 4) int64.
    """
    k_top = top_k_blocks.shape[1]
    chosen = np.zeros((n_skus, 4), dtype=np.int64)
    for s in range(n_skus):
        if n_boxes + s < len(chrom):
            key = chrom[n_boxes + s]
        else:
            key = 0.0
        idx = min(k_top - 1, max(0, int(key * k_top)))
        chosen[s] = top_k_blocks[s, idx]
    return chosen


def enumerate_best_block_per_sku(
    boxes: List[Box],
    sku_id_per_box: np.ndarray,
    n_rots_arr: np.ndarray,
    dims_all: np.ndarray,
    L: int, W: int, H: int,
) -> np.ndarray:
    """For each SKU, find the BEST (k, l, m, rotation) block by max volume.

    Returns (n_skus, 4) int64 array: best_blocks[sku] = [k, l, m, rot_idx].
    For singleton SKUs (count=1), returns [1, 1, 1, 0] (no real block).
    """
    n_skus = int(sku_id_per_box.max()) + 1 if len(sku_id_per_box) > 0 else 0
    best_blocks = np.ones((n_skus, 4), dtype=np.int64)  # default (1,1,1,0)
    sku_groups = [[] for _ in range(n_skus)]
    for i in range(len(boxes)):
        sku_groups[int(sku_id_per_box[i])].append(i)
    for sku in range(n_skus):
        idxs = sku_groups[sku]
        if len(idxs) < 2:
            best_blocks[sku, 3] = 0
            continue
        sample = idxs[0]
        count = len(idxs)
        n_rots = int(n_rots_arr[sample])
        best_n = 1
        best_k, best_l, best_m, best_rot = 1, 1, 1, 0
        for r in range(n_rots):
            dx = int(dims_all[sample, r, 0])
            dy = int(dims_all[sample, r, 1])
            dz = int(dims_all[sample, r, 2])
            if dx <= 0 or dy <= 0 or dz <= 0:
                continue
            max_k = min(L // dx, count)
            max_l = min(W // dy, count)
            max_m = min(H // dz, count)
            for k in range(1, max_k + 1):
                if k > count:
                    break
                for l in range(1, max_l + 1):
                    if k * l > count:
                        break
                    for m in range(1, max_m + 1):
                        n = k * l * m
                        if n > count:
                            break
                        if n > best_n:
                            best_n = n
                            best_k, best_l, best_m, best_rot = k, l, m, r
        best_blocks[sku, 0] = best_k
        best_blocks[sku, 1] = best_l
        best_blocks[sku, 2] = best_m
        best_blocks[sku, 3] = best_rot
    return best_blocks


def compute_sku_groups(boxes: List[Box], sku_id_per_box: np.ndarray) -> List[List[int]]:
    """For each SKU id, return the list of box indices that belong to it.

    Returns list of lists, length n_skus. sku_box_indices[s][k] = index of
    the k-th box in SKU s.
    """
    n_skus = int(sku_id_per_box.max()) + 1 if len(sku_id_per_box) > 0 else 0
    sku_box_indices: List[List[int]] = [[] for _ in range(n_skus)]
    for i in range(len(boxes)):
        sku_box_indices[int(sku_id_per_box[i])].append(i)
    return sku_box_indices


def sku_chrom_to_full_chrom(
    sku_chrom: np.ndarray,
    n_skus: int,
    sku_box_indices: List[List[int]],
    n_total_boxes: int,
) -> np.ndarray:
    """Translate a 2S+1 SKU-aware chromosome into a 2N+1 box-level chromosome.

    SKU chromosome layout:
      [0:S]       — SKU order keys (argsort gives SKU processing order)
      [S:2S]      — Rotation choice key per SKU (used for VBO)
      [-1]        — Mode selector (we'll force mode 4 = blocks)

    Full chromosome layout (compatible with decode_chromosome mode=4):
      [0:N]       — BPS keys (SKU-grouped: all SKU0 boxes consecutive, etc.)
      [N:2N]      — VBO keys (per-box rotation, derived from SKU rotation key)
      [-1]        — Mode selector
    """
    sku_order_keys = sku_chrom[:n_skus]
    sku_rot_keys = sku_chrom[n_skus:2 * n_skus]
    selector = sku_chrom[-1] if len(sku_chrom) > 2 * n_skus else 0.0

    sku_order = np.argsort(sku_order_keys).astype(np.int64)
    full_chrom = np.zeros(2 * n_total_boxes + 1, dtype=np.float64)
    pos = 0
    for sku_id in sku_order:
        for box_id in sku_box_indices[int(sku_id)]:
            full_chrom[box_id] = (pos + 0.5) / n_total_boxes
            full_chrom[n_total_boxes + box_id] = float(sku_rot_keys[int(sku_id)])
            pos += 1
    # Force mode 4 (blocks) for SKU-aware decoding
    full_chrom[-1] = 0.85  # selector >= 0.8 picks mode 4 with n_modes=5
    return full_chrom


def brkga_sku_aware_search(
    boxes: List[Box],
    pallet: Pallet,
    config: PackerConfig,
    n_rots_arr: np.ndarray,
    dims_all: np.ndarray,
    sku_id_per_box: np.ndarray,
    *,
    time_budget_s: float = 8.0,
    pop_size: int = 80,
    generations: int = 1000,
    elite_fraction: float = 0.25,
    mutant_fraction: float = 0.20,
    p_elite_inherit: float = 0.70,
    seed: int = 42,
    max_pallets: int = 1,
    verbose: bool = False,
) -> Tuple[Optional[PackResult], Optional[np.ndarray], int]:
    """Mini-BRKGA over 2S+1 SKU-aware chromosomes.

    Translates each candidate to a full 2N+1 chromosome and decodes via
    mode 4 (DFTRC + blocks). Returns (best_result, best_full_chromosome,
    n_decodes).

    Designed for problems with S << N (homogeneous loads). For BR1 (S=3,
    N=112), search space = 3! × ~6^3 ≈ 1300 states; BRKGA enumerates
    most of them in seconds.
    """
    n_total = len(boxes)
    if n_total == 0:
        return None, None, 0

    n_skus = int(sku_id_per_box.max()) + 1
    sku_box_indices = compute_sku_groups(boxes, sku_id_per_box)
    chrom_size = 2 * n_skus + 1

    rng = np.random.default_rng(seed)
    pop = rng.random((pop_size, chrom_size))

    # ------------- Inject informed seeds (use diverse SKU orderings) ------
    sku_vols = np.array([sum(boxes[i].volume for i in sku_box_indices[s])
                          for s in range(n_skus)])
    sku_counts = np.array([len(sku_box_indices[s]) for s in range(n_skus)])

    def encode_order(order_arr: np.ndarray, slot: int) -> None:
        """Encode SKU order into pop[slot]."""
        for pos, sku_id in enumerate(order_arr):
            pop[slot, sku_id] = (pos + 0.5) / n_skus
        # Rotation keys: try rotation 0 explicitly
        pop[slot, n_skus:2 * n_skus] = 0.05  # picks rotation 0 (first allowed)
        pop[slot, -1] = 0.85

    seed_slot = 0
    # Order 1: by total volume descending
    encode_order(np.argsort(-sku_vols), seed_slot); seed_slot += 1
    # Order 2: by count descending
    encode_order(np.argsort(-sku_counts), seed_slot); seed_slot += 1
    # Order 3: by total volume ascending
    encode_order(np.argsort(sku_vols), seed_slot); seed_slot += 1
    # Order 4-6: explicitly different rotation choices for largest-vol order
    for r_key in (0.5, 0.95, 0.3):
        if seed_slot >= pop_size:
            break
        encode_order(np.argsort(-sku_vols), seed_slot)
        pop[seed_slot, n_skus:2 * n_skus] = r_key
        seed_slot += 1

    n_elite = max(1, int(pop_size * elite_fraction))
    n_mutant = max(1, int(pop_size * mutant_fraction))
    n_cross = max(0, pop_size - n_elite - n_mutant)

    best_fit = float('inf')
    best_full_chrom: Optional[np.ndarray] = None
    best_result: Optional[PackResult] = None
    t0 = time.time()
    total_decodes = 0

    fits = np.zeros(pop_size)
    for gen in range(generations):
        if time.time() - t0 > time_budget_s:
            if verbose:
                print(f"  [SKU-aware] gen {gen}: time hit, decodes={total_decodes}")
            break

        # Evaluate
        for i in range(pop_size):
            full_chrom = sku_chrom_to_full_chrom(
                pop[i], n_skus, sku_box_indices, n_total)
            res = decode_chromosome(
                full_chrom, boxes, pallet, config,
                n_rots_arr, dims_all, mode=4,
                max_pallets=max_pallets, sku_id_per_box=sku_id_per_box,
            )
            fits[i] = _fitness_pallet1(res, pallet)
            total_decodes += 1
            if fits[i] < best_fit - 1e-9:
                best_fit = float(fits[i])
                best_full_chrom = full_chrom.copy()
                best_result = res
                if verbose:
                    print(f"  [SKU-aware] gen {gen} ind {i}: util={(1-best_fit)*100:.2f}%")

        # Evolve
        sorted_idx = np.argsort(fits)
        new_pop = np.zeros_like(pop)
        new_pop[:n_elite] = pop[sorted_idx[:n_elite]]
        new_pop[n_elite:n_elite + n_mutant] = rng.random((n_mutant, chrom_size))
        elite_pool = pop[sorted_idx[:n_elite]]
        non_elite_pool = pop[sorted_idx[n_elite:]]
        if n_cross > 0 and len(non_elite_pool) > 0:
            pa_idx = rng.integers(0, n_elite, size=n_cross)
            pb_idx = rng.integers(0, len(non_elite_pool), size=n_cross)
            pa_parents = elite_pool[pa_idx]
            pb_parents = non_elite_pool[pb_idx]
            mask = rng.random((n_cross, chrom_size)) < p_elite_inherit
            new_pop[n_elite + n_mutant:] = np.where(mask, pa_parents, pb_parents)
        pop = new_pop

    return best_result, best_full_chrom, total_decodes


# ============================================================================
# v3.5 main driver
# ============================================================================

def brkga_pack_v35(
    boxes: List[Box],
    pallet: Pallet,
    config: PackerConfig,
    *,
    time_limit_s: float = 30.0,
    max_pallets: int = 1,
    seed: int = 42,
    # BRKGA params
    population_size: int = 600,
    generations: int = 1000,
    n_populations: int = 3,
    elite_fraction: float = 0.20,
    mutant_fraction: float = 0.15,
    p_elite_inherit: float = 0.70,
    migration_interval: int = 15,
    migrants_per_swap: int = 3,
    patience: int = 100,
    # v3.5 features
    use_multi_decoder: bool = True,
    n_modes: int = 6,  # 6 = DFTRC + wall + corner + layer + DFTRC+blocks + precomp-blocks
    use_v2_seed: bool = True,
    use_smart_init: bool = True,
    use_local_search: bool = True,
    local_search_budget_s: float = 4.0,
    use_lns: bool = False,
    lns_budget_s: float = 3.0,
    # SKU-aware mini-BRKGA (v3.7, experimental): runs first with small budget,
    # uses 2S+1 chromosome (S=#SKUs). Designed to help homogeneous loads, but
    # empirical n=3 test showed neutral-to-negative impact (-0.10 to +0.70pp
    # depending on set, -0.59 on BR7). Default OFF; enable via flag for
    # experimentation. See docs/reports/17_v37_sku_aware.md.
    use_sku_aware: bool = False,
    sku_aware_budget_s: float = 5.0,
    use_v2_hybrid_polish: bool = True,
    # Multi-restart: run K times with different seeds, take best.
    # Reduces variance at the cost of less compute per run.
    n_restarts: int = 1,
    verbose: bool = False,
) -> PackResult:
    """v3.5: hybrid BRKGA with all quality improvements."""
    # If multi-restart, recurse into single-run with split budget.
    if n_restarts > 1:
        time_per = time_limit_s / n_restarts
        best_result: Optional[PackResult] = None
        best_util = -1.0
        cap = pallet.length * pallet.width * pallet.height
        for r_idx in range(n_restarts):
            res = brkga_pack_v35(
                boxes, pallet, config,
                time_limit_s=time_per, max_pallets=max_pallets,
                seed=seed + r_idx * 17,
                population_size=population_size, generations=generations,
                n_populations=n_populations,
                elite_fraction=elite_fraction, mutant_fraction=mutant_fraction,
                p_elite_inherit=p_elite_inherit,
                migration_interval=migration_interval,
                migrants_per_swap=migrants_per_swap,
                patience=patience,
                use_multi_decoder=use_multi_decoder,
                n_modes=n_modes,
                use_v2_seed=use_v2_seed and r_idx == 0,  # only first run uses v2 (slow)
                use_smart_init=use_smart_init,
                use_local_search=use_local_search,
                local_search_budget_s=local_search_budget_s,
                use_lns=use_lns,
                lns_budget_s=lns_budget_s,
                use_sku_aware=use_sku_aware,
                sku_aware_budget_s=sku_aware_budget_s,
                use_v2_hybrid_polish=use_v2_hybrid_polish and r_idx == 0,
                n_restarts=1,
                verbose=verbose,
            )
            used = (sum(p.box.volume for p in res.pallets[0].placements)
                    if res.pallets else 0)
            util = used / cap if cap > 0 else 0.0
            if verbose:
                print(f"  [v3.5 multi-restart] run {r_idx+1}/{n_restarts}: util={util*100:.2f}%")
            if util > best_util:
                best_util = util
                best_result = res
        return best_result if best_result is not None else PackResult(
            pallets=[], unpacked=list(boxes))
    # ----------------------- single-run path -----------------------
    n = len(boxes)
    if n == 0:
        return PackResult(pallets=[], unpacked=[])

    warmup_jit()
    n_rots_arr, dims_all, sku_id_per_box = precompute_box_dims_and_sku(boxes)
    L = int(round(pallet.length))
    W = int(round(pallet.width))
    H = int(round(pallet.height))
    sku_best_block = enumerate_best_block_per_sku(
        boxes, sku_id_per_box, n_rots_arr, dims_all, L, W, H)
    sku_top_k_blocks = enumerate_top_k_blocks_per_sku(
        boxes, sku_id_per_box, n_rots_arr, dims_all, L, W, H, k_top=8)
    chrom_size = 2 * n + (1 if use_multi_decoder else 0)
    if use_multi_decoder:
        decoder = lambda c, b, p, cfg, na, da, max_pallets=1: decode_auto_mode(
            c, b, p, cfg, na, da, max_pallets=max_pallets, n_modes=n_modes,
            sku_id_per_box=sku_id_per_box, sku_best_block=sku_best_block,
            sku_top_k_blocks=sku_top_k_blocks)
    else:
        decoder = lambda c, b, p, cfg, na, da, max_pallets=1: decode_chromosome(
            c, b, p, cfg, na, da, mode=0, max_pallets=max_pallets)

    t_start = time.time()

    # ----------------------------------------------------------------------
    # Phase 0: v2 seed (cheap, gives us a strong baseline chromosome)
    # ----------------------------------------------------------------------
    v2_result: Optional[PackResult] = None
    v2_seed_chrom: Optional[np.ndarray] = None
    v2_seed_chroms_per_mode: List[np.ndarray] = []
    if use_v2_seed:
        try:
            from .packer import PalletPacker
            v2_packer = PalletPacker(pallet, config)
            v2_result = v2_packer.pack(boxes)
            if use_multi_decoder:
                for m in range(n_modes):
                    cs = chromosome_from_v2_result(v2_result, boxes, n_rots_arr,
                                                    decoder_mode=m,
                                                    n_modes=n_modes,
                                                    rng=np.random.default_rng(seed + 7 + m))
                    if cs is not None:
                        v2_seed_chroms_per_mode.append(cs)
            else:
                v2_seed_chrom = chromosome_from_v2_result(
                    v2_result, boxes, n_rots_arr,
                    rng=np.random.default_rng(seed + 7))
        except Exception as e:
            if verbose:
                print(f"  [v3.5] v2 seed failed: {e}")
    v2_time = time.time() - t_start
    if verbose:
        u = 0
        if v2_result and v2_result.pallets:
            cap = pallet.length * pallet.width * pallet.height
            u = sum(p.box.volume for p in v2_result.pallets[0].placements) / cap * 100
        print(f"  [v3.5] v2 seed: {v2_time:.2f}s, v2 util={u:.2f}%")

    # ----------------------------------------------------------------------
    # Phase 1: Smart init + random fill of populations
    # ----------------------------------------------------------------------
    K = max(1, n_populations)
    pop_size = max(20, population_size // K)
    n_elite = max(1, int(pop_size * elite_fraction))
    n_mutant = max(1, int(pop_size * mutant_fraction))
    n_cross = max(0, pop_size - n_elite - n_mutant)

    rngs = [np.random.default_rng(seed + i) for i in range(K)]
    pops = [rngs[i].random((pop_size, chrom_size)) for i in range(K)]

    if use_smart_init or use_v2_seed:
        # Inject informed chromosomes into the front of pop 0
        smart = []
        if use_smart_init:
            if use_multi_decoder:
                for m in range(n_modes):
                    smart.extend(make_informed_chromosomes(
                        boxes, n_rots_arr, decoder_mode=m,
                        n_modes=n_modes, seed=seed + 100 + m))
            else:
                smart.extend(make_informed_chromosomes(
                    boxes, n_rots_arr, decoder_mode=None, seed=seed + 100))
        if v2_seed_chrom is not None:
            smart.insert(0, v2_seed_chrom)
        if v2_seed_chroms_per_mode:
            for c in v2_seed_chroms_per_mode:
                smart.insert(0, c)
        # Insert into front of pop 0 (and spread across pops if K > 1)
        for i, c in enumerate(smart[:pop_size]):
            pops[0][i] = c[:chrom_size]
        # Also seed pop 1 with one smart chromosome if K > 1
        if K > 1 and smart:
            for i, c in enumerate(smart[:pop_size // 2]):
                idx_in_pop1 = i % pop_size
                pops[1 % K][idx_in_pop1] = c[:chrom_size]

    # ----------------------------------------------------------------------
    # Phase 1b: SKU-aware mini-BRKGA (v3.7)
    # ----------------------------------------------------------------------
    # Runs early with small budget. Best result becomes a strong seed for
    # main BRKGA (Phase 2). Especially crucial when S << N (BR1, real-world
    # homogeneous stock).
    sku_aware_result: Optional[PackResult] = None
    sku_aware_chrom: Optional[np.ndarray] = None
    sku_aware_time = 0.0
    if use_sku_aware and sku_aware_budget_s > 0:
        t_sku = time.time()
        sku_aware_result, sku_aware_chrom, sku_aware_decodes = brkga_sku_aware_search(
            boxes, pallet, config, n_rots_arr, dims_all, sku_id_per_box,
            time_budget_s=sku_aware_budget_s,
            seed=seed + 11, max_pallets=max_pallets, verbose=verbose,
        )
        sku_aware_time = time.time() - t_sku
        if verbose and sku_aware_result is not None:
            sku_u = (1 - _fitness_pallet1(sku_aware_result, pallet)) * 100
            print(f"  [v3.5] SKU-aware: util={sku_u:.2f}% in {sku_aware_time:.2f}s "
                  f"({sku_aware_decodes} decodes)")

    # ----------------------------------------------------------------------
    # Phase 2: BRKGA
    # ----------------------------------------------------------------------
    # Reserve budget for LS + PR (if enabled)
    polish_budget = ((local_search_budget_s if use_local_search else 0)
                     + (lns_budget_s if use_lns else 0))
    brkga_budget = time_limit_s - v2_time - sku_aware_time - polish_budget
    brkga_budget = max(1.0, brkga_budget)

    best_fitness = float('inf')
    best_result: Optional[PackResult] = None
    best_chrom: Optional[np.ndarray] = None
    # If SKU-aware found a result, seed best with it
    if sku_aware_result is not None and sku_aware_chrom is not None:
        sku_fit = _fitness_pallet1(sku_aware_result, pallet)
        best_fitness = float(sku_fit)
        best_result = sku_aware_result
        best_chrom = sku_aware_chrom
        # Also inject SKU-aware chromosome into pop[0] as seed
        if (sku_aware_chrom is not None and chrom_size == len(sku_aware_chrom)
                and pops[0].shape[0] > 0):
            pops[0][0] = sku_aware_chrom.copy()
    # Track top-2 distinct elites for path relinking
    second_best_fitness = float('inf')
    second_best_chrom: Optional[np.ndarray] = None
    gens_no_improve = 0
    t0 = time.time()
    total_decodes = 0
    pop_fits = [np.zeros(pop_size) for _ in range(K)]

    for gen in range(generations):
        if time.time() - t0 > brkga_budget:
            if verbose:
                print(f"  [v3.5 BRKGA] gen {gen}: time hit, decodes={total_decodes}")
            break

        for k in range(K):
            fits = pop_fits[k]
            for i in range(pop_size):
                res = decoder(pops[k][i], boxes, pallet, config,
                              n_rots_arr, dims_all, max_pallets=max_pallets)
                fits[i] = _fitness_pallet1(res, pallet)
                total_decodes += 1
                if fits[i] < best_fitness - 1e-9:
                    # Demote current best to second
                    second_best_fitness = best_fitness
                    second_best_chrom = best_chrom
                    best_fitness = float(fits[i])
                    best_result = res
                    best_chrom = pops[k][i].copy()
                    gens_no_improve = 0
                    if verbose:
                        print(f"  [v3.5 BRKGA] gen {gen} pop {k}: util={(1-best_fitness)*100:.2f}%")
                elif (fits[i] < second_best_fitness - 1e-9 and
                      fits[i] > best_fitness + 1e-9):
                    # Update second-best (distinct from best)
                    second_best_fitness = float(fits[i])
                    second_best_chrom = pops[k][i].copy()

        gens_no_improve += 1
        if gens_no_improve >= patience:
            if verbose:
                print(f"  [v3.5 BRKGA] gen {gen}: patience, decodes={total_decodes}")
            break

        # Migration
        if K > 1 and gen > 0 and gen % migration_interval == 0:
            for k in range(K):
                src = k
                dst = (k + 1) % K
                src_sorted = np.argsort(pop_fits[src])[:migrants_per_swap]
                dst_sorted = np.argsort(pop_fits[dst])[-migrants_per_swap:]
                for s, d in zip(src_sorted, dst_sorted):
                    pops[dst][d] = pops[src][s].copy()

        # Evolve
        new_pops = []
        for k in range(K):
            sorted_idx = np.argsort(pop_fits[k])
            new_pop = np.zeros_like(pops[k])
            new_pop[:n_elite] = pops[k][sorted_idx[:n_elite]]
            new_pop[n_elite:n_elite + n_mutant] = rngs[k].random((n_mutant, chrom_size))
            elite_pool = pops[k][sorted_idx[:n_elite]]
            non_elite_pool = pops[k][sorted_idx[n_elite:]]
            if n_cross > 0 and len(non_elite_pool) > 0:
                pa_idx = rngs[k].integers(0, n_elite, size=n_cross)
                pb_idx = rngs[k].integers(0, len(non_elite_pool), size=n_cross)
                pa_parents = elite_pool[pa_idx]
                pb_parents = non_elite_pool[pb_idx]
                mask = rngs[k].random((n_cross, chrom_size)) < p_elite_inherit
                new_pop[n_elite + n_mutant:] = np.where(mask, pa_parents, pb_parents)
            new_pops.append(new_pop)
        pops = new_pops

    # ----------------------------------------------------------------------
    # Phase 3a: Path relinking between top-2 elites (cheap)
    # ----------------------------------------------------------------------
    if (best_chrom is not None and second_best_chrom is not None
            and use_local_search):
        pr_res, pr_chrom, pr_fit = path_relinking(
            best_chrom, second_best_chrom,
            boxes, pallet, config, n_rots_arr, dims_all,
            multi_decoder=use_multi_decoder, n_modes=n_modes,
            sku_id_per_box=sku_id_per_box,
            max_pallets=max_pallets, max_evals=50, verbose=verbose,
        )
        if pr_fit < best_fitness - 1e-9:
            best_fitness = pr_fit
            best_result = pr_res
            best_chrom = pr_chrom
            if verbose:
                print(f"  [v3.5] path relinking improved: util={(1-best_fitness)*100:.2f}%")

    # ----------------------------------------------------------------------
    # Phase 3b: Local search polish
    # ----------------------------------------------------------------------
    if use_local_search and best_chrom is not None:
        ls_res, ls_chrom, accepts = local_search_2opt(
            best_chrom, boxes, pallet, config, n_rots_arr, dims_all,
            time_budget_s=local_search_budget_s,
            multi_decoder=use_multi_decoder, n_modes=n_modes,
            sku_id_per_box=sku_id_per_box,
            max_pallets=max_pallets,
            seed=seed + 999, verbose=verbose,
        )
        ls_fit = _fitness_pallet1(ls_res, pallet)
        if ls_fit < best_fitness - 1e-9:
            best_fitness = ls_fit
            best_result = ls_res
            best_chrom = ls_chrom
            if verbose:
                print(f"  [v3.5] local search improved: util={(1-best_fitness)*100:.2f}% ({accepts} accepts)")

    # ----------------------------------------------------------------------
    # Phase 3c: Large Neighborhood Search (LNS) polish
    # ----------------------------------------------------------------------
    if use_lns and best_chrom is not None:
        lns_res, lns_chrom, lns_accepts = lns_polish(
            best_chrom, boxes, pallet, config, n_rots_arr, dims_all,
            multi_decoder=use_multi_decoder, n_modes=n_modes,
            sku_id_per_box=sku_id_per_box,
            max_pallets=max_pallets,
            time_budget_s=lns_budget_s,
            seed=seed + 1234, verbose=verbose,
        )
        lns_fit = _fitness_pallet1(lns_res, pallet)
        if lns_fit < best_fitness - 1e-9:
            best_fitness = lns_fit
            best_result = lns_res
            best_chrom = lns_chrom
            if verbose:
                print(f"  [v3.5] LNS improved: util={(1-best_fitness)*100:.2f}% ({lns_accepts} accepts)")

    # ----------------------------------------------------------------------
    # Phase 4: Compare with v2 hybrid (if enabled)
    # ----------------------------------------------------------------------
    if use_v2_hybrid_polish and v2_result is not None and v2_result.pallets:
        v2_fit = _fitness_pallet1(v2_result, pallet)
        if v2_fit < best_fitness - 1e-9:
            if verbose:
                print(f"  [v3.5] v2 hybrid wins: util={(1-v2_fit)*100:.2f}%")
            return v2_result

    if best_result is None:
        if v2_result is not None:
            return v2_result
        return PackResult(pallets=[], unpacked=list(boxes))
    return best_result
