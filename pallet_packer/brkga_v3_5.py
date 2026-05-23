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

def decode_chromosome(
    chrom: np.ndarray,
    boxes: List[Box],
    pallet: Pallet,
    config: PackerConfig,
    n_rots_arr: np.ndarray,
    dims_all: np.ndarray,
    mode: int,
    max_pallets: int = 1,
) -> PackResult:
    """Decode chromosome with chosen mode (0=DFTRC, 1=wall, 2=corner, 3=layer)."""
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
    n_modes: int = 4,
) -> PackResult:
    """Use last key in chromosome to pick decoder mode (0=DFTRC, 1=wall,
    2=corner, 3=layer-build).
    """
    n = len(boxes)
    if len(chrom) >= 2 * n + 1:
        selector = chrom[-1]
        mode = min(n_modes - 1, int(selector * n_modes))
    else:
        mode = 0
    return decode_chromosome(chrom, boxes, pallet, config,
                             n_rots_arr, dims_all, mode, max_pallets)


# ============================================================================
# Smart initialization
# ============================================================================

def chromosome_from_order(order: List[int], n: int, n_rots_arr: np.ndarray,
                          rotation_choices: Optional[List[int]] = None,
                          decoder_mode: Optional[int] = None,
                          n_modes: int = 4,
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
    n_modes: int = 4,
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
    n_modes: int = 4,
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
    decoder = decode_auto_mode if multi_decoder else (
        lambda c, b, p, cfg, na, da, max_pallets=1:
        decode_chromosome(c, b, p, cfg, na, da, mode=0, max_pallets=max_pallets)
    )
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
    n_modes: int = 4,  # 4 = DFTRC + wall + corner + layer-build
    use_v2_seed: bool = True,
    use_smart_init: bool = True,
    use_local_search: bool = True,
    local_search_budget_s: float = 4.0,
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

    n_rots_arr, dims_all = precompute_box_dims(boxes)
    chrom_size = 2 * n + (1 if use_multi_decoder else 0)
    if use_multi_decoder:
        decoder = lambda c, b, p, cfg, na, da, max_pallets=1: decode_auto_mode(
            c, b, p, cfg, na, da, max_pallets=max_pallets, n_modes=n_modes)
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
    # Phase 2: BRKGA
    # ----------------------------------------------------------------------
    brkga_budget = (time_limit_s - v2_time
                    - (local_search_budget_s if use_local_search else 0))
    brkga_budget = max(1.0, brkga_budget)

    best_fitness = float('inf')
    best_result: Optional[PackResult] = None
    best_chrom: Optional[np.ndarray] = None
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
                    best_fitness = float(fits[i])
                    best_result = res
                    best_chrom = pops[k][i].copy()
                    gens_no_improve = 0
                    if verbose:
                        print(f"  [v3.5 BRKGA] gen {gen} pop {k}: util={(1-best_fitness)*100:.2f}%")

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
    # Phase 3: Local search polish
    # ----------------------------------------------------------------------
    if use_local_search and best_chrom is not None:
        ls_res, ls_chrom, accepts = local_search_2opt(
            best_chrom, boxes, pallet, config, n_rots_arr, dims_all,
            time_budget_s=local_search_budget_s,
            multi_decoder=use_multi_decoder,
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
