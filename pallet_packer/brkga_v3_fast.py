"""
pallet_packer.brkga_v3_fast — Numba-accelerated BRKGA-2013 decoder.

Reimplements the hot path of brkga_v3.py with numba @njit for 10-50×
speedup over pure numpy. Enables BRKGA at literature-scale compute
within practical wall-clock budgets.

Same algorithm as brkga_v3: EMS-based maximal-space tracking + DFTRC-2
placement + Lai-Chan 1997 difference process. Just much faster.
"""
from __future__ import annotations

import time
from typing import List, Optional

import numpy as np
from numba import njit, types
from numba.typed import List as NumbaList

from .models import Box, Pallet, PackerConfig, Placement, Rotation
from .packer import PackResult, PalletState


# Max EMS list size per bin. EMS lists in 3D-BPP rarely exceed ~200 for
# typical N=100-200 instances. Numba prefers fixed-size arrays.
MAX_EMS = 512


@njit(cache=True, fastmath=True)
def find_best_dftrc_njit(
    emss: np.ndarray, n_ems: int,
    dx: int, dy: int, dz: int,
    L: int, W: int, H: int,
) -> tuple:
    """For dims (dx, dy, dz), find the EMS where DFTRC-2 is maximized.
    Returns (best_idx, x, y, z) or (-1, 0, 0, 0) if no fit.
    """
    best_idx = -1
    best_score = -1
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
        # Place at EMS min corner.
        x = ex_min
        y = ey_min
        z = ez_min
        d_sq = ((L - x - dx) * (L - x - dx)
                + (W - y - dy) * (W - y - dy)
                + (H - z - dz) * (H - z - dz))
        if d_sq > best_score:
            best_score = d_sq
            best_idx = i
            bx, by, bz = x, y, z
    return best_idx, bx, by, bz


@njit(cache=True, fastmath=True)
def commit_ems_njit(
    emss: np.ndarray, n_ems: int,
    bx1: int, by1: int, bz1: int,
    bx2: int, by2: int, bz2: int,
    out_buf: np.ndarray,
) -> int:
    """Apply the difference process: replace overlapping EMSs with up
    to 6 children each, then prune dominated. Writes into out_buf and
    returns new count.

    emss: current EMS array (MAX_EMS, 2, 3), only first n_ems valid
    out_buf: scratch buffer (MAX_EMS, 2, 3) for new EMS list
    """
    # Phase 1: copy non-overlapping EMSs + generate children for overlapping.
    tmp_count = 0
    for i in range(n_ems):
        ex1 = emss[i, 0, 0]
        ey1 = emss[i, 0, 1]
        ez1 = emss[i, 0, 2]
        ex2 = emss[i, 1, 0]
        ey2 = emss[i, 1, 1]
        ez2 = emss[i, 1, 2]
        if not (bx1 < ex2 and bx2 > ex1
                and by1 < ey2 and by2 > ey1
                and bz1 < ez2 and bz2 > ez1):
            # No overlap — keep.
            if tmp_count < MAX_EMS:
                out_buf[tmp_count, 0, 0] = ex1
                out_buf[tmp_count, 0, 1] = ey1
                out_buf[tmp_count, 0, 2] = ez1
                out_buf[tmp_count, 1, 0] = ex2
                out_buf[tmp_count, 1, 1] = ey2
                out_buf[tmp_count, 1, 2] = ez2
                tmp_count += 1
            continue
        # 6 child candidates (Lai-Chan 1997).
        for which in range(6):
            if which == 0:
                lo0, lo1, lo2 = bx2, ey1, ez1
                hi0, hi1, hi2 = ex2, ey2, ez2
            elif which == 1:
                lo0, lo1, lo2 = ex1, ey1, ez1
                hi0, hi1, hi2 = bx1, ey2, ez2
            elif which == 2:
                lo0, lo1, lo2 = ex1, by2, ez1
                hi0, hi1, hi2 = ex2, ey2, ez2
            elif which == 3:
                lo0, lo1, lo2 = ex1, ey1, ez1
                hi0, hi1, hi2 = ex2, by1, ez2
            elif which == 4:
                lo0, lo1, lo2 = ex1, ey1, bz2
                hi0, hi1, hi2 = ex2, ey2, ez2
            else:
                lo0, lo1, lo2 = ex1, ey1, ez1
                hi0, hi1, hi2 = ex2, ey2, bz1
            if hi0 > lo0 and hi1 > lo1 and hi2 > lo2:
                if tmp_count < MAX_EMS:
                    out_buf[tmp_count, 0, 0] = lo0
                    out_buf[tmp_count, 0, 1] = lo1
                    out_buf[tmp_count, 0, 2] = lo2
                    out_buf[tmp_count, 1, 0] = hi0
                    out_buf[tmp_count, 1, 1] = hi1
                    out_buf[tmp_count, 1, 2] = hi2
                    tmp_count += 1

    # Phase 2: dominance pruning — remove EMS i if some j strictly contains i.
    # Vectorized inner check would be nice but numba does well with explicit loops.
    # Process in-place by writing kept EMSs to the front.
    keep_count = 0
    for i in range(tmp_count):
        i_lo0 = out_buf[i, 0, 0]; i_lo1 = out_buf[i, 0, 1]; i_lo2 = out_buf[i, 0, 2]
        i_hi0 = out_buf[i, 1, 0]; i_hi1 = out_buf[i, 1, 1]; i_hi2 = out_buf[i, 1, 2]
        dominated = False
        for j in range(tmp_count):
            if i == j:
                continue
            j_lo0 = out_buf[j, 0, 0]; j_lo1 = out_buf[j, 0, 1]; j_lo2 = out_buf[j, 0, 2]
            j_hi0 = out_buf[j, 1, 0]; j_hi1 = out_buf[j, 1, 1]; j_hi2 = out_buf[j, 1, 2]
            # j contains i if j_lo <= i_lo and i_hi <= j_hi (all dims).
            if (j_lo0 <= i_lo0 and j_lo1 <= i_lo1 and j_lo2 <= i_lo2
                    and i_hi0 <= j_hi0 and i_hi1 <= j_hi1 and i_hi2 <= j_hi2):
                # Mutual containment = identical. Keep earlier index (i < j).
                if (j_lo0 == i_lo0 and j_lo1 == i_lo1 and j_lo2 == i_lo2
                        and j_hi0 == i_hi0 and j_hi1 == i_hi1 and j_hi2 == i_hi2):
                    if j < i:
                        dominated = True
                        break
                else:
                    # Strict dominance.
                    dominated = True
                    break
        if not dominated:
            # Write i to position keep_count if different.
            if keep_count != i:
                out_buf[keep_count, 0, 0] = i_lo0
                out_buf[keep_count, 0, 1] = i_lo1
                out_buf[keep_count, 0, 2] = i_lo2
                out_buf[keep_count, 1, 0] = i_hi0
                out_buf[keep_count, 1, 1] = i_hi1
                out_buf[keep_count, 1, 2] = i_hi2
            keep_count += 1
    return keep_count


@njit(cache=True, fastmath=True)
def decode_njit(
    bps_order: np.ndarray,        # shape (N,) int64 — pre-sorted box indices
    n_rots_per_box: np.ndarray,   # shape (N,) int64
    dims_all: np.ndarray,         # shape (N, 6, 3) int64 — dx/dy/dz per rotation
    L: int, W: int, H: int,
    max_pallets: int,             # 1 for BR
    placements_out: np.ndarray,   # shape (N, 6) int64: bin_idx, rot_idx, x, y, z, packed
) -> int:
    """Decode a chromosome into placements. Returns number of bins used.

    placements_out[i] = (bin_idx, rot_idx, x, y, z, packed) for box bps_order[i]
    placed = 1 if packed, 0 if unpacked.
    """
    n = bps_order.shape[0]
    # EMS arrays per bin. We pre-allocate.
    MAX_BINS = max_pallets if max_pallets > 0 else 32
    bin_emss = np.zeros((MAX_BINS, MAX_EMS, 2, 3), dtype=np.int64)
    bin_ems_count = np.zeros(MAX_BINS, dtype=np.int64)
    n_bins = 0
    scratch = np.zeros((MAX_EMS, 2, 3), dtype=np.int64)

    for i in range(n):
        box_idx = bps_order[i]
        n_rots = n_rots_per_box[box_idx]
        # Try existing bins.
        placed = False
        best_bin = -1
        best_rot = -1
        best_x = 0
        best_y = 0
        best_z = 0
        best_score = -1
        # Try to fit in any existing bin, choose the one with max DFTRC across rotations.
        for b in range(n_bins):
            for r in range(n_rots):
                dx = dims_all[box_idx, r, 0]
                dy = dims_all[box_idx, r, 1]
                dz = dims_all[box_idx, r, 2]
                idx, x, y, z = find_best_dftrc_njit(
                    bin_emss[b], bin_ems_count[b], dx, dy, dz, L, W, H,
                )
                if idx < 0:
                    continue
                score = ((L - x - dx) * (L - x - dx)
                         + (W - y - dy) * (W - y - dy)
                         + (H - z - dz) * (H - z - dz))
                if score > best_score:
                    best_score = score
                    best_bin = b
                    best_rot = r
                    best_x = x
                    best_y = y
                    best_z = z
            if best_bin >= 0:
                break  # FFD-style: first fitting bin wins
        if best_bin >= 0:
            r = best_rot
            dx = dims_all[box_idx, r, 0]
            dy = dims_all[box_idx, r, 1]
            dz = dims_all[box_idx, r, 2]
            # Commit.
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
            # Find best rotation in new bin.
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
                    bin_emss[n_bins], bin_ems_count[n_bins], dx, dy, dz, L, W, H,
                )
                if idx < 0:
                    continue
                score = ((L - x - dx) * (L - x - dx)
                         + (W - y - dy) * (W - y - dy)
                         + (H - z - dz) * (H - z - dz))
                if score > best_score_n:
                    best_score_n = score
                    best_rot_n = r
                    best_x_n = x
                    best_y_n = y
                    best_z_n = z
            if best_rot_n < 0:
                # Doesn't fit even in empty bin.
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


# Python-level wrappers ------------------------------------------------------

def precompute_box_dims(boxes: List[Box]) -> tuple:
    """Compute the (n_rots, dims) arrays for the JIT decoder."""
    n = len(boxes)
    n_rots_arr = np.zeros(n, dtype=np.int64)
    dims_all = np.zeros((n, 6, 3), dtype=np.int64)
    for i, b in enumerate(boxes):
        n_rots_arr[i] = len(b.allowed_rotations)
        for r_idx, r in enumerate(b.allowed_rotations):
            dx, dy, dz = b.dims_for(r)
            dims_all[i, r_idx, 0] = int(round(dx))
            dims_all[i, r_idx, 1] = int(round(dy))
            dims_all[i, r_idx, 2] = int(round(dz))
    return n_rots_arr, dims_all


def decode_chromosome_fast(
    chrom: np.ndarray,
    boxes: List[Box],
    pallet: Pallet,
    config: PackerConfig,
    n_rots_arr: np.ndarray,
    dims_all: np.ndarray,
    max_pallets: int = 1,
) -> PackResult:
    """Wrapper: random keys → BPS order → JIT decoder → PackResult."""
    n = len(boxes)
    if n == 0:
        return PackResult(pallets=[], unpacked=[])
    bps = chrom[:n]
    order = np.argsort(bps).astype(np.int64)

    L = int(round(pallet.length))
    W = int(round(pallet.width))
    H = int(round(pallet.height))

    placements_out = np.zeros((n, 6), dtype=np.int64)
    n_bins = decode_njit(order, n_rots_arr, dims_all, L, W, H,
                         max_pallets, placements_out)

    # Build PackResult.
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


def _fitness_pallet1(result: PackResult, pallet: Pallet) -> float:
    """Lower = better. For max_pallets=1: minimize 1 - util_pallet1."""
    cap = pallet.length * pallet.width * pallet.height
    if not result.pallets:
        return 1.0
    used = sum(p.box.volume for p in result.pallets[0].placements)
    return 1.0 - (used / cap if cap > 0 else 0.0)


def brkga_pack_fast(
    boxes: List[Box],
    pallet: Pallet,
    config: PackerConfig,
    *,
    population_size: int = 500,
    generations: int = 200,
    elite_fraction: float = 0.25,
    mutant_fraction: float = 0.20,
    p_elite_inherit: float = 0.70,
    time_limit_s: float = 30.0,
    max_pallets: int = 1,
    seed: int = 42,
    patience: int = 30,
    n_populations: int = 3,
    migration_interval: int = 10,
    migrants_per_swap: int = 3,
    verbose: bool = False,
) -> PackResult:
    """Multi-population BRKGA with JIT-compiled decoder.

    Defaults are tuned for BR-scale problems with a 30s/instance budget.
    """
    n = len(boxes)
    if n == 0:
        return PackResult(pallets=[], unpacked=[])

    n_rots_arr, dims_all = precompute_box_dims(boxes)

    K = max(1, n_populations)
    P = max(10, population_size // K) * K
    pop_size = P // K
    n_elite = max(1, int(pop_size * elite_fraction))
    n_mutant = max(1, int(pop_size * mutant_fraction))
    n_cross = max(0, pop_size - n_elite - n_mutant)

    rngs = [np.random.default_rng(seed + i) for i in range(K)]
    pops = [rngs[i].random((pop_size, 2 * n)) for i in range(K)]

    best_fitness = float('inf')
    best_result: Optional[PackResult] = None
    gens_no_improve = 0
    t0 = time.time()
    total_decodes = 0

    pop_fits = [np.zeros(pop_size) for _ in range(K)]

    for gen in range(generations):
        if time.time() - t0 > time_limit_s:
            if verbose:
                print(f"  [fast-BRKGA] gen {gen}: time hit, decodes={total_decodes}")
            break

        for k in range(K):
            fits = pop_fits[k]
            for i in range(pop_size):
                res = decode_chromosome_fast(
                    pops[k][i], boxes, pallet, config,
                    n_rots_arr, dims_all, max_pallets=max_pallets,
                )
                fits[i] = _fitness_pallet1(res, pallet)
                total_decodes += 1
                if fits[i] < best_fitness - 1e-9:
                    best_fitness = float(fits[i])
                    best_result = res
                    gens_no_improve = 0
                    if verbose:
                        print(f"  [fast-BRKGA] gen {gen} pop {k}: util={(1-best_fitness)*100:.2f}%")

        gens_no_improve += 1
        if gens_no_improve >= patience:
            if verbose:
                print(f"  [fast-BRKGA] gen {gen}: patience, decodes={total_decodes}")
            break

        # Migration.
        if K > 1 and gen > 0 and gen % migration_interval == 0:
            for k in range(K):
                src = k
                dst = (k + 1) % K
                src_sorted = np.argsort(pop_fits[src])[:migrants_per_swap]
                dst_sorted = np.argsort(pop_fits[dst])[-migrants_per_swap:]
                for s, d in zip(src_sorted, dst_sorted):
                    pops[dst][d] = pops[src][s].copy()

        # Evolve.
        new_pops = []
        for k in range(K):
            sorted_idx = np.argsort(pop_fits[k])
            new_pop = np.zeros_like(pops[k])
            new_pop[:n_elite] = pops[k][sorted_idx[:n_elite]]
            new_pop[n_elite:n_elite + n_mutant] = rngs[k].random((n_mutant, 2 * n))
            elite_pool = pops[k][sorted_idx[:n_elite]]
            non_elite_pool = pops[k][sorted_idx[n_elite:]]
            if n_cross > 0:
                pa_idx = rngs[k].integers(0, n_elite, size=n_cross)
                pb_idx = rngs[k].integers(0, len(non_elite_pool), size=n_cross)
                pa_parents = elite_pool[pa_idx]
                pb_parents = non_elite_pool[pb_idx]
                mask = rngs[k].random((n_cross, 2 * n)) < p_elite_inherit
                new_pop[n_elite + n_mutant:] = np.where(mask, pa_parents, pb_parents)
            new_pops.append(new_pop)
        pops = new_pops

    if best_result is None:
        return PackResult(pallets=[], unpacked=list(boxes))
    return best_result
