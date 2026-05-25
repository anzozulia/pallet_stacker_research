"""
pallet_packer._brkga_core.polish — post-BRKGA improvement phases.

  local_search_2opt   position-based BPS swaps + segment reverse + insert
                      + rotation flip + decoder-mode flip
  path_relinking      walk from elite #1 to elite #2 by batch key transfer,
                      keep any improving intermediate
  lns_polish          destroy K BPS keys + repair via decoder; escapes
                      local optima the swap-based LS can't reach

All three operate on a single best chromosome from main BRKGA and route
through decode_auto_mode (with full constraint forwarding) so the polish
phase respects the same constraint stack as the search.
"""
from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np

from ..models import Box, Pallet, PackerConfig
from ..packer import PackResult
from ..brkga_v3_fast import _fitness_pallet1
from .precompute import _NO_LIMIT
from .dispatch import decode_chromosome, decode_auto_mode


def local_search_2opt(
    best_chrom: np.ndarray,
    boxes: List[Box], pallet: Pallet, config: PackerConfig,
    n_rots_arr: np.ndarray, dims_all: np.ndarray,
    *,
    time_budget_s: float = 3.0,
    multi_decoder: bool = False,
    n_modes: int = 5,
    sku_id_per_box: Optional[np.ndarray] = None,
    sku_best_block: Optional[np.ndarray] = None,
    sku_top_k_blocks: Optional[np.ndarray] = None,
    mode_cdf: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    mlot: Optional[np.ndarray] = None,
    pallet_max_weight: float = _NO_LIMIT,
    has_constraints: bool = False,
    support_ratio: float = 0.0,
    rfs: Optional[np.ndarray] = None,
    require_centroid: int = 0,
    cog_x_min: float = -1e18, cog_x_max: float = 1e18,
    cog_y_min: float = -1e18, cog_y_max: float = 1e18,
    cog_min_load_frac: float = 0.0,
    cog_active: int = 0,
    max_overhang: float = 0.0,
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
            sku_id_per_box=sku_id_per_box,
            sku_best_block=sku_best_block,
            sku_top_k_blocks=sku_top_k_blocks,
            mode_cdf=mode_cdf,
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
    else:
        decoder = lambda c, b, p, cfg, na, da, max_pallets=1: decode_chromosome(
            c, b, p, cfg, na, da, mode=0, max_pallets=max_pallets,
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
    sku_best_block: Optional[np.ndarray] = None,
    sku_top_k_blocks: Optional[np.ndarray] = None,
    mode_cdf: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    mlot: Optional[np.ndarray] = None,
    pallet_max_weight: float = _NO_LIMIT,
    has_constraints: bool = False,
    support_ratio: float = 0.0,
    rfs: Optional[np.ndarray] = None,
    require_centroid: int = 0,
    cog_x_min: float = -1e18, cog_x_max: float = 1e18,
    cog_y_min: float = -1e18, cog_y_max: float = 1e18,
    cog_min_load_frac: float = 0.0,
    cog_active: int = 0,
    max_overhang: float = 0.0,
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
            sku_id_per_box=sku_id_per_box,
            sku_best_block=sku_best_block,
            sku_top_k_blocks=sku_top_k_blocks,
            mode_cdf=mode_cdf,
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
    else:
        decoder = lambda c, b, p, cfg, na, da, max_pallets=1: decode_chromosome(
            c, b, p, cfg, na, da, mode=0, max_pallets=max_pallets,
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
    sku_best_block: Optional[np.ndarray] = None,
    sku_top_k_blocks: Optional[np.ndarray] = None,
    mode_cdf: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    mlot: Optional[np.ndarray] = None,
    pallet_max_weight: float = _NO_LIMIT,
    has_constraints: bool = False,
    support_ratio: float = 0.0,
    rfs: Optional[np.ndarray] = None,
    require_centroid: int = 0,
    cog_x_min: float = -1e18, cog_x_max: float = 1e18,
    cog_y_min: float = -1e18, cog_y_max: float = 1e18,
    cog_min_load_frac: float = 0.0,
    cog_active: int = 0,
    max_overhang: float = 0.0,
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
            sku_id_per_box=sku_id_per_box,
            sku_best_block=sku_best_block,
            sku_top_k_blocks=sku_top_k_blocks,
            mode_cdf=mode_cdf,
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
    else:
        decoder = lambda c, b, p, cfg, na, da, max_pallets=1: decode_chromosome(
            c, b, p, cfg, na, da, mode=0, max_pallets=max_pallets,
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
