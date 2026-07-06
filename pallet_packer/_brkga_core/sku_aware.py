"""
pallet_packer._brkga_core.sku_aware — experimental SKU-aware mini-BRKGA.

Default OFF. Compresses the chromosome from 2N + 1 to 2S + 1 keys
(S = number of distinct SKUs), runs a small BRKGA over this shorter
search space, and seeds the main BRKGA with the best result.

Empirically neutral-to-negative on BR (mean Δ ≈ 0 pp; restricts the
search to SKU-grouped BPS orderings which sometimes sacrifices density
on interleaved-SKU optima). Kept as opt-in for cases with very few SKUs
or very tight budgets. See docs/reports/17_v37_sku_aware.md.

  compute_sku_groups        list of (sku_id, box_indices) for each SKU
  sku_chrom_to_full_chrom   2S+1 SKU chromosome → 2N+1 full chromosome
  brkga_sku_aware_search    mini-BRKGA driver
"""
from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np

from ..models import Box, Pallet, PackerConfig
from ..packer import PackResult
from ..brkga_v3_fast import _fitness_pallet1
from .dispatch import decode_chromosome


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
    realism=None,   # RealismContext — keeps this ranker on the driver's scale
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
            fits[i] = _fitness_pallet1(res, pallet, realism=realism,
                                       max_pallets=max_pallets)
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
