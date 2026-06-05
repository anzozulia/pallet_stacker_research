"""
pallet_packer._brkga_core.blocks — block enumeration (Bischoff 2002).

  enumerate_top_k_blocks_per_sku   top-K candidate (k, l, m, rot) blocks
                                   per SKU (v3.8 BR1 improvement)
  resolve_sku_blocks_from_chrom    chromosome's VBO keys → chosen block
                                   per SKU (mode-5 dispatch)
  enumerate_best_block_per_sku     single best block per SKU (v3.7 fallback)

These compute block candidates upfront so the mode-5 decoder can place
multi-SKU blocks at literature-recommended sizes.
"""
from __future__ import annotations

from typing import List

import numpy as np

from ..models import Box


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
        # Sort by count desc, then by floor footprint (k*l) desc, so that the
        # one shape kept per count is the FLATTEST (floor-first — avoids the
        # candidate set offering only corner-tower shapes for a given count).
        candidates.sort(key=lambda c: (-c[0], -(c[1] * c[2])))
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
        # Floor-first: pick the block with the largest floor footprint (k*l),
        # then the most boxes (count) within it — mirrors
        # find_best_block_at_pos_njit so the precomputed fallback also spreads
        # on the floor instead of building corner towers.
        best_fp = 1
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
                    fp = k * l
                    if fp > count:
                        break
                    m = max_m
                    if fp * m > count:
                        m = count // fp
                    n = fp * m
                    if fp > best_fp or (fp == best_fp and n > best_n):
                        best_fp = fp
                        best_n = n
                        best_k, best_l, best_m, best_rot = k, l, m, r
        best_blocks[sku, 0] = best_k
        best_blocks[sku, 1] = best_l
        best_blocks[sku, 2] = best_m
        best_blocks[sku, 3] = best_rot
    return best_blocks

