"""
pallet_packer._brkga_core.chromosome — chromosome construction helpers.

  chromosome_from_order        build a chromosome whose argsort yields
                               the given BPS order
  make_informed_chromosomes    smart-init: vol-desc, sku-grouped, etc.
  chromosome_from_v2_result    convert a v2 PalletPacker result into a
                               chromosome (used for the v2 warm-start
                               seed in Phase 0 of brkga_pack_v35)
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..models import Box, Pallet, Rotation
from ..packer import PackResult


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
