"""
ab_test_decoder_precomputed.py — Phase 3d A/B test:
Cython decode_precomputed_blocks_njit_mode vs Numba reference.

Coverage:
  - homogeneous (skus=1) with big precomputed block (Phase 1 fires)
  - mixed (skus=4) with moderate blocks (mix of Phase 1 + Phase 2 fallback)
  - heterogeneous (skus=30) with (1,1,1) blocks (Phase 1 always short-circuits)
  - degenerate (skus=1) where precomputed block > pallet (Phase 1 skips,
    Phase 2 fallback always runs)

Run inside Docker:

    docker run --rm -v "$(pwd)":/app pallet-packer:dev \
        python scripts/ab_test_decoder_precomputed.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from pallet_packer._brkga_core import jit_decoders_geom as nb
from pallet_packer._brkga_core import jit_decoders_geom_cy as cy


def make_random_instance(rng, n_boxes, n_skus, L, W, H):
    """Random box catalog + SKU assignment + BPS permutation."""
    n_rots_per_box = np.full(n_boxes, 6, dtype=np.int64)
    dims_all = np.zeros((n_boxes, 6, 3), dtype=np.int64)
    sku_id_per_box = rng.integers(0, n_skus, size=n_boxes).astype(np.int64)
    sku_base = {}
    for s in range(n_skus):
        sku_base[s] = (
            int(rng.integers(40, max(41, L // 4))),
            int(rng.integers(40, max(41, W // 4))),
            int(rng.integers(40, max(41, H // 4))),
        )
    for i in range(n_boxes):
        bx, by, bz = sku_base[int(sku_id_per_box[i])]
        rots = [
            (bx, by, bz), (bx, bz, by),
            (by, bx, bz), (by, bz, bx),
            (bz, bx, by), (bz, by, bx),
        ]
        for r in range(6):
            dims_all[i, r] = rots[r]
    bps_order = rng.permutation(n_boxes).astype(np.int64)
    return bps_order, n_rots_per_box, dims_all, sku_id_per_box, sku_base


def make_sku_best_block(rng, n_skus, sku_base, L, W, H, max_k=4,
                        oversize_chance=0.0):
    """Pre-computed (k, l, m, rot) per SKU. With oversize_chance>0, some
    SKUs get blocks that exceed pallet dims — exercising the Phase 1 skip /
    Phase 2 fallback path."""
    out = np.zeros((n_skus, 4), dtype=np.int64)
    for s in range(n_skus):
        bx, by, bz = sku_base[s]
        if rng.random() < oversize_chance:
            # Force-oversized so Phase 1 skips
            k = max(2, L // bx + 1)
            l = max(2, W // by + 1)
            m = max(2, H // bz + 1)
        else:
            k = int(rng.integers(1, max(2, min(max_k, L // bx) + 1)))
            l = int(rng.integers(1, max(2, min(max_k, W // by) + 1)))
            m = int(rng.integers(1, max(2, min(max_k, H // bz) + 1)))
        rot = int(rng.integers(0, 6))
        out[s] = (k, l, m, rot)
    return out


def ab_one(rng, n_boxes, n_skus, L, W, H, max_pallets, oversize_chance=0.0):
    bps, n_rots, dims, sku_ids, sku_base = make_random_instance(
        rng, n_boxes, n_skus, L, W, H)
    sku_bb = make_sku_best_block(
        rng, n_skus, sku_base, L, W, H, oversize_chance=oversize_chance)
    po_nb = np.zeros((n_boxes, 6), dtype=np.int64)
    po_cy = np.zeros((n_boxes, 6), dtype=np.int64)
    nb_bins = nb.decode_precomputed_blocks_njit_mode(
        bps, n_rots, dims, sku_ids, sku_bb, L, W, H,
        max_pallets, po_nb, n_skus)
    cy_bins = cy.decode_precomputed_blocks_njit_mode(
        bps, n_rots, dims, sku_ids, sku_bb, L, W, H,
        max_pallets, po_cy, n_skus)
    same = np.array_equal(po_nb, po_cy) and (nb_bins == cy_bins)
    diff = 0 if same else int(np.sum(np.any(po_nb != po_cy, axis=1)))
    return same, nb_bins, cy_bins, diff


def ab_test_suite(rng, n_cases=30):
    """Cover the Phase 1 / Phase 2 / Phase 3 branch combinations."""
    cases = [
        # (n_boxes, n_skus, L, W, H, max_pallets, oversize_chance, label)
        (60,   1, 800, 600, 500, 4, 0.0,  "homog skus=1 normal-block"),
        (120,  1, 1000, 800, 600, 4, 0.0, "homog skus=1 normal-block"),
        (80,   4, 1000, 800, 600, 4, 0.0, "mixed skus=4 normal-block"),
        (150,  4, 1200, 1000, 700, 6, 0.0, "mixed skus=4 normal-block"),
        (60,  30, 800, 600, 500, 4, 0.0,  "heterogeneous skus=30"),
        (50,   1, 800, 600, 500, 4, 1.0,  "homog skus=1 OVERSIZE block"),
        (80,   4, 1000, 800, 600, 4, 0.5, "mixed skus=4 50%-oversize"),
    ]
    print("=== A/B correctness — decode_precomputed_blocks_njit_mode ===")
    fail = 0
    for n_boxes, n_skus, L, W, H, mp, ovsz, label in cases:
        per_fail = 0
        for _ in range(n_cases):
            ok, nb_bins, cy_bins, diff = ab_one(
                rng, n_boxes, n_skus, L, W, H, mp,
                oversize_chance=ovsz)
            if not ok:
                per_fail += 1
                if per_fail <= 2:
                    print(f"  MISMATCH {label}: "
                          f"nb_bins={nb_bins} cy_bins={cy_bins} diff_rows={diff}")
        fail += per_fail
        tag = "OK" if per_fail == 0 else f"FAIL ({per_fail})"
        print(f"  n={n_boxes:3d}  {label:38s}  {tag}")
    if fail:
        print(f"\n{fail} mismatches — STOP and diff.")
        sys.exit(1)
    print(f"\nAll {len(cases) * n_cases} cases bit-identical.")


def microbenchmark(rng):
    """End-to-end decode timing on a fixed representative instance."""
    print("\n=== Microbenchmark (1000 repeats) ===")
    n_boxes, n_skus, L, W, H, mp = 100, 4, 1000, 800, 600, 6
    bps, n_rots, dims, sku_ids, sku_base = make_random_instance(
        rng, n_boxes, n_skus, L, W, H)
    sku_bb = make_sku_best_block(rng, n_skus, sku_base, L, W, H)
    po_nb = np.zeros((n_boxes, 6), dtype=np.int64)
    po_cy = np.zeros((n_boxes, 6), dtype=np.int64)
    for _ in range(20):
        nb.decode_precomputed_blocks_njit_mode(
            bps, n_rots, dims, sku_ids, sku_bb, L, W, H, mp, po_nb, n_skus)
        cy.decode_precomputed_blocks_njit_mode(
            bps, n_rots, dims, sku_ids, sku_bb, L, W, H, mp, po_cy, n_skus)
    n_repeat = 1000
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        nb.decode_precomputed_blocks_njit_mode(
            bps, n_rots, dims, sku_ids, sku_bb, L, W, H, mp, po_nb, n_skus)
    t_nb = (time.perf_counter() - t0) / n_repeat
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        cy.decode_precomputed_blocks_njit_mode(
            bps, n_rots, dims, sku_ids, sku_bb, L, W, H, mp, po_cy, n_skus)
    t_cy = (time.perf_counter() - t0) / n_repeat
    speedup = t_nb / t_cy if t_cy > 0 else float('inf')
    print(f"  decode_precomputed (n={n_boxes}, skus={n_skus}): "
          f"numba {t_nb*1e6:.1f} µs/call  "
          f"cython {t_cy*1e6:.1f} µs/call  "
          f"speedup {speedup:.2f}×")


def main():
    rng = np.random.default_rng(43)
    ab_test_suite(rng, n_cases=30)
    microbenchmark(rng)


if __name__ == "__main__":
    main()
