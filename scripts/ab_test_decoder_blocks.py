"""
ab_test_decoder_blocks.py — Phase 3c A/B test: Cython
decode_blocks_njit_mode + find_best_block_at_pos_njit vs Numba reference.

Block extension only fires when consecutive same-SKU boxes appear in BPS
order, so the suite covers three SKU-concentration regimes:
  - homogeneous  (n_skus=1)   — block extension fires constantly (BR1/BR3-like)
  - mixed        (n_skus=4)   — partial block runs
  - heterogeneous(n_skus=n/2) — block extension degenerates to k=l=m=1 (BR7-like)

Run inside Docker:

    docker run --rm -v "$(pwd)":/app pallet-packer:dev \
        python scripts/ab_test_decoder_blocks.py
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
    """Generate a random box catalog with SKU assignment + BPS permutation."""
    n_rots_per_box = np.full(n_boxes, 6, dtype=np.int64)
    dims_all = np.zeros((n_boxes, 6, 3), dtype=np.int64)
    sku_id_per_box = rng.integers(0, n_skus, size=n_boxes).astype(np.int64)

    # All boxes within a SKU get the same base dims — this is what makes
    # block extension actually accumulate.
    sku_dims = {}
    for s in range(n_skus):
        sku_dims[s] = (
            int(rng.integers(40, max(41, L // 4))),
            int(rng.integers(40, max(41, W // 4))),
            int(rng.integers(40, max(41, H // 4))),
        )
    for i in range(n_boxes):
        bx, by, bz = sku_dims[int(sku_id_per_box[i])]
        rots = [
            (bx, by, bz), (bx, bz, by),
            (by, bx, bz), (by, bz, bx),
            (bz, bx, by), (bz, by, bx),
        ]
        for r in range(6):
            dims_all[i, r] = rots[r]
    bps_order = rng.permutation(n_boxes).astype(np.int64)
    return bps_order, n_rots_per_box, dims_all, sku_id_per_box


def ab_one(rng, n_boxes, n_skus, L, W, H, max_pallets):
    bps, n_rots, dims, sku_ids = make_random_instance(
        rng, n_boxes, n_skus, L, W, H)
    po_nb = np.zeros((n_boxes, 6), dtype=np.int64)
    po_cy = np.zeros((n_boxes, 6), dtype=np.int64)
    nb_bins = nb.decode_blocks_njit_mode(
        bps, n_rots, dims, sku_ids, L, W, H, max_pallets, po_nb, n_skus)
    cy_bins = cy.decode_blocks_njit_mode(
        bps, n_rots, dims, sku_ids, L, W, H, max_pallets, po_cy, n_skus)
    same = np.array_equal(po_nb, po_cy) and (nb_bins == cy_bins)
    diff = 0 if same else int(np.sum(np.any(po_nb != po_cy, axis=1)))
    return same, nb_bins, cy_bins, diff


def ab_test_decoder(rng, n_cases=30):
    """Run the decoder A/B suite across SKU regimes + instance sizes."""
    # (n_boxes, n_skus, L, W, H, max_pallets)
    cases = [
        # Homogeneous (BR1/BR3-like)
        (50,   1, 800, 600, 500, 4),
        (100,  1, 1000, 800, 600, 4),
        (200,  1, 1200, 1000, 700, 6),
        # Mixed
        (80,   4, 1000, 800, 600, 4),
        (150,  4, 1200, 1000, 700, 6),
        # Heterogeneous (BR7-like)
        (60,  30, 800, 600, 500, 4),
        (120, 60, 1200, 1000, 700, 6),
    ]
    print("=== A/B correctness — decode_blocks_njit_mode ===")
    fail = 0
    for n_boxes, n_skus, L, W, H, mp in cases:
        per_fail = 0
        for _ in range(n_cases):
            ok, nb_bins, cy_bins, diff = ab_one(
                rng, n_boxes, n_skus, L, W, H, mp)
            if not ok:
                per_fail += 1
                if per_fail <= 2:
                    print(f"  MISMATCH n={n_boxes} skus={n_skus}: "
                          f"nb_bins={nb_bins} cy_bins={cy_bins} diff_rows={diff}")
        fail += per_fail
        tag = "OK" if per_fail == 0 else f"FAIL ({per_fail})"
        regime = ("homogeneous" if n_skus == 1
                  else "mixed" if n_skus <= 10
                  else "heterogeneous")
        print(f"  n={n_boxes:3d}  skus={n_skus:3d}  L,W,H={L},{W},{H}  "
              f"({regime:13s}): {n_cases} cases  {tag}")
    if fail:
        print(f"\n{fail} mismatches across decoder suite — STOP and diff.")
        sys.exit(1)
    print(f"\nAll {len(cases) * n_cases} decoder cases bit-identical.")


def ab_test_helper(rng, n_calls=200):
    """A/B test the find_best_block_at_pos helper directly."""
    print("\n=== A/B correctness — find_best_block_at_pos ===")
    L, W, H = 1000, 800, 600
    fail = 0
    for _ in range(n_calls):
        n_ems = int(rng.integers(1, 64))
        # Generate plausible EMS array of size 128 (matches MAX_EMS layout).
        emss = np.zeros((128, 2, 3), dtype=np.int64)
        for i in range(n_ems):
            mn = rng.integers(0, L // 2, size=3)
            mx = mn + rng.integers(50, max(51, L // 2), size=3)
            mx[0] = min(mx[0], L)
            mx[1] = min(mx[1], W)
            mx[2] = min(mx[2], H)
            emss[i, 0] = mn
            emss[i, 1] = mx
        # Pick a position inside one of the EMSs so the function has work.
        ei = int(rng.integers(0, n_ems))
        x = int(emss[ei, 0, 0])
        y = int(emss[ei, 0, 1])
        z = int(emss[ei, 0, 2])
        dx = int(rng.integers(20, 100))
        dy = int(rng.integers(20, 100))
        dz = int(rng.integers(20, 100))
        max_count = int(rng.integers(1, 50))
        r_nb = nb.find_best_block_at_pos_njit(
            emss, n_ems, x, y, z, dx, dy, dz, max_count)
        r_cy = cy.find_best_block_at_pos_njit(
            emss, n_ems, x, y, z, dx, dy, dz, max_count)
        if r_nb != r_cy:
            fail += 1
            if fail <= 3:
                print(f"  MISMATCH: nb={r_nb} cy={r_cy} "
                      f"pos=({x},{y},{z}) box=({dx},{dy},{dz}) max={max_count}")
    if fail:
        print(f"\n{fail} mismatches — STOP and diff.")
        sys.exit(1)
    print(f"  {n_calls} calls all bit-identical ✓")


def microbenchmark(rng):
    """End-to-end decode timing on a fixed representative instance."""
    print("\n=== Microbenchmark (1000 repeats) ===")
    n_boxes, n_skus, L, W, H, mp = 100, 4, 1000, 800, 600, 4
    bps, n_rots, dims, sku_ids = make_random_instance(
        rng, n_boxes, n_skus, L, W, H)
    po_nb = np.zeros((n_boxes, 6), dtype=np.int64)
    po_cy = np.zeros((n_boxes, 6), dtype=np.int64)
    for _ in range(20):
        nb.decode_blocks_njit_mode(
            bps, n_rots, dims, sku_ids, L, W, H, mp, po_nb, n_skus)
        cy.decode_blocks_njit_mode(
            bps, n_rots, dims, sku_ids, L, W, H, mp, po_cy, n_skus)
    n_repeat = 1000
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        nb.decode_blocks_njit_mode(
            bps, n_rots, dims, sku_ids, L, W, H, mp, po_nb, n_skus)
    t_nb = (time.perf_counter() - t0) / n_repeat
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        cy.decode_blocks_njit_mode(
            bps, n_rots, dims, sku_ids, L, W, H, mp, po_cy, n_skus)
    t_cy = (time.perf_counter() - t0) / n_repeat
    speedup = t_nb / t_cy if t_cy > 0 else float('inf')
    print(f"  decode_blocks (n={n_boxes}, skus={n_skus}): "
          f"numba {t_nb*1e6:.1f} µs/call  "
          f"cython {t_cy*1e6:.1f} µs/call  "
          f"speedup {speedup:.2f}×")


def main():
    rng = np.random.default_rng(43)
    ab_test_decoder(rng, n_cases=30)
    ab_test_helper(rng, n_calls=200)
    microbenchmark(rng)


if __name__ == "__main__":
    main()
