"""
ab_test_decoder_layer.py — Phase 3b A/B test: Cython decode_layer_njit vs
the Numba reference.

Generates random BPS orderings + box dim sets across several pallet sizes,
runs both implementations, asserts the entire placements_out array is
np.array_equal, and reports a microbenchmark for end-to-end decode timing.

Run inside Docker:

    docker run --rm -v "$(pwd)":/app pallet-packer:dev \
        python scripts/ab_test_decoder_layer.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from pallet_packer._brkga_core import jit_decoders_geom as nb
from pallet_packer._brkga_core import jit_decoders_geom_cy as cy


def make_random_instance(rng, n_boxes, L, W, H):
    """Generate a random box catalog + BPS permutation for the layer decoder.

    Each box gets up to 6 axis-aligned rotations (pre-expanded), matching
    the dims_all layout the BRKGA pipeline produces.
    """
    n_rots_per_box = np.full(n_boxes, 6, dtype=np.int64)
    dims_all = np.zeros((n_boxes, 6, 3), dtype=np.int64)
    for i in range(n_boxes):
        # Pick base dims that comfortably fit the pallet
        bx = int(rng.integers(40, max(41, L // 4)))
        by = int(rng.integers(40, max(41, W // 4)))
        bz = int(rng.integers(40, max(41, H // 4)))
        rots = [
            (bx, by, bz), (bx, bz, by),
            (by, bx, bz), (by, bz, bx),
            (bz, bx, by), (bz, by, bx),
        ]
        for r in range(6):
            dims_all[i, r] = rots[r]
    bps_order = rng.permutation(n_boxes).astype(np.int64)
    return bps_order, n_rots_per_box, dims_all


def ab_one(rng, n_boxes, L, W, H, max_pallets):
    """Run one A/B comparison; returns (ok, n_bins_nb, n_bins_cy, diff_count)."""
    bps, n_rots, dims = make_random_instance(rng, n_boxes, L, W, H)
    po_nb = np.zeros((n_boxes, 6), dtype=np.int64)
    po_cy = np.zeros((n_boxes, 6), dtype=np.int64)
    nb_bins = nb.decode_layer_njit(bps, n_rots, dims, L, W, H, max_pallets, po_nb)
    cy_bins = cy.decode_layer_njit(bps, n_rots, dims, L, W, H, max_pallets, po_cy)
    same = np.array_equal(po_nb, po_cy) and (nb_bins == cy_bins)
    diff = 0 if same else int(np.sum(np.any(po_nb != po_cy, axis=1)))
    return same, nb_bins, cy_bins, diff


def ab_test_suite(rng, n_cases=40):
    """Run the full A/B suite across varied instance sizes."""
    sizes = [
        # (n_boxes, L, W, H, max_pallets)
        (20,  500, 400, 300, 4),
        (50,  800, 600, 500, 4),
        (80,  1000, 800, 600, 6),
        (120, 1200, 1000, 700, 8),
        (200, 1500, 1100, 800, 8),
    ]
    print("=== A/B correctness ===")
    fail = 0
    for size in sizes:
        n_boxes, L, W, H, max_pallets = size
        per_size_fail = 0
        for _ in range(n_cases):
            ok, nb_bins, cy_bins, diff = ab_one(
                rng, n_boxes, L, W, H, max_pallets)
            if not ok:
                per_size_fail += 1
                if per_size_fail <= 2:
                    print(f"  MISMATCH n={n_boxes} L,W,H={L},{W},{H}: "
                          f"nb_bins={nb_bins} cy_bins={cy_bins} "
                          f"differing_rows={diff}")
        fail += per_size_fail
        tag = "OK" if per_size_fail == 0 else f"FAIL ({per_size_fail})"
        print(f"  n={n_boxes:3d}  L,W,H={L},{W},{H}: {n_cases} cases  {tag}")
    if fail:
        print(f"\n{fail} mismatches across the suite — STOP and diff.")
        sys.exit(1)
    print(f"\nAll {len(sizes) * n_cases} cases bit-identical.")


def microbenchmark(rng):
    """End-to-end decode timing on a single fixed representative instance."""
    print("\n=== Microbenchmark (single decode, 1000 repeats) ===")
    # Mid-size representative case.
    n_boxes, L, W, H, max_pallets = 100, 1000, 800, 600, 6
    bps, n_rots, dims = make_random_instance(rng, n_boxes, L, W, H)
    po_nb = np.zeros((n_boxes, 6), dtype=np.int64)
    po_cy = np.zeros((n_boxes, 6), dtype=np.int64)
    # Warm up JIT on both sides.
    for _ in range(20):
        nb.decode_layer_njit(bps, n_rots, dims, L, W, H, max_pallets, po_nb)
        cy.decode_layer_njit(bps, n_rots, dims, L, W, H, max_pallets, po_cy)
    n_repeat = 1000
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        nb.decode_layer_njit(bps, n_rots, dims, L, W, H, max_pallets, po_nb)
    t_nb = (time.perf_counter() - t0) / n_repeat
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        cy.decode_layer_njit(bps, n_rots, dims, L, W, H, max_pallets, po_cy)
    t_cy = (time.perf_counter() - t0) / n_repeat
    speedup = t_nb / t_cy if t_cy > 0 else float('inf')
    print(f"  decode_layer (n={n_boxes}): "
          f"numba {t_nb*1e6:.1f} µs/call  "
          f"cython {t_cy*1e6:.1f} µs/call  "
          f"speedup {speedup:.2f}×")


def main():
    rng = np.random.default_rng(43)  # canonical seed
    ab_test_suite(rng, n_cases=40)
    microbenchmark(rng)


if __name__ == "__main__":
    main()
