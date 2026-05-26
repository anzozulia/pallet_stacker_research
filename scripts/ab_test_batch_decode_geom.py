"""
ab_test_batch_decode_geom.py — Phase 5a A/B + benchmark:
decode_batch_njit_mode vs serial decode_njit_mode.

Two gates:
  1. Correctness: batch[i] must be bit-identical to serial(chromosomes[i]).
  2. Speed: time pop_size=80 decodes serial vs batch (uses all cores).

The batch path produces n_bins via an output buffer; the bit-identical
contract is on the entire placements_out_all array.

Run inside Docker (controls OMP_NUM_THREADS to demonstrate scaling):
    docker run --rm -v "$(pwd)":/app -w /app pallet-packer:dev \
        python scripts/ab_test_batch_decode_geom.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from pallet_packer._brkga_core import jit_decoders_geom_cy as cy


def make_random_population(rng, pop_size, n_boxes, L, W, H):
    """Generate a population of pop_size chromosomes + shared box catalog."""
    n_rots_per_box = np.full(n_boxes, 6, dtype=np.int64)
    dims_all = np.zeros((n_boxes, 6, 3), dtype=np.int64)
    for i in range(n_boxes):
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
    # BRKGA chromosomes are random keys in [0, 1).
    chromosomes = rng.random((pop_size, n_boxes)).astype(np.float64)
    return chromosomes, n_rots_per_box, dims_all


def ab_correctness(rng, pop_size, n_boxes, L, W, H, max_pallets, mode):
    """Compare batch output vs serial output, chromosome by chromosome."""
    chroms, n_rots, dims = make_random_population(
        rng, pop_size, n_boxes, L, W, H)

    # Batch path.
    po_batch = np.zeros((pop_size, n_boxes, 6), dtype=np.int64)
    nb_batch = np.zeros(pop_size, dtype=np.int64)
    cy.decode_batch_njit_mode(
        chroms, n_rots, dims, L, W, H, max_pallets, mode,
        po_batch, nb_batch,
    )

    # Serial path — replicate exactly what the wrapper does (argsort + decode).
    po_serial = np.zeros((pop_size, n_boxes, 6), dtype=np.int64)
    nb_serial = np.zeros(pop_size, dtype=np.int64)
    for i in range(pop_size):
        order = np.argsort(chroms[i]).astype(np.int64)
        nb_serial[i] = cy.decode_njit_mode(
            order, n_rots, dims, L, W, H, max_pallets, po_serial[i], mode,
        )

    same = np.array_equal(po_batch, po_serial) and np.array_equal(nb_batch, nb_serial)
    diff_chroms = 0
    if not same:
        for i in range(pop_size):
            if not np.array_equal(po_batch[i], po_serial[i]):
                diff_chroms += 1
    return same, diff_chroms


def benchmark(rng, pop_size, n_boxes, L, W, H, max_pallets, mode, n_runs=10):
    """Time pop_size decodes via batch (parallel) vs serial loop."""
    chroms, n_rots, dims = make_random_population(
        rng, pop_size, n_boxes, L, W, H)
    po_batch = np.zeros((pop_size, n_boxes, 6), dtype=np.int64)
    nb_batch = np.zeros(pop_size, dtype=np.int64)
    po_serial = np.zeros((pop_size, n_boxes, 6), dtype=np.int64)

    # Warmup.
    for _ in range(2):
        cy.decode_batch_njit_mode(
            chroms, n_rots, dims, L, W, H, max_pallets, mode,
            po_batch, nb_batch,
        )
        order = np.argsort(chroms[0]).astype(np.int64)
        cy.decode_njit_mode(
            order, n_rots, dims, L, W, H, max_pallets, po_serial[0], mode,
        )

    # Serial.
    t0 = time.perf_counter()
    for _ in range(n_runs):
        for i in range(pop_size):
            order = np.argsort(chroms[i]).astype(np.int64)
            cy.decode_njit_mode(
                order, n_rots, dims, L, W, H, max_pallets, po_serial[i], mode,
            )
    t_serial = (time.perf_counter() - t0) / n_runs

    # Batch (parallel).
    t0 = time.perf_counter()
    for _ in range(n_runs):
        cy.decode_batch_njit_mode(
            chroms, n_rots, dims, L, W, H, max_pallets, mode,
            po_batch, nb_batch,
        )
    t_batch = (time.perf_counter() - t0) / n_runs

    return t_serial, t_batch


def main():
    rng = np.random.default_rng(43)

    print("=== A/B correctness ===")
    for mode in (0, 1, 2):
        for pop in (8, 80):
            ok, diff = ab_correctness(rng, pop, 80, 1000, 800, 600, 6, mode)
            tag = "OK" if ok else f"FAIL ({diff} diff chroms)"
            print(f"  mode={mode}  pop={pop:3d}  n_boxes=80   {tag}")
            if not ok:
                sys.exit(1)

    print("\n=== Benchmark — pop_size=80 decodes ===")
    print(f"  OMP_NUM_THREADS = {os.environ.get('OMP_NUM_THREADS', '(unset → use all)')}")
    for mode in (0, 1, 2):
        t_serial, t_batch = benchmark(
            rng, 80, 80, 1000, 800, 600, 6, mode, n_runs=5)
        speedup = t_serial / t_batch if t_batch > 0 else float('inf')
        print(f"  mode={mode}  serial {t_serial*1e3:7.1f} ms  "
              f"batch {t_batch*1e3:7.1f} ms  speedup {speedup:.2f}×")


if __name__ == "__main__":
    main()
