"""
ab_test_primitives.py — A/B test the Cython port of jit_primitives against
the Numba reference.

Generates random EMS arrays + random box queries, calls both
implementations, and asserts bit-identical (idx, x, y, z) returns. Also
prints a microbenchmark so we see the per-call speedup right away.

Run inside Docker:

    docker run --rm -v "$(pwd)":/app pallet-packer:dev \
        python scripts/ab_test_primitives.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from pallet_packer._brkga_core import jit_primitives as nb
from pallet_packer._brkga_core import jit_primitives_cy as cy


def make_random_emss(rng, n_ems, L=1000, W=800, H=600):
    """Generate a plausible EMS array — (n_ems, 2, 3) with min < max per dim."""
    emss = np.zeros((n_ems, 2, 3), dtype=np.int64)
    for i in range(n_ems):
        mn = rng.integers(0, L // 2, size=3)
        mx = mn + rng.integers(50, max(51, L // 2), size=3)
        # Clip to pallet
        mx[0] = min(mx[0], L)
        mx[1] = min(mx[1], W)
        mx[2] = min(mx[2], H)
        emss[i, 0] = mn
        emss[i, 1] = mx
    return emss


def ab_test(name, fn_nb, fn_cy, args_list):
    """Compare two implementations on a list of arg tuples."""
    n_mismatch = 0
    for args in args_list:
        r_nb = fn_nb(*args)
        r_cy = fn_cy(*args)
        if r_nb != r_cy:
            n_mismatch += 1
            if n_mismatch <= 3:
                print(f"  MISMATCH {name}: numba={r_nb}  cython={r_cy}")
                # Show distinguishing args excluding the huge array
                non_arr = [a for a in args if not isinstance(a, np.ndarray)]
                print(f"    args[non-array]={non_arr}")
    if n_mismatch == 0:
        print(f"  {name}: {len(args_list)} calls all bit-identical ✓")
    else:
        print(f"  {name}: {n_mismatch} / {len(args_list)} mismatch")
        sys.exit(1)


def microbenchmark(name, fn_nb, fn_cy, args_iter, n_repeat=10000):
    """Time both implementations on a single representative call."""
    args = next(args_iter)
    # Warm up JIT on both sides.
    for _ in range(100):
        fn_nb(*args)
        fn_cy(*args)
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        fn_nb(*args)
    t_nb = (time.perf_counter() - t0) / n_repeat
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        fn_cy(*args)
    t_cy = (time.perf_counter() - t0) / n_repeat
    speedup = t_nb / t_cy if t_cy > 0 else float('inf')
    print(f"  {name}: numba {t_nb*1e6:.2f} µs/call  "
          f"cython {t_cy*1e6:.2f} µs/call  "
          f"speedup {speedup:.2f}×")


def main():
    rng = np.random.default_rng(42)
    L, W, H = 1000, 800, 600

    # Build a diverse set of (emss, n_ems, dx, dy, dz, L, W, H) tuples.
    arg_sets_wall = []
    arg_sets_corner = []
    arg_sets_slab = []
    for _ in range(50):
        n_ems = int(rng.integers(1, 128))
        emss = make_random_emss(rng, max(n_ems, 1), L, W, H)
        # Pad the EMS array up to 128 (matches MAX_EMS layout from brkga_v3_fast).
        if emss.shape[0] < 128:
            pad = np.zeros((128 - emss.shape[0], 2, 3), dtype=np.int64)
            emss = np.concatenate([emss, pad], axis=0)
        dx = int(rng.integers(50, 300))
        dy = int(rng.integers(50, 300))
        dz = int(rng.integers(50, 300))
        arg_sets_wall.append((emss, n_ems, dx, dy, dz, L, W, H))
        arg_sets_corner.append((emss, n_ems, dx, dy, dz, L, W, H))
        slab_min = int(rng.integers(0, L // 2))
        slab_max = slab_min + int(rng.integers(50, L // 2))
        arg_sets_slab.append((emss, n_ems, dx, dy, dz, L, W, H,
                              slab_min, min(slab_max, L)))

    print("=== A/B correctness ===")
    ab_test("find_best_wall",
            nb.find_best_wall_njit, cy.find_best_wall_njit,
            arg_sets_wall)
    ab_test("find_best_corner",
            nb.find_best_corner_njit, cy.find_best_corner_njit,
            arg_sets_corner)
    ab_test("find_best_in_slab",
            nb.find_best_in_slab_njit, cy.find_best_in_slab_njit,
            arg_sets_slab)

    print("\n=== Microbenchmark (single call, 10k repeats) ===")
    microbenchmark("find_best_wall",
                   nb.find_best_wall_njit, cy.find_best_wall_njit,
                   iter(arg_sets_wall))
    microbenchmark("find_best_corner",
                   nb.find_best_corner_njit, cy.find_best_corner_njit,
                   iter(arg_sets_corner))
    microbenchmark("find_best_in_slab",
                   nb.find_best_in_slab_njit, cy.find_best_in_slab_njit,
                   iter(arg_sets_slab))


if __name__ == "__main__":
    main()
