"""
ab_test_decoder_cstr_modes012.py — Phase 4b A/B test:
Cython decode_njit_mode_cstr vs Numba reference, across cstr modes 0/1/2.

The cstr decoder has many constraint switches. Coverage matrix:
  - mode in {0, 1, 2}  (DFTRC, wall, corner)
  - weight cap         (none / tight / no slack)
  - max_load_on_top    (none / typical)
  - support_ratio      (0.0 / 0.5 / 1.0)
  - require_centroid   (off / on)
  - rfs                (off / mixed: half boxes require full support)
  - CoG envelope       (off / tight envelope)

We sample combinations and confirm bit-identical placements.

Run inside Docker:
    docker run --rm -v "$(pwd)":/app pallet-packer:dev \
        python scripts/ab_test_decoder_cstr_modes012.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from pallet_packer._brkga_core import jit_decoders_cstr as nb
from pallet_packer._brkga_core import jit_decoders_cstr_cy as cy


def make_random_instance(rng, n_boxes, L, W, H, weight_max=10.0):
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
    bps_order = rng.permutation(n_boxes).astype(np.int64)
    weights = rng.uniform(0.5, weight_max, size=n_boxes).astype(np.float64)
    return bps_order, n_rots_per_box, dims_all, weights


def ab_one(rng, n_boxes, L, W, H, max_pallets, mode,
           weight_cap, support_ratio, require_centroid,
           rfs_frac, cog_active, mlot_value):
    bps, n_rots, dims, weights = make_random_instance(rng, n_boxes, L, W, H)
    mlot = np.full(n_boxes, mlot_value, dtype=np.float64)
    rfs = (rng.random(n_boxes) < rfs_frac).astype(np.int64)

    if cog_active:
        # Narrow envelope around pallet centre.
        cxmn, cxmx = 0.3 * L, 0.7 * L
        cymn, cymx = 0.3 * W, 0.7 * W
        cmlf = 0.1
        cact = 1
    else:
        cxmn, cxmx = -1e18, 1e18
        cymn, cymx = -1e18, 1e18
        cmlf = 0.0
        cact = 0

    po_nb = np.zeros((n_boxes, 6), dtype=np.int64)
    po_cy = np.zeros((n_boxes, 6), dtype=np.int64)

    nb_bins = nb.decode_njit_mode_cstr(
        bps, n_rots, dims, L, W, H, max_pallets, po_nb, mode,
        weights, mlot, rfs, weight_cap, support_ratio,
        require_centroid,
        cxmn, cxmx, cymn, cymx, cmlf, cact,
    )
    cy_bins = cy.decode_njit_mode_cstr(
        bps, n_rots, dims, L, W, H, max_pallets, po_cy, mode,
        weights, mlot, rfs, weight_cap, support_ratio,
        require_centroid,
        cxmn, cxmx, cymn, cymx, cmlf, cact,
    )
    same = np.array_equal(po_nb, po_cy) and (nb_bins == cy_bins)
    diff = 0 if same else int(np.sum(np.any(po_nb != po_cy, axis=1)))
    return same, nb_bins, cy_bins, diff


def ab_test_suite(rng, n_cases=10):
    """Cover the constraint switch matrix."""
    cases = []
    for mode in (0, 1, 2):
        for weight_cap in (1e18, 500.0, 100.0):
            for sr in (0.0, 0.5, 1.0):
                for rc in (0, 1):
                    for rfs_frac in (0.0, 0.5):
                        for cog in (0, 1):
                            cases.append({
                                "mode": mode,
                                "weight_cap": weight_cap,
                                "support_ratio": sr,
                                "require_centroid": rc,
                                "rfs_frac": rfs_frac,
                                "cog_active": cog,
                                "mlot": 50.0,
                            })

    print(f"=== A/B correctness — decode_njit_mode_cstr ({len(cases)} configs) ===")
    fail = 0
    fail_configs = []
    for c in cases:
        per_fail = 0
        for _ in range(n_cases):
            ok, nb_bins, cy_bins, diff = ab_one(
                rng, 100, 1000, 800, 600, 8,
                c["mode"], c["weight_cap"], c["support_ratio"],
                c["require_centroid"], c["rfs_frac"],
                c["cog_active"], c["mlot"],
            )
            if not ok:
                per_fail += 1
                if fail < 3:
                    print(f"  MISMATCH cfg={c}: nb_bins={nb_bins} cy_bins={cy_bins} diff={diff}")
        if per_fail:
            fail += per_fail
            fail_configs.append(c)
    if fail:
        print(f"\n{fail} mismatches across {len(fail_configs)} configs — STOP and diff.")
        sys.exit(1)
    total = len(cases) * n_cases
    print(f"All {total} cases bit-identical across {len(cases)} configs.")


def microbenchmark(rng):
    print("\n=== Microbenchmark (mode=0, n=100, 1000 repeats) ===")
    bps, n_rots, dims, weights = make_random_instance(rng, 100, 1000, 800, 600)
    mlot = np.full(100, 50.0, dtype=np.float64)
    rfs = np.zeros(100, dtype=np.int64)
    po_nb = np.zeros((100, 6), dtype=np.int64)
    po_cy = np.zeros((100, 6), dtype=np.int64)
    args = (bps, n_rots, dims, 1000, 800, 600, 8)
    cstr_args = (weights, mlot, rfs, 1e18, 0.5, 0,
                 -1e18, 1e18, -1e18, 1e18, 0.0, 0)
    for _ in range(20):
        nb.decode_njit_mode_cstr(*args, po_nb, 0, *cstr_args)
        cy.decode_njit_mode_cstr(*args, po_cy, 0, *cstr_args)
    n_repeat = 1000
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        nb.decode_njit_mode_cstr(*args, po_nb, 0, *cstr_args)
    t_nb = (time.perf_counter() - t0) / n_repeat
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        cy.decode_njit_mode_cstr(*args, po_cy, 0, *cstr_args)
    t_cy = (time.perf_counter() - t0) / n_repeat
    speedup = t_nb / t_cy if t_cy > 0 else float('inf')
    print(f"  decode_njit_mode_cstr (mode=0): "
          f"numba {t_nb*1e6:.1f} µs/call  "
          f"cython {t_cy*1e6:.1f} µs/call  "
          f"speedup {speedup:.2f}×")


def main():
    rng = np.random.default_rng(43)
    ab_test_suite(rng, n_cases=10)
    microbenchmark(rng)


if __name__ == "__main__":
    main()
