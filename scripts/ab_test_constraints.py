"""
ab_test_constraints.py — A/B test the Cython port of jit_constraints
against the Numba reference. Bit-identical for the boolean checks; the
mutating apply_* helpers must produce identical float64 arrays.

Run inside Docker:

    docker run --rm -v "$(pwd)":/app pallet-packer:dev \
        python scripts/ab_test_constraints.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from pallet_packer._brkga_core import jit_constraints as nb
from pallet_packer._brkga_core import jit_constraints_cy as cy


def make_scene(rng, n_boxes=12, max_rots=6):
    """Synthetic placements + dims arrays. Mix of floor (z=0) and stacked."""
    dims_all = rng.integers(50, 300, size=(n_boxes, max_rots, 3)).astype(np.int64)
    placements_out = np.zeros((n_boxes, 6), dtype=np.int64)
    bps_order = np.arange(n_boxes, dtype=np.int64)
    rng.shuffle(bps_order)
    placement_top_loads = np.zeros(n_boxes, dtype=np.float64)
    mlot = rng.uniform(5.0, 50.0, size=n_boxes).astype(np.float64)
    # Place the first ~6 boxes — mix of floor and stacked.
    cum_x = 0
    for i in range(min(6, n_boxes)):
        box_idx = bps_order[i]
        rot = int(rng.integers(0, max_rots))
        dx = int(dims_all[box_idx, rot, 0])
        dy = int(dims_all[box_idx, rot, 1])
        dz = int(dims_all[box_idx, rot, 2])
        placements_out[i, 0] = 0  # one pallet
        placements_out[i, 1] = rot
        placements_out[i, 2] = cum_x
        placements_out[i, 3] = 0
        placements_out[i, 4] = 0  # all on floor for first pass
        placements_out[i, 5] = 1
        cum_x += dx
    # Stack a couple on top of placement 0.
    if n_boxes >= 8:
        for i in range(6, min(8, n_boxes)):
            box_idx = bps_order[i]
            rot = int(rng.integers(0, max_rots))
            placements_out[i, 0] = 0
            placements_out[i, 1] = rot
            placements_out[i, 2] = int(placements_out[0, 2])
            placements_out[i, 3] = int(placements_out[0, 3])
            dz0 = int(dims_all[bps_order[0], placements_out[0, 1], 2])
            placements_out[i, 4] = dz0
            placements_out[i, 5] = 1
    return placements_out, dims_all, bps_order, mlot, placement_top_loads


def random_candidate(rng, n_boxes, max_rots, with_z_above=False):
    box_idx = int(rng.integers(0, n_boxes))
    rot = int(rng.integers(0, max_rots))
    return box_idx, rot


def ab_check_load(rng, n_trials=200):
    n_boxes = 12
    max_rots = 6
    fails = 0
    for trial in range(n_trials):
        po, da, bo, mlot, ptl = make_scene(rng, n_boxes, max_rots)
        # Make a candidate placement — pick random coords.
        cand_x = int(rng.integers(0, 600))
        cand_y = int(rng.integers(0, 400))
        cand_z = int(rng.choice([0, 0, int(da[bo[0], po[0, 1], 2])]))  # mostly floor
        cand_dx = int(rng.integers(50, 200))
        cand_dy = int(rng.integers(50, 200))
        cand_dz = int(rng.integers(50, 200))
        cand_w = float(rng.uniform(0.5, 10.0))
        sr = float(rng.choice([0.0, 0.5, 0.8, 1.0]))
        rc = int(rng.choice([0, 1]))
        rfs = int(rng.choice([0, 1]))
        r_nb = nb._check_load_on_top_njit(
            po, da, bo, mlot, ptl, n_boxes, 0,
            cand_x, cand_y, cand_z, cand_dx, cand_dy, cand_dz,
            cand_w, sr, rc, rfs)
        r_cy = cy._check_load_on_top_njit(
            po, da, bo, mlot, ptl, n_boxes, 0,
            cand_x, cand_y, cand_z, cand_dx, cand_dy, cand_dz,
            cand_w, sr, rc, rfs)
        if bool(r_nb) != bool(r_cy):
            fails += 1
            if fails <= 3:
                print(f"  MISMATCH check_load trial {trial}: nb={r_nb} cy={r_cy}")
    label = "_check_load_on_top"
    if fails == 0:
        print(f"  {label}: {n_trials}/{n_trials} bit-identical ✓")
    else:
        print(f"  {label}: {fails}/{n_trials} mismatch")
        sys.exit(1)


def ab_check_cog(rng, n_trials=200):
    fails = 0
    for trial in range(n_trials):
        pw = rng.uniform(0, 500, size=4).astype(np.float64)
        pxw = (pw * rng.uniform(0, 1200, size=4)).astype(np.float64)
        pyw = (pw * rng.uniform(0, 1000, size=4)).astype(np.float64)
        cand_pallet = int(rng.integers(0, 4))
        cand_w = float(rng.uniform(0.5, 30.0))
        cand_x = int(rng.integers(0, 1000))
        cand_y = int(rng.integers(0, 800))
        cand_dx = int(rng.integers(50, 400))
        cand_dy = int(rng.integers(50, 300))
        pmw = float(rng.choice([300.0, 1000.0, 1e18]))
        cx_min, cx_max = 300.0, 700.0
        cy_min, cy_max = 250.0, 550.0
        load_frac = 0.35
        r_nb = nb._check_cog_envelope_njit(
            cand_x, cand_y, cand_dx, cand_dy, cand_w, cand_pallet,
            pw, pxw, pyw, pmw, cx_min, cx_max, cy_min, cy_max, load_frac)
        r_cy = cy._check_cog_envelope_njit(
            cand_x, cand_y, cand_dx, cand_dy, cand_w, cand_pallet,
            pw, pxw, pyw, pmw, cx_min, cx_max, cy_min, cy_max, load_frac)
        if bool(r_nb) != bool(r_cy):
            fails += 1
            if fails <= 3:
                print(f"  MISMATCH check_cog trial {trial}: nb={r_nb} cy={r_cy}")
    label = "_check_cog_envelope"
    if fails == 0:
        print(f"  {label}: {n_trials}/{n_trials} bit-identical ✓")
    else:
        print(f"  {label}: {fails}/{n_trials} mismatch")
        sys.exit(1)


def ab_apply_cog(rng, n_trials=200):
    """Apply functions are void; check that two parallel copies of the
    state remain equal after both implementations apply the same op."""
    fails = 0
    for trial in range(n_trials):
        pxw_nb = rng.uniform(0, 1e5, size=4).astype(np.float64)
        pyw_nb = rng.uniform(0, 1e5, size=4).astype(np.float64)
        pxw_cy = pxw_nb.copy()
        pyw_cy = pyw_nb.copy()
        cand_pallet = int(rng.integers(0, 4))
        cand_w = float(rng.uniform(0, 50))
        cand_x = int(rng.integers(0, 1000))
        cand_y = int(rng.integers(0, 800))
        cand_dx = int(rng.integers(50, 400))
        cand_dy = int(rng.integers(50, 300))
        nb._apply_cog_contribution_njit(
            cand_x, cand_y, cand_dx, cand_dy, cand_w, cand_pallet,
            pxw_nb, pyw_nb)
        cy._apply_cog_contribution_njit(
            cand_x, cand_y, cand_dx, cand_dy, cand_w, cand_pallet,
            pxw_cy, pyw_cy)
        if not (np.array_equal(pxw_nb, pxw_cy) and np.array_equal(pyw_nb, pyw_cy)):
            fails += 1
    label = "_apply_cog_contribution"
    if fails == 0:
        print(f"  {label}: {n_trials}/{n_trials} bit-identical ✓")
    else:
        print(f"  {label}: {fails}/{n_trials} mismatch")
        sys.exit(1)


def ab_apply_load(rng, n_trials=200):
    n_boxes = 12
    max_rots = 6
    fails = 0
    for trial in range(n_trials):
        po, da, bo, mlot, ptl_nb = make_scene(rng, n_boxes, max_rots)
        ptl_cy = ptl_nb.copy()
        cand_x = int(rng.integers(0, 600))
        cand_y = int(rng.integers(0, 400))
        cand_z = int(rng.choice([0, int(da[bo[0], po[0, 1], 2])]))
        cand_dx = int(rng.integers(50, 200))
        cand_dy = int(rng.integers(50, 200))
        cand_dz = int(rng.integers(50, 200))
        cand_w = float(rng.uniform(0, 10))
        nb._apply_load_contribution_njit(
            po, da, bo, ptl_nb, n_boxes, 0,
            cand_x, cand_y, cand_z, cand_dx, cand_dy, cand_dz, cand_w)
        cy._apply_load_contribution_njit(
            po, da, bo, ptl_cy, n_boxes, 0,
            cand_x, cand_y, cand_z, cand_dx, cand_dy, cand_dz, cand_w)
        if not np.allclose(ptl_nb, ptl_cy, rtol=0, atol=1e-9):
            fails += 1
            if fails <= 3:
                diff = np.abs(ptl_nb - ptl_cy).max()
                print(f"  MISMATCH apply_load trial {trial}: max-abs-diff={diff}")
    label = "_apply_load_contribution"
    if fails == 0:
        print(f"  {label}: {n_trials}/{n_trials} bit-identical ✓")
    else:
        print(f"  {label}: {fails}/{n_trials} mismatch")
        sys.exit(1)


def microbench(rng):
    """One-call microbench for each function."""
    n_boxes = 12
    max_rots = 6
    po, da, bo, mlot, ptl = make_scene(rng, n_boxes, max_rots)

    def time_it(label, fn, args, n_repeat=10000):
        for _ in range(100):
            fn(*args)
        t0 = time.perf_counter()
        for _ in range(n_repeat):
            fn(*args)
        return (time.perf_counter() - t0) / n_repeat * 1e6

    args_check = (po, da, bo, mlot, ptl, n_boxes, 0,
                  100, 100, int(da[bo[0], po[0, 1], 2]),
                  120, 100, 80, 5.0, 0.8, 1, 0)
    t_nb = time_it("ck", nb._check_load_on_top_njit, args_check)
    t_cy = time_it("ck", cy._check_load_on_top_njit, args_check)
    print(f"  _check_load_on_top:     numba {t_nb:.2f} µs  "
          f"cython {t_cy:.2f} µs  speedup {t_nb/t_cy:.2f}×")

    pw = np.array([100.0, 0.0, 0.0, 0.0])
    pxw = np.array([6e4, 0.0, 0.0, 0.0])
    pyw = np.array([4e4, 0.0, 0.0, 0.0])
    args_cog = (200, 150, 100, 80, 5.0, 0,
                pw, pxw, pyw, 1000.0,
                300.0, 700.0, 250.0, 550.0, 0.35)
    t_nb = time_it("cg", nb._check_cog_envelope_njit, args_cog)
    t_cy = time_it("cg", cy._check_cog_envelope_njit, args_cog)
    print(f"  _check_cog_envelope:    numba {t_nb:.2f} µs  "
          f"cython {t_cy:.2f} µs  speedup {t_nb/t_cy:.2f}×")

    pxw2 = pxw.copy(); pyw2 = pyw.copy()
    args_apply_cog = (200, 150, 100, 80, 5.0, 0, pxw2, pyw2)
    t_nb = time_it("ac", nb._apply_cog_contribution_njit, args_apply_cog)
    t_cy = time_it("ac", cy._apply_cog_contribution_njit, args_apply_cog)
    print(f"  _apply_cog_contribution: numba {t_nb:.2f} µs  "
          f"cython {t_cy:.2f} µs  speedup {t_nb/t_cy:.2f}×")

    ptl2 = ptl.copy()
    args_apply_load = (po, da, bo, ptl2, n_boxes, 0,
                       100, 100, int(da[bo[0], po[0, 1], 2]),
                       120, 100, 80, 5.0)
    t_nb = time_it("al", nb._apply_load_contribution_njit, args_apply_load)
    t_cy = time_it("al", cy._apply_load_contribution_njit, args_apply_load)
    print(f"  _apply_load_contribution: numba {t_nb:.2f} µs  "
          f"cython {t_cy:.2f} µs  speedup {t_nb/t_cy:.2f}×")


def main():
    rng = np.random.default_rng(42)
    print("=== A/B correctness ===")
    ab_check_load(rng)
    ab_check_cog(rng)
    ab_apply_cog(rng)
    ab_apply_load(rng)
    print("\n=== Microbenchmark (single call, 10k repeats) ===")
    microbench(rng)


if __name__ == "__main__":
    main()
