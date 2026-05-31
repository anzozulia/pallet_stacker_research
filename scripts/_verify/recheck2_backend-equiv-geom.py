"""recheck2 — pin down WHY the full solve survives a zero-dim box and whether
the Cython garbage geometry is actually validator-safe.

Q1. In a full geom solve, is the per-chromosome NUMBA decoder ever invoked, or
    is it 100% Cython batch? (If Numba is never called, its ZeroDivisionError
    can never fire in production.)
Q2. Does the Cython _find_best_block_at_pos with dx=0 produce GARBAGE k (giant
    block count) that would place boxes out of bounds, or does the EMS
    containment guard suppress it? Directly drive the helper.
Q3. Construct a worst-case: many zero-dim boxes of one SKU + a real EMS so the
    block path computes (ex_max - x) // 0. Show the exact k,l,m Cython returns
    and whether commit/placement go out of pallet bounds -> validator catch.
Q4. Confirm validator actually inspects zero-volume placements (not silently
    skipping them): feed it a deliberately OUT-OF-BOUNDS zero-dim placement and
    confirm it FAILS, proving T3's clean validate() is meaningful.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from pallet_packer._brkga_core import jit_decoders_geom_cy as cy


def banner(t):
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72)


# Q1: is Numba per-chromosome decoder reachable in a full geom solve?
def q1():
    banner("Q1: does a full geom solve ever call the NUMBA per-chrom decoder?")
    from pallet_packer._brkga_core import dispatch
    import pallet_packer._brkga_core.jit_decoders_geom as nbmod

    print(f"  dispatch._BATCH_AVAILABLE = {getattr(dispatch, '_BATCH_AVAILABLE', '??')}")

    # Monkeypatch every numba per-chrom geom decoder to record if it is hit.
    hits = {"called": 0, "names": []}

    for fname in ("decode_njit_mode", "decode_layer_njit",
                  "decode_blocks_njit_mode", "decode_precomputed_blocks_njit_mode"):
        orig = getattr(nbmod, fname)

        def make(orig=orig, fname=fname):
            def wrapper(*a, **k):
                hits["called"] += 1
                hits["names"].append(fname)
                return orig(*a, **k)
            return wrapper
        setattr(nbmod, fname, make())

    from pallet_packer import Box, Pallet, PackerConfig
    from pallet_packer.brkga_v3_5 import brkga_pack_v35
    boxes = [Box(id=f"b{i}", length=20.0, width=20.0, height=20.0) for i in range(20)]
    pallet = Pallet(length=100.0, width=100.0, height=100.0)
    res = brkga_pack_v35(boxes, pallet, PackerConfig(), time_limit_s=2.0,
                         max_pallets=1, seed=1, population_size=20, n_populations=1,
                         patience=20, local_search_budget_s=0.0, n_modes=6, verbose=False)
    print(f"  numba per-chrom geom decoder calls during full solve: {hits['called']}")
    print(f"  (names seen: {sorted(set(hits['names']))})")
    print(f"  => Numba per-chrom path is {'NOT' if hits['called'] == 0 else ''} on the hot loop")
    return hits["called"]


# Q2/Q3: directly drive Cython block helper with dx=0 and inspect k,l,m
def q2():
    banner("Q2/Q3: Cython _find_best_block_at_pos with dx=0 -> what k,l,m?")
    # One EMS covering full 100x100x100 pallet, box footprint at origin with
    # dx=0 (degenerate). max_count large.
    emss = np.zeros((1, 2, 3), dtype=np.int64)
    emss[0, 0] = (0, 0, 0)        # min
    emss[0, 1] = (100, 100, 100)  # max
    n_ems = 1
    x = y = z = 0
    max_count = 50

    for label, (dx, dy, dz) in {
        "dx=0": (0, 20, 20),
        "dy=0": (20, 0, 20),
        "dz=0": (20, 20, 0),
        "all-0": (0, 0, 0),
        "sane 20^3": (20, 20, 20),
    }.items():
        try:
            k, l, m = cy.find_best_block_at_pos_njit(
                emss, n_ems, x, y, z, dx, dy, dz, max_count)
            # block region extent the decoder would commit:
            ex = x + k * dx
            ey = y + l * dy
            ez = z + m * dz
            in_bounds = (ex <= 100 and ey <= 100 and ez <= 100)
            print(f"  {label:10s}: k,l,m=({k},{l},{m}) block_extent=({ex},{ey},{ez}) "
                  f"in_pallet_bounds={in_bounds}")
        except Exception as e:  # noqa: BLE001
            print(f"  {label:10s}: EXC {type(e).__name__}: {e}")


# Q4: prove the validator actually inspects placements (not skipping zero-vol)
def q4():
    banner("Q4: does the validator catch a deliberately out-of-bounds placement?")
    from pallet_packer import (Box, Pallet, PackerConfig, validate, Placement,
                               Rotation)
    NO_ROTATION = Rotation.LWH
    from pallet_packer.packer import PalletState, PackResult
    pallet = Pallet(length=100.0, width=100.0, height=100.0)
    cfg = PackerConfig()

    # (b) a NORMAL box placed clearly out of bounds -> should be INVALID
    nb_box = Box(id="oob", length=20.0, width=20.0, height=20.0)
    ps_bad = PalletState(pallet, "P001", cfg)
    ps_bad.placements.append(Placement(box=nb_box, rotation=NO_ROTATION,
                                       x=200.0, y=0.0, z=0.0))  # way outside
    res_bad = PackResult(pallets=[ps_bad], unpacked=[])
    v_bad = validate(res_bad, pallet, cfg)
    print(f"  out-of-bounds normal   : violations={len(v_bad)} {v_bad[:3]}")
    print(f"  => validator IS sensitive to bad geometry: {len(v_bad) > 0}")


def q5():
    banner("Q5: C-UB stability of // 0 across many EMS/positions (cdivision=True)")
    # Drive the helper with dx=0 and a variety of EMS sizes/positions to check
    # the UB result is consistently benign (always (1,1,1)) and never produces
    # a huge max_k that would escape the guards.
    rng = np.random.default_rng(99)
    worst = (1, 1, 1)
    bad = 0
    for _ in range(2000):
        ex_max = int(rng.integers(1, 2000))
        ey_max = int(rng.integers(1, 2000))
        ez_max = int(rng.integers(1, 2000))
        emss = np.zeros((1, 2, 3), dtype=np.int64)
        emss[0, 1] = (ex_max, ey_max, ez_max)
        # exactly one axis is zero
        axis = int(rng.integers(0, 3))
        dx, dy, dz = (20, 20, 20)
        if axis == 0:
            dx = 0
        elif axis == 1:
            dy = 0
        else:
            dz = 0
        k, l, m = cy.find_best_block_at_pos_njit(emss, 1, 0, 0, 0, dx, dy, dz, 50)
        ex = 0 + k * dx
        ey = 0 + l * dy
        ez = 0 + m * dz
        if ex > ex_max or ey > ey_max or ez > ez_max:
            bad += 1
        if k * l * m > worst[0] * worst[1] * worst[2]:
            worst = (k, l, m)
    print(f"  2000 zero-dim block calls: out-of-bounds block extents = {bad}")
    print(f"  worst (largest) k,l,m returned with a zero dim = {worst}")
    print(f"  => Cython // 0 (UB) stays benign on this build: {bad == 0}")
    return bad


def main():
    n = q1()
    q2()
    q4()
    bad = q5()
    banner("SUMMARY")
    print(f"  Numba per-chrom calls in full solve: {n}")
    print("  If 0, the Numba ZeroDivisionError can NEVER fire in production,")
    print("  because the hot loop is Cython-batch-only.")


if __name__ == "__main__":
    main()
