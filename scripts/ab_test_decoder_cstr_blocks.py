"""
ab_test_decoder_cstr_blocks.py — Phase 4c A/B test:
Cython decode_blocks_njit_mode_cstr vs Numba reference.

Like the geometric blocks test, but with the constraint switch matrix
attached. Block extension fires when consecutive same-SKU boxes appear
in BPS, and constraints prune (k, l, m) via _max_block_under_constraints
+ _check_load_on_top.

Coverage:
  - sku_count in {1, 4, 30}  (homog / mixed / heterogeneous)
  - constraint configs combining weight cap, load_on_top, support_ratio,
    require_centroid, rfs, cog

Run inside Docker:
    docker run --rm -v "$(pwd)":/app pallet-packer:dev \
        python scripts/ab_test_decoder_cstr_blocks.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from pallet_packer._brkga_core import jit_decoders_cstr as nb
from pallet_packer._brkga_core import jit_decoders_cstr_cy as cy


def make_random_instance(rng, n_boxes, n_skus, L, W, H, weight_max=10.0):
    n_rots_per_box = np.full(n_boxes, 6, dtype=np.int64)
    dims_all = np.zeros((n_boxes, 6, 3), dtype=np.int64)
    sku_id_per_box = rng.integers(0, n_skus, size=n_boxes).astype(np.int64)
    sku_base = {}
    sku_w = {}
    for s in range(n_skus):
        sku_base[s] = (
            int(rng.integers(40, max(41, L // 4))),
            int(rng.integers(40, max(41, W // 4))),
            int(rng.integers(40, max(41, H // 4))),
        )
        sku_w[s] = float(rng.uniform(0.5, weight_max))
    for i in range(n_boxes):
        bx, by, bz = sku_base[int(sku_id_per_box[i])]
        rots = [
            (bx, by, bz), (bx, bz, by),
            (by, bx, bz), (by, bz, bx),
            (bz, bx, by), (bz, by, bx),
        ]
        for r in range(6):
            dims_all[i, r] = rots[r]
    weights = np.array(
        [sku_w[int(s)] for s in sku_id_per_box], dtype=np.float64)
    bps_order = rng.permutation(n_boxes).astype(np.int64)
    return bps_order, n_rots_per_box, dims_all, sku_id_per_box, weights


def ab_one(rng, n_boxes, n_skus, L, W, H, max_pallets,
           weight_cap, support_ratio, require_centroid,
           rfs_frac, cog_active, mlot_value):
    bps, n_rots, dims, sku_ids, weights = make_random_instance(
        rng, n_boxes, n_skus, L, W, H)
    mlot = np.full(n_boxes, mlot_value, dtype=np.float64)
    rfs = (rng.random(n_boxes) < rfs_frac).astype(np.int64)

    if cog_active:
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
    nb_bins = nb.decode_blocks_njit_mode_cstr(
        bps, n_rots, dims, sku_ids, L, W, H, max_pallets, po_nb, n_skus,
        weights, mlot, rfs, weight_cap, support_ratio,
        require_centroid,
        cxmn, cxmx, cymn, cymx, cmlf, cact,
    )
    cy_bins = cy.decode_blocks_njit_mode_cstr(
        bps, n_rots, dims, sku_ids, L, W, H, max_pallets, po_cy, n_skus,
        weights, mlot, rfs, weight_cap, support_ratio,
        require_centroid,
        cxmn, cxmx, cymn, cymx, cmlf, cact,
    )
    same = np.array_equal(po_nb, po_cy) and (nb_bins == cy_bins)
    diff = 0 if same else int(np.sum(np.any(po_nb != po_cy, axis=1)))
    return same, nb_bins, cy_bins, diff


def ab_test_suite(rng, n_cases=8):
    configs = []
    for n_skus in (1, 4, 30):
        for weight_cap in (1e18, 500.0, 100.0):
            for sr in (0.0, 0.5, 1.0):
                for rc in (0, 1):
                    for rfs_frac in (0.0, 0.5):
                        for cog in (0, 1):
                            configs.append({
                                "n_skus": n_skus,
                                "weight_cap": weight_cap,
                                "support_ratio": sr,
                                "require_centroid": rc,
                                "rfs_frac": rfs_frac,
                                "cog_active": cog,
                                "mlot": 30.0,
                            })
    print(f"=== A/B correctness — decode_blocks_njit_mode_cstr ({len(configs)} configs) ===")
    fail = 0
    fail_cfgs = []
    for c in configs:
        per_fail = 0
        for _ in range(n_cases):
            ok, nb_bins, cy_bins, diff = ab_one(
                rng, 100, c["n_skus"], 1000, 800, 600, 8,
                c["weight_cap"], c["support_ratio"],
                c["require_centroid"], c["rfs_frac"],
                c["cog_active"], c["mlot"],
            )
            if not ok:
                per_fail += 1
                if fail < 3:
                    print(f"  MISMATCH cfg={c}: nb={nb_bins} cy={cy_bins} diff={diff}")
        if per_fail:
            fail += per_fail
            fail_cfgs.append(c)
    if fail:
        print(f"\n{fail} mismatches across {len(fail_cfgs)} configs — STOP and diff.")
        sys.exit(1)
    total = len(configs) * n_cases
    print(f"All {total} cases bit-identical across {len(configs)} configs.")


def microbenchmark(rng):
    print("\n=== Microbenchmark (n=100, skus=4, 500 repeats) ===")
    bps, n_rots, dims, sku_ids, weights = make_random_instance(
        rng, 100, 4, 1000, 800, 600)
    mlot = np.full(100, 30.0, dtype=np.float64)
    rfs = np.zeros(100, dtype=np.int64)
    po_nb = np.zeros((100, 6), dtype=np.int64)
    po_cy = np.zeros((100, 6), dtype=np.int64)
    args = (bps, n_rots, dims, sku_ids, 1000, 800, 600, 8)
    tail = (4, weights, mlot, rfs, 1e18, 0.5, 0,
            -1e18, 1e18, -1e18, 1e18, 0.0, 0)
    for _ in range(10):
        nb.decode_blocks_njit_mode_cstr(*args, po_nb, *tail)
        cy.decode_blocks_njit_mode_cstr(*args, po_cy, *tail)
    n_repeat = 500
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        nb.decode_blocks_njit_mode_cstr(*args, po_nb, *tail)
    t_nb = (time.perf_counter() - t0) / n_repeat
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        cy.decode_blocks_njit_mode_cstr(*args, po_cy, *tail)
    t_cy = (time.perf_counter() - t0) / n_repeat
    speedup = t_nb / t_cy if t_cy > 0 else float('inf')
    print(f"  decode_blocks_njit_mode_cstr: "
          f"numba {t_nb*1e6:.1f} µs/call  "
          f"cython {t_cy*1e6:.1f} µs/call  "
          f"speedup {speedup:.2f}×")


def main():
    rng = np.random.default_rng(43)
    ab_test_suite(rng, n_cases=8)
    microbenchmark(rng)


if __name__ == "__main__":
    main()
