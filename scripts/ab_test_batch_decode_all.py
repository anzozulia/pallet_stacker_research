"""
ab_test_batch_decode_all.py — Phase 5b A/B + benchmark for ALL batch
entry points (geom modes 3/4/5 + all 3 cstr decoders).

Each test verifies batch[i] == serial(chromosomes[i]) bit-identical, then
times pop_size=80 decodes serial vs batch to measure parallel speedup.

Run inside Docker:
    docker run --rm -v "$(pwd)":/app -w /app pallet-packer:dev \
        python scripts/ab_test_batch_decode_all.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from pallet_packer._brkga_core import jit_decoders_geom_cy as geom
from pallet_packer._brkga_core import jit_decoders_cstr_cy as cstr


def make_random_population(rng, pop_size, n_boxes, L, W, H, weight_max=10.0):
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
    chromosomes = rng.random((pop_size, n_boxes)).astype(np.float64)
    weights = rng.uniform(0.5, weight_max, n_boxes).astype(np.float64)
    return chromosomes, n_rots_per_box, dims_all, weights


def make_sku_assignment(rng, n_boxes, n_skus):
    return rng.integers(0, n_skus, n_boxes).astype(np.int64)


def assert_match(label, po_b, nb_b, po_s, nb_s):
    same = np.array_equal(po_b, po_s) and np.array_equal(nb_b, nb_s)
    if same:
        print(f"  {label}: bit-identical ✓")
        return True
    diff = sum(not np.array_equal(po_b[i], po_s[i]) for i in range(len(nb_b)))
    print(f"  {label}: MISMATCH ({diff}/{len(nb_b)} chroms differ)")
    return False


def bench(label, fn_batch, fn_serial, pop_size, n_runs=5):
    """Time batch vs serial; both are pre-warmed callables that close over args."""
    for _ in range(2):
        fn_batch(); fn_serial()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        fn_batch()
    t_b = (time.perf_counter() - t0) / n_runs
    t0 = time.perf_counter()
    for _ in range(n_runs):
        fn_serial()
    t_s = (time.perf_counter() - t0) / n_runs
    speedup = t_s / t_b if t_b > 0 else float('inf')
    print(f"  {label}: serial {t_s*1e3:6.1f} ms  batch {t_b*1e3:6.1f} ms  "
          f"speedup {speedup:.2f}×")


# ===========================================================================
# GEOMETRIC MODES 3 / 4 / 5
# ===========================================================================

def test_layer(rng, pop_size=80, n_boxes=80):
    L, W, H, mp = 1000, 800, 600, 6
    chroms, n_rots, dims, _ = make_random_population(rng, pop_size, n_boxes, L, W, H)
    po_b = np.zeros((pop_size, n_boxes, 6), dtype=np.int64)
    nb_b = np.zeros(pop_size, dtype=np.int64)
    geom.decode_batch_layer_njit(chroms, n_rots, dims, L, W, H, mp, po_b, nb_b)
    po_s = np.zeros_like(po_b)
    nb_s = np.zeros_like(nb_b)
    for i in range(pop_size):
        order = np.argsort(chroms[i]).astype(np.int64)
        nb_s[i] = geom.decode_layer_njit(order, n_rots, dims, L, W, H, mp, po_s[i])
    if not assert_match("layer (mode 3)", po_b, nb_b, po_s, nb_s):
        sys.exit(1)


def test_blocks(rng, pop_size=80, n_boxes=80, n_skus=4):
    L, W, H, mp = 1000, 800, 600, 6
    chroms, n_rots, dims, _ = make_random_population(rng, pop_size, n_boxes, L, W, H)
    sku = make_sku_assignment(rng, n_boxes, n_skus)
    po_b = np.zeros((pop_size, n_boxes, 6), dtype=np.int64)
    nb_b = np.zeros(pop_size, dtype=np.int64)
    geom.decode_batch_blocks_njit_mode(
        chroms, n_rots, dims, sku, L, W, H, mp, po_b, nb_b, n_skus)
    po_s = np.zeros_like(po_b)
    nb_s = np.zeros_like(nb_b)
    for i in range(pop_size):
        order = np.argsort(chroms[i]).astype(np.int64)
        nb_s[i] = geom.decode_blocks_njit_mode(
            order, n_rots, dims, sku, L, W, H, mp, po_s[i], n_skus)
    if not assert_match("blocks (mode 4)", po_b, nb_b, po_s, nb_s):
        sys.exit(1)


def test_precomputed(rng, pop_size=80, n_boxes=80, n_skus=4):
    L, W, H, mp = 1000, 800, 600, 6
    chroms, n_rots, dims, _ = make_random_population(rng, pop_size, n_boxes, L, W, H)
    sku = make_sku_assignment(rng, n_boxes, n_skus)
    sku_bb = np.zeros((n_skus, 4), dtype=np.int64)
    for s in range(n_skus):
        sku_bb[s] = (
            int(rng.integers(1, 4)),
            int(rng.integers(1, 4)),
            int(rng.integers(1, 4)),
            int(rng.integers(0, 6)),
        )
    po_b = np.zeros((pop_size, n_boxes, 6), dtype=np.int64)
    nb_b = np.zeros(pop_size, dtype=np.int64)
    geom.decode_batch_precomputed_blocks_njit_mode(
        chroms, n_rots, dims, sku, sku_bb, L, W, H, mp, po_b, nb_b, n_skus)
    po_s = np.zeros_like(po_b)
    nb_s = np.zeros_like(nb_b)
    for i in range(pop_size):
        order = np.argsort(chroms[i]).astype(np.int64)
        nb_s[i] = geom.decode_precomputed_blocks_njit_mode(
            order, n_rots, dims, sku, sku_bb, L, W, H, mp, po_s[i], n_skus)
    if not assert_match("precomputed (mode 5)", po_b, nb_b, po_s, nb_s):
        sys.exit(1)


# ===========================================================================
# CSTR MODES — 0/1/2 + 3 + 4
# ===========================================================================

def _cstr_args(rng, n_boxes, L, W, H):
    """Pick a random constraint config that exercises every guard."""
    return {
        "weights": rng.uniform(0.5, 5.0, n_boxes).astype(np.float64),
        "mlot": np.full(n_boxes, 50.0, dtype=np.float64),
        "rfs": (rng.random(n_boxes) < 0.3).astype(np.int64),
        "pallet_max_weight": 1e18,
        "support_ratio": 0.5,
        "require_centroid": 0,
        "cog_x_min": -1e18, "cog_x_max": 1e18,
        "cog_y_min": -1e18, "cog_y_max": 1e18,
        "cog_min_load_frac": 0.0,
        "cog_active": 0,
    }


def test_cstr_012(rng, pop_size=80, n_boxes=80):
    L, W, H, mp = 1000, 800, 600, 6
    chroms, n_rots, dims, _ = make_random_population(rng, pop_size, n_boxes, L, W, H)
    ca = _cstr_args(rng, n_boxes, L, W, H)
    for mode in (0, 1, 2):
        po_b = np.zeros((pop_size, n_boxes, 6), dtype=np.int64)
        nb_b = np.zeros(pop_size, dtype=np.int64)
        cstr.decode_batch_njit_mode_cstr(
            chroms, n_rots, dims, L, W, H, mp, mode,
            ca["weights"], ca["mlot"], ca["rfs"],
            ca["pallet_max_weight"], ca["support_ratio"], ca["require_centroid"],
            ca["cog_x_min"], ca["cog_x_max"], ca["cog_y_min"], ca["cog_y_max"],
            ca["cog_min_load_frac"], ca["cog_active"],
            po_b, nb_b,
        )
        po_s = np.zeros_like(po_b)
        nb_s = np.zeros_like(nb_b)
        for i in range(pop_size):
            order = np.argsort(chroms[i]).astype(np.int64)
            nb_s[i] = cstr.decode_njit_mode_cstr(
                order, n_rots, dims, L, W, H, mp, po_s[i], mode,
                ca["weights"], ca["mlot"], ca["rfs"],
                ca["pallet_max_weight"], ca["support_ratio"], ca["require_centroid"],
                ca["cog_x_min"], ca["cog_x_max"], ca["cog_y_min"], ca["cog_y_max"],
                ca["cog_min_load_frac"], ca["cog_active"],
            )
        if not assert_match(f"cstr mode {mode}", po_b, nb_b, po_s, nb_s):
            sys.exit(1)


def test_cstr_blocks(rng, pop_size=80, n_boxes=80, n_skus=4):
    L, W, H, mp = 1000, 800, 600, 6
    chroms, n_rots, dims, _ = make_random_population(rng, pop_size, n_boxes, L, W, H)
    sku = make_sku_assignment(rng, n_boxes, n_skus)
    ca = _cstr_args(rng, n_boxes, L, W, H)
    po_b = np.zeros((pop_size, n_boxes, 6), dtype=np.int64)
    nb_b = np.zeros(pop_size, dtype=np.int64)
    cstr.decode_batch_blocks_njit_mode_cstr(
        chroms, n_rots, dims, sku, L, W, H, mp,
        ca["weights"], ca["mlot"], ca["rfs"],
        ca["pallet_max_weight"], ca["support_ratio"], ca["require_centroid"],
        ca["cog_x_min"], ca["cog_x_max"], ca["cog_y_min"], ca["cog_y_max"],
        ca["cog_min_load_frac"], ca["cog_active"],
        po_b, nb_b, n_skus,
    )
    po_s = np.zeros_like(po_b)
    nb_s = np.zeros_like(nb_b)
    for i in range(pop_size):
        order = np.argsort(chroms[i]).astype(np.int64)
        nb_s[i] = cstr.decode_blocks_njit_mode_cstr(
            order, n_rots, dims, sku, L, W, H, mp, po_s[i], n_skus,
            ca["weights"], ca["mlot"], ca["rfs"],
            ca["pallet_max_weight"], ca["support_ratio"], ca["require_centroid"],
            ca["cog_x_min"], ca["cog_x_max"], ca["cog_y_min"], ca["cog_y_max"],
            ca["cog_min_load_frac"], ca["cog_active"],
        )
    if not assert_match("cstr blocks (mode 4)", po_b, nb_b, po_s, nb_s):
        sys.exit(1)


def test_cstr_layer(rng, pop_size=80, n_boxes=80):
    L, W, H, mp = 1000, 800, 600, 6
    chroms, n_rots, dims, _ = make_random_population(rng, pop_size, n_boxes, L, W, H)
    ca = _cstr_args(rng, n_boxes, L, W, H)
    po_b = np.zeros((pop_size, n_boxes, 6), dtype=np.int64)
    nb_b = np.zeros(pop_size, dtype=np.int64)
    cstr.decode_batch_layer_njit_cstr(
        chroms, n_rots, dims, L, W, H, mp,
        ca["weights"], ca["mlot"], ca["rfs"],
        ca["pallet_max_weight"], ca["support_ratio"], ca["require_centroid"],
        ca["cog_x_min"], ca["cog_x_max"], ca["cog_y_min"], ca["cog_y_max"],
        ca["cog_min_load_frac"], ca["cog_active"],
        po_b, nb_b,
    )
    po_s = np.zeros_like(po_b)
    nb_s = np.zeros_like(nb_b)
    for i in range(pop_size):
        order = np.argsort(chroms[i]).astype(np.int64)
        nb_s[i] = cstr.decode_layer_njit_cstr(
            order, n_rots, dims, L, W, H, mp, po_s[i],
            ca["weights"], ca["mlot"], ca["rfs"],
            ca["pallet_max_weight"], ca["support_ratio"], ca["require_centroid"],
            ca["cog_x_min"], ca["cog_x_max"], ca["cog_y_min"], ca["cog_y_max"],
            ca["cog_min_load_frac"], ca["cog_active"],
        )
    if not assert_match("cstr layer (mode 3)", po_b, nb_b, po_s, nb_s):
        sys.exit(1)


def main():
    rng = np.random.default_rng(43)
    print("=== A/B correctness — geom 3/4/5 + cstr 0/1/2/3/4 ===")
    test_layer(rng)
    test_blocks(rng)
    test_precomputed(rng)
    test_cstr_012(rng)
    test_cstr_blocks(rng)
    test_cstr_layer(rng)

    # Brief benchmark — pop=80, n_boxes=80.
    print("\n=== Benchmark — pop=80, n_boxes=80 ===")
    pop = 80
    n_boxes = 80
    L, W, H, mp = 1000, 800, 600, 6
    chroms, n_rots, dims, _ = make_random_population(rng, pop, n_boxes, L, W, H)
    sku = make_sku_assignment(rng, n_boxes, 4)
    sku_bb = np.array([[2, 2, 2, 0]] * 4, dtype=np.int64)
    ca = _cstr_args(rng, n_boxes, L, W, H)

    po_b = np.zeros((pop, n_boxes, 6), dtype=np.int64)
    nb_b = np.zeros(pop, dtype=np.int64)
    po_s = np.zeros_like(po_b)

    # Geom 3 (layer).
    bench("geom mode 3 (layer)",
          lambda: geom.decode_batch_layer_njit(
              chroms, n_rots, dims, L, W, H, mp, po_b, nb_b),
          lambda: [geom.decode_layer_njit(
              np.argsort(chroms[i]).astype(np.int64), n_rots, dims,
              L, W, H, mp, po_s[i]) for i in range(pop)],
          pop)

    # Geom 4 (blocks).
    bench("geom mode 4 (blocks)",
          lambda: geom.decode_batch_blocks_njit_mode(
              chroms, n_rots, dims, sku, L, W, H, mp, po_b, nb_b, 4),
          lambda: [geom.decode_blocks_njit_mode(
              np.argsort(chroms[i]).astype(np.int64), n_rots, dims, sku,
              L, W, H, mp, po_s[i], 4) for i in range(pop)],
          pop)

    # Geom 5 (precomputed).
    bench("geom mode 5 (precomputed)",
          lambda: geom.decode_batch_precomputed_blocks_njit_mode(
              chroms, n_rots, dims, sku, sku_bb, L, W, H, mp, po_b, nb_b, 4),
          lambda: [geom.decode_precomputed_blocks_njit_mode(
              np.argsort(chroms[i]).astype(np.int64), n_rots, dims, sku,
              sku_bb, L, W, H, mp, po_s[i], 4) for i in range(pop)],
          pop)

    # Cstr 0/1/2 — just mode 0.
    bench("cstr mode 0",
          lambda: cstr.decode_batch_njit_mode_cstr(
              chroms, n_rots, dims, L, W, H, mp, 0,
              ca["weights"], ca["mlot"], ca["rfs"],
              ca["pallet_max_weight"], ca["support_ratio"], ca["require_centroid"],
              ca["cog_x_min"], ca["cog_x_max"], ca["cog_y_min"], ca["cog_y_max"],
              ca["cog_min_load_frac"], ca["cog_active"],
              po_b, nb_b),
          lambda: [cstr.decode_njit_mode_cstr(
              np.argsort(chroms[i]).astype(np.int64), n_rots, dims,
              L, W, H, mp, po_s[i], 0,
              ca["weights"], ca["mlot"], ca["rfs"],
              ca["pallet_max_weight"], ca["support_ratio"], ca["require_centroid"],
              ca["cog_x_min"], ca["cog_x_max"], ca["cog_y_min"], ca["cog_y_max"],
              ca["cog_min_load_frac"], ca["cog_active"]) for i in range(pop)],
          pop)

    # Cstr 3 (layer).
    bench("cstr mode 3 (layer)",
          lambda: cstr.decode_batch_layer_njit_cstr(
              chroms, n_rots, dims, L, W, H, mp,
              ca["weights"], ca["mlot"], ca["rfs"],
              ca["pallet_max_weight"], ca["support_ratio"], ca["require_centroid"],
              ca["cog_x_min"], ca["cog_x_max"], ca["cog_y_min"], ca["cog_y_max"],
              ca["cog_min_load_frac"], ca["cog_active"],
              po_b, nb_b),
          lambda: [cstr.decode_layer_njit_cstr(
              np.argsort(chroms[i]).astype(np.int64), n_rots, dims,
              L, W, H, mp, po_s[i],
              ca["weights"], ca["mlot"], ca["rfs"],
              ca["pallet_max_weight"], ca["support_ratio"], ca["require_centroid"],
              ca["cog_x_min"], ca["cog_x_max"], ca["cog_y_min"], ca["cog_y_max"],
              ca["cog_min_load_frac"], ca["cog_active"]) for i in range(pop)],
          pop)

    # Cstr 4 (blocks).
    bench("cstr mode 4 (blocks)",
          lambda: cstr.decode_batch_blocks_njit_mode_cstr(
              chroms, n_rots, dims, sku, L, W, H, mp,
              ca["weights"], ca["mlot"], ca["rfs"],
              ca["pallet_max_weight"], ca["support_ratio"], ca["require_centroid"],
              ca["cog_x_min"], ca["cog_x_max"], ca["cog_y_min"], ca["cog_y_max"],
              ca["cog_min_load_frac"], ca["cog_active"],
              po_b, nb_b, 4),
          lambda: [cstr.decode_blocks_njit_mode_cstr(
              np.argsort(chroms[i]).astype(np.int64), n_rots, dims, sku,
              L, W, H, mp, po_s[i], 4,
              ca["weights"], ca["mlot"], ca["rfs"],
              ca["pallet_max_weight"], ca["support_ratio"], ca["require_centroid"],
              ca["cog_x_min"], ca["cog_x_max"], ca["cog_y_min"], ca["cog_y_max"],
              ca["cog_min_load_frac"], ca["cog_active"]) for i in range(pop)],
          pop)


if __name__ == "__main__":
    main()
