"""
verify_batch_consistency.py — adversarial batch-vs-serial consistency probe.

Verifies the parallel batch decoders (decode_batch_* in jit_decoders_geom_cy
and jit_decoders_cstr_cy) produce results BIT-IDENTICAL to looping the
per-chromosome Cython decoder, across ADVERSARIAL shapes the existing test
(scripts/ab_test_batch_decode_all.py) misses:

  - pop_size in {1, 2, 500}
  - n_boxes from 1..200 (sweep)
  - n_skus = 1 (all same SKU)  and  n_skus = n_boxes (all distinct)
  - degenerate dims (box larger than pallet on an axis; zero-ish footprints)
  - mode 5 per-chromosome top-K resolution path
      (decode_batch_precomputed_blocks_per_chrom via dispatch.decode_population_fitness
       vs decode_chromosome loop)
  - constraint combinations (weight cap, max_load_on_top, requires_full_support,
    support_ratio, CoG envelope)
  - decode_population_fitness fitness array == manual per-chromosome
    decode_chromosome + _fitness_pallet1 loop, BIT-IDENTICAL.

Run inside Docker (results are thread-invariant; OMP=1 for low CPU thrash):
    docker run --rm -e OMP_NUM_THREADS=1 -e PYTHONHASHSEED=0 \
        -v "$(pwd)":/app -w /app pallet-packer:dev \
        python scripts/_verify/verify_batch_consistency.py
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from pallet_packer._brkga_core import jit_decoders_geom_cy as geom
from pallet_packer._brkga_core import jit_decoders_cstr_cy as cstr
from pallet_packer._brkga_core import dispatch
from pallet_packer._brkga_core.blocks import (
    enumerate_top_k_blocks_per_sku,
    enumerate_best_block_per_sku,
    resolve_sku_blocks_from_chrom,
)
from pallet_packer._brkga_core.precompute import (
    precompute_box_dims_and_sku,
    precompute_constraint_arrays,
    precompute_cog_envelope,
    _NO_LIMIT,
)
from pallet_packer.brkga_v3_fast import precompute_box_dims, _fitness_pallet1
from pallet_packer.models import Box, Pallet, PackerConfig, Rotation, ALL_ROTATIONS


# Global tally.
N_CHECKS = 0
N_FAIL = 0
FAILS = []   # list of (label, detail) for worst-case reporting


def record(label, ok, detail=""):
    global N_CHECKS, N_FAIL
    N_CHECKS += 1
    if ok:
        print(f"  PASS  {label}")
    else:
        N_FAIL += 1
        FAILS.append((label, detail))
        print(f"  FAIL  {label}  -- {detail}")


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def make_random_dims(rng, n_boxes, L, W, H, n_rots=6):
    """Build (n_rots_per_box, dims_all) with random integer box dims that
    actually fit (each axis <= ~1/3 pallet so they pack)."""
    n_rots_per_box = np.full(n_boxes, n_rots, dtype=np.int64)
    dims_all = np.zeros((n_boxes, 6, 3), dtype=np.int64)
    for i in range(n_boxes):
        bx = int(rng.integers(30, max(31, L // 3)))
        by = int(rng.integers(30, max(31, W // 3)))
        bz = int(rng.integers(30, max(31, H // 3)))
        rots = [(bx, by, bz), (bx, bz, by), (by, bx, bz),
                (by, bz, bx), (bz, bx, by), (bz, by, bx)]
        for r in range(6):
            dims_all[i, r] = rots[r]
    return n_rots_per_box, dims_all


def random_chroms(rng, pop_size, chrom_len):
    return rng.random((pop_size, chrom_len)).astype(np.float64)


# ---------------------------------------------------------------------------
# Comparison helpers — batch buffer vs serial-loop buffer, bit-identical.
# ---------------------------------------------------------------------------
def cmp_buffers(label, po_b, nb_b, po_s, nb_s):
    same_nb = np.array_equal(nb_b, nb_s)
    same_po = np.array_equal(po_b, po_s)
    if same_nb and same_po:
        record(label, True)
        return True
    n_diff = sum(not np.array_equal(po_b[i], po_s[i]) for i in range(len(nb_b)))
    nb_diff = int(np.sum(nb_b != nb_s))
    # find first differing chromosome + cell
    detail = f"{n_diff}/{len(nb_b)} chroms differ in placements, {nb_diff} n_bins differ"
    for i in range(len(nb_b)):
        if not np.array_equal(po_b[i], po_s[i]):
            d = np.argwhere(po_b[i] != po_s[i])
            if len(d):
                box, col = int(d[0][0]), int(d[0][1])
                detail += (f"; first@chrom{i} box{box} col{col}: "
                           f"batch={po_b[i, box, col]} serial={po_s[i, box, col]}")
            break
    record(label, False, detail)
    return False


# ===========================================================================
# GEOMETRIC decoders — batch vs per-chrom loop
# ===========================================================================
def run_geom_all_modes(rng, label_prefix, pop_size, n_boxes, n_skus, L, W, H, mp=1):
    n_rots, dims = make_random_dims(rng, n_boxes, L, W, H)
    chrom_len = 2 * n_boxes + 1
    chroms = random_chroms(rng, pop_size, chrom_len)
    sku = rng.integers(0, max(1, n_skus), n_boxes).astype(np.int64)
    n_skus_eff = int(sku.max()) + 1

    # ---- mode 0/1/2 ----
    for mode in (0, 1, 2):
        po_b = np.zeros((pop_size, n_boxes, 6), dtype=np.int64)
        nb_b = np.zeros(pop_size, dtype=np.int64)
        geom.decode_batch_njit_mode(chroms, n_rots, dims, L, W, H, mp, mode, po_b, nb_b)
        po_s = np.zeros_like(po_b); nb_s = np.zeros_like(nb_b)
        for i in range(pop_size):
            order = np.argsort(chroms[i, :n_boxes]).astype(np.int64)
            nb_s[i] = geom.decode_njit_mode(order, n_rots, dims, L, W, H, mp, po_s[i], mode)
        cmp_buffers(f"{label_prefix} geom mode {mode}", po_b, nb_b, po_s, nb_s)

    # ---- mode 3 (layer) ----
    po_b = np.zeros((pop_size, n_boxes, 6), dtype=np.int64)
    nb_b = np.zeros(pop_size, dtype=np.int64)
    geom.decode_batch_layer_njit(chroms, n_rots, dims, L, W, H, mp, po_b, nb_b)
    po_s = np.zeros_like(po_b); nb_s = np.zeros_like(nb_b)
    for i in range(pop_size):
        order = np.argsort(chroms[i, :n_boxes]).astype(np.int64)
        nb_s[i] = geom.decode_layer_njit(order, n_rots, dims, L, W, H, mp, po_s[i])
    cmp_buffers(f"{label_prefix} geom mode 3 (layer)", po_b, nb_b, po_s, nb_s)

    # ---- mode 4 (dynamic blocks) ----
    po_b = np.zeros((pop_size, n_boxes, 6), dtype=np.int64)
    nb_b = np.zeros(pop_size, dtype=np.int64)
    geom.decode_batch_blocks_njit_mode(chroms, n_rots, dims, sku, L, W, H, mp, po_b, nb_b, n_skus_eff)
    po_s = np.zeros_like(po_b); nb_s = np.zeros_like(nb_b)
    for i in range(pop_size):
        order = np.argsort(chroms[i, :n_boxes]).astype(np.int64)
        nb_s[i] = geom.decode_blocks_njit_mode(order, n_rots, dims, sku, L, W, H, mp, po_s[i], n_skus_eff)
    cmp_buffers(f"{label_prefix} geom mode 4 (blocks)", po_b, nb_b, po_s, nb_s)

    # ---- mode 5 single shared best block ----
    sku_bb = enumerate_best_block_per_sku(
        [None] * 0 if False else _dummy_boxes_for_blocks(n_boxes, dims, n_rots, sku, n_skus_eff),
        sku, n_rots, dims, L, W, H)
    po_b = np.zeros((pop_size, n_boxes, 6), dtype=np.int64)
    nb_b = np.zeros(pop_size, dtype=np.int64)
    geom.decode_batch_precomputed_blocks_njit_mode(
        chroms, n_rots, dims, sku, sku_bb, L, W, H, mp, po_b, nb_b, n_skus_eff)
    po_s = np.zeros_like(po_b); nb_s = np.zeros_like(nb_b)
    for i in range(pop_size):
        order = np.argsort(chroms[i, :n_boxes]).astype(np.int64)
        nb_s[i] = geom.decode_precomputed_blocks_njit_mode(
            order, n_rots, dims, sku, sku_bb, L, W, H, mp, po_s[i], n_skus_eff)
    cmp_buffers(f"{label_prefix} geom mode 5 (best block)", po_b, nb_b, po_s, nb_s)

    # ---- mode 5 per-chrom top-K resolution ----
    top_k = enumerate_top_k_blocks_per_sku(
        _dummy_boxes_for_blocks(n_boxes, dims, n_rots, sku, n_skus_eff),
        sku, n_rots, dims, L, W, H, k_top=8)
    chosen_per_chrom = np.zeros((pop_size, n_skus_eff, 4), dtype=np.int64)
    for ci in range(pop_size):
        chosen_per_chrom[ci] = resolve_sku_blocks_from_chrom(
            chroms[ci], n_boxes, n_skus_eff, top_k)
    po_b = np.zeros((pop_size, n_boxes, 6), dtype=np.int64)
    nb_b = np.zeros(pop_size, dtype=np.int64)
    geom.decode_batch_precomputed_blocks_per_chrom(
        chroms, n_rots, dims, sku, chosen_per_chrom, L, W, H, mp, po_b, nb_b, n_skus_eff)
    po_s = np.zeros_like(po_b); nb_s = np.zeros_like(nb_b)
    for i in range(pop_size):
        order = np.argsort(chroms[i, :n_boxes]).astype(np.int64)
        chosen_i = resolve_sku_blocks_from_chrom(chroms[i], n_boxes, n_skus_eff, top_k)
        nb_s[i] = geom.decode_precomputed_blocks_njit_mode(
            order, n_rots, dims, sku, chosen_i, L, W, H, mp, po_s[i], n_skus_eff)
    cmp_buffers(f"{label_prefix} geom mode 5 (per-chrom top-K)", po_b, nb_b, po_s, nb_s)


def _dummy_boxes_for_blocks(n_boxes, dims, n_rots, sku, n_skus_eff):
    """Build Box objects matching the integer dims so enumerate_*_block_per_sku
    (which reads dims_all, not box dims) produces consistent blocks. The block
    enumerators only use dims_all/n_rots_arr/sku, so any Box list of right len
    with matching allowed_rotations count works; we still build real ones."""
    boxes = []
    for i in range(n_boxes):
        dx, dy, dz = int(dims[i, 0, 0]), int(dims[i, 0, 1]), int(dims[i, 0, 2])
        boxes.append(Box(id=f"b{i}", length=dx, width=dy, height=dz,
                         allowed_rotations=list(ALL_ROTATIONS)))
    return boxes


# ===========================================================================
# CSTR decoders — batch vs per-chrom loop
# ===========================================================================
def cstr_args(rng, n_boxes, *, pmw, mlot_val, rfs_frac, support_ratio, cog_active,
              cog_x_min=-1e18, cog_x_max=1e18, cog_y_min=-1e18, cog_y_max=1e18,
              cog_min_load_frac=0.0, require_centroid=0):
    return {
        "weights": rng.uniform(0.5, 5.0, n_boxes).astype(np.float64),
        "mlot": np.full(n_boxes, mlot_val, dtype=np.float64),
        "rfs": (rng.random(n_boxes) < rfs_frac).astype(np.int64),
        "pmw": pmw,
        "support_ratio": support_ratio,
        "require_centroid": require_centroid,
        "cog_x_min": cog_x_min, "cog_x_max": cog_x_max,
        "cog_y_min": cog_y_min, "cog_y_max": cog_y_max,
        "cog_min_load_frac": cog_min_load_frac,
        "cog_active": cog_active,
    }


def run_cstr_all(rng, label_prefix, pop_size, n_boxes, n_skus, L, W, H, ca, mp=1):
    n_rots, dims = make_random_dims(rng, n_boxes, L, W, H)
    chrom_len = 2 * n_boxes + 1
    chroms = random_chroms(rng, pop_size, chrom_len)
    sku = rng.integers(0, max(1, n_skus), n_boxes).astype(np.int64)
    n_skus_eff = int(sku.max()) + 1

    def call_serial_012(mode, po_s, nb_s):
        for i in range(pop_size):
            order = np.argsort(chroms[i, :n_boxes]).astype(np.int64)
            nb_s[i] = cstr.decode_njit_mode_cstr(
                order, n_rots, dims, L, W, H, mp, po_s[i], mode,
                ca["weights"], ca["mlot"], ca["rfs"], ca["pmw"], ca["support_ratio"],
                ca["require_centroid"], ca["cog_x_min"], ca["cog_x_max"],
                ca["cog_y_min"], ca["cog_y_max"], ca["cog_min_load_frac"], ca["cog_active"])

    for mode in (0, 1, 2):
        po_b = np.zeros((pop_size, n_boxes, 6), dtype=np.int64)
        nb_b = np.zeros(pop_size, dtype=np.int64)
        cstr.decode_batch_njit_mode_cstr(
            chroms, n_rots, dims, L, W, H, mp, mode,
            ca["weights"], ca["mlot"], ca["rfs"], ca["pmw"], ca["support_ratio"],
            ca["require_centroid"], ca["cog_x_min"], ca["cog_x_max"],
            ca["cog_y_min"], ca["cog_y_max"], ca["cog_min_load_frac"], ca["cog_active"],
            po_b, nb_b)
        po_s = np.zeros_like(po_b); nb_s = np.zeros_like(nb_b)
        call_serial_012(mode, po_s, nb_s)
        cmp_buffers(f"{label_prefix} cstr mode {mode}", po_b, nb_b, po_s, nb_s)

    # cstr layer (mode 3)
    po_b = np.zeros((pop_size, n_boxes, 6), dtype=np.int64)
    nb_b = np.zeros(pop_size, dtype=np.int64)
    cstr.decode_batch_layer_njit_cstr(
        chroms, n_rots, dims, L, W, H, mp,
        ca["weights"], ca["mlot"], ca["rfs"], ca["pmw"], ca["support_ratio"],
        ca["require_centroid"], ca["cog_x_min"], ca["cog_x_max"],
        ca["cog_y_min"], ca["cog_y_max"], ca["cog_min_load_frac"], ca["cog_active"],
        po_b, nb_b)
    po_s = np.zeros_like(po_b); nb_s = np.zeros_like(nb_b)
    for i in range(pop_size):
        order = np.argsort(chroms[i, :n_boxes]).astype(np.int64)
        nb_s[i] = cstr.decode_layer_njit_cstr(
            order, n_rots, dims, L, W, H, mp, po_s[i],
            ca["weights"], ca["mlot"], ca["rfs"], ca["pmw"], ca["support_ratio"],
            ca["require_centroid"], ca["cog_x_min"], ca["cog_x_max"],
            ca["cog_y_min"], ca["cog_y_max"], ca["cog_min_load_frac"], ca["cog_active"])
    cmp_buffers(f"{label_prefix} cstr mode 3 (layer)", po_b, nb_b, po_s, nb_s)

    # cstr blocks (mode 4)
    po_b = np.zeros((pop_size, n_boxes, 6), dtype=np.int64)
    nb_b = np.zeros(pop_size, dtype=np.int64)
    cstr.decode_batch_blocks_njit_mode_cstr(
        chroms, n_rots, dims, sku, L, W, H, mp,
        ca["weights"], ca["mlot"], ca["rfs"], ca["pmw"], ca["support_ratio"],
        ca["require_centroid"], ca["cog_x_min"], ca["cog_x_max"],
        ca["cog_y_min"], ca["cog_y_max"], ca["cog_min_load_frac"], ca["cog_active"],
        po_b, nb_b, n_skus_eff)
    po_s = np.zeros_like(po_b); nb_s = np.zeros_like(nb_b)
    for i in range(pop_size):
        order = np.argsort(chroms[i, :n_boxes]).astype(np.int64)
        nb_s[i] = cstr.decode_blocks_njit_mode_cstr(
            order, n_rots, dims, sku, L, W, H, mp, po_s[i], n_skus_eff,
            ca["weights"], ca["mlot"], ca["rfs"], ca["pmw"], ca["support_ratio"],
            ca["require_centroid"], ca["cog_x_min"], ca["cog_x_max"],
            ca["cog_y_min"], ca["cog_y_max"], ca["cog_min_load_frac"], ca["cog_active"])
    cmp_buffers(f"{label_prefix} cstr mode 4 (blocks)", po_b, nb_b, po_s, nb_s)


# ===========================================================================
# decode_population_fitness vs manual per-chromosome decode_chromosome loop
# ===========================================================================
def make_boxes(rng, n_boxes, L, W, H, *, n_skus, weighted=False, mlot=False,
               rfs=False, integer=True):
    """Build a real Box list. n_skus distinct shapes cycled across boxes."""
    shapes = []
    for s in range(max(1, n_skus)):
        if integer:
            bx = int(rng.integers(30, max(31, L // 3)))
            by = int(rng.integers(30, max(31, W // 3)))
            bz = int(rng.integers(30, max(31, H // 3)))
        else:
            bx = float(rng.uniform(30, L / 3)) + 0.37
            by = float(rng.uniform(30, W / 3)) + 0.61
            bz = float(rng.uniform(30, H / 3)) + 0.29
        shapes.append((bx, by, bz))
    boxes = []
    for i in range(n_boxes):
        bx, by, bz = shapes[i % len(shapes)]
        kw = dict(id=f"b{i}", length=bx, width=by, height=bz,
                  allowed_rotations=list(ALL_ROTATIONS))
        if weighted:
            kw["weight"] = float(rng.uniform(0.5, 5.0))
        if mlot:
            kw["max_load_on_top"] = 50.0
        if rfs and (i % 4 == 0):
            kw["requires_full_support"] = True
        boxes.append(Box(**kw))
    return boxes


def fitness_consistency(rng, label, boxes, pallet, config, pop_size, *,
                        n_modes, use_multi_decoder, use_top_k):
    """Compare decode_population_fitness array vs manual per-chrom decode loop."""
    global N_CHECKS, N_FAIL
    n = len(boxes)
    chrom_len = 2 * n + 1
    population = rng.random((pop_size, chrom_len)).astype(np.float64)

    n_rots_arr, dims_all, sku_id_per_box = precompute_box_dims_and_sku(boxes)
    weights, mlot, rfs, pmw, has_cstr = precompute_constraint_arrays(boxes, pallet)
    cx0, cx1, cy0, cy1, cmlf, cog_active = precompute_cog_envelope(pallet, config)
    require_centroid = 1 if config.require_centroid_supported else 0

    sku_best_block = None
    sku_top_k_blocks = None
    if n_modes >= 6 and not has_cstr:
        L = int(round(pallet.length)); W = int(round(pallet.width)); H = int(round(pallet.height))
        if use_top_k:
            sku_top_k_blocks = enumerate_top_k_blocks_per_sku(
                boxes, sku_id_per_box, n_rots_arr, dims_all, L, W, H, k_top=8)
        else:
            sku_best_block = enumerate_best_block_per_sku(
                boxes, sku_id_per_box, n_rots_arr, dims_all, L, W, H)

    # BATCH path.
    fits_batch = dispatch.decode_population_fitness(
        population, boxes, pallet, config, n_rots_arr, dims_all, max_pallets=1,
        use_multi_decoder=use_multi_decoder, n_modes=n_modes,
        sku_id_per_box=sku_id_per_box, sku_best_block=sku_best_block,
        sku_top_k_blocks=sku_top_k_blocks, mode_cdf=None,
        weights=weights, mlot=mlot, rfs=rfs, pallet_max_weight=pmw,
        has_constraints=has_cstr, support_ratio=config.support_ratio,
        require_centroid=require_centroid,
        cog_x_min=cx0, cog_x_max=cx1, cog_y_min=cy0, cog_y_max=cy1,
        cog_min_load_frac=cmlf, cog_active=1 if cog_active else 0,
        max_overhang=pallet.max_overhang)

    # SERIAL path: per-chrom decode_chromosome with the SAME mode selection.
    fits_serial = np.zeros(pop_size, dtype=np.float64)
    for i in range(pop_size):
        chrom = population[i]
        if use_multi_decoder and chrom_len >= 2 * n + 1:
            selector = float(chrom[-1])
            mode = dispatch._selector_to_mode(selector, n_modes, None)
        else:
            mode = 0
        res = dispatch.decode_chromosome(
            chrom, boxes, pallet, config, n_rots_arr, dims_all, mode, max_pallets=1,
            sku_id_per_box=sku_id_per_box, sku_best_block=sku_best_block,
            sku_top_k_blocks=sku_top_k_blocks,
            weights=weights, mlot=mlot, rfs=rfs, pallet_max_weight=pmw,
            has_constraints=has_cstr, support_ratio=config.support_ratio,
            require_centroid=require_centroid,
            cog_x_min=cx0, cog_x_max=cx1, cog_y_min=cy0, cog_y_max=cy1,
            cog_min_load_frac=cmlf, cog_active=1 if cog_active else 0,
            max_overhang=pallet.max_overhang)
        fits_serial[i] = _fitness_pallet1(res, pallet)

    same = np.array_equal(fits_batch, fits_serial)
    if same:
        record(label, True)
        return
    # Distinguish a bit-difference (real) from a tiny float rounding gap.
    diff = np.abs(fits_batch - fits_serial)
    max_diff = float(diff.max())
    n_diff = int(np.sum(fits_batch != fits_serial))
    worst = int(np.argmax(diff))
    detail = (f"{n_diff}/{pop_size} fits differ; max|d|={max_diff:.3e}; "
              f"worst chrom{worst}: batch={fits_batch[worst]:.12g} "
              f"serial={fits_serial[worst]:.12g}")
    # Classify: integer-vs-float volume discrepancy is a KNOWN architecture
    # gap (batch uses int dims_all volume, _fitness_pallet1 uses float box
    # volume). Flag it explicitly so it's distinguishable from a true
    # batch/serial DECODE divergence.
    record(label, False, detail)


# ===========================================================================
# DRIVER
# ===========================================================================
def main():
    dispatch.warmup_jit()
    print("=== backend check ===")
    print(f"  decode_njit_mode module: {dispatch.decode_njit_mode.__module__}")
    print(f"  _BATCH_AVAILABLE = {dispatch._BATCH_AVAILABLE}")
    if not dispatch._BATCH_AVAILABLE:
        print("  FATAL: batch decoders unavailable")
        return 2

    rng = np.random.default_rng(12345)
    L, W, H = 1200, 1000, 1500

    # --------------------------------------------------------------------
    # 1. pop_size edge cases (1, 2, 500) x geom + cstr decoders
    # --------------------------------------------------------------------
    print("\n=== 1. pop_size edge cases (geom, all modes) ===")
    for ps in (1, 2, 500):
        run_geom_all_modes(rng, f"[pop={ps}, n=40, sku=4]", ps, 40, 4, L, W, H)

    print("\n=== 1b. pop_size edge cases (cstr, all modes) ===")
    ca = cstr_args(rng, 40, pmw=200.0, mlot_val=50.0, rfs_frac=0.3,
                   support_ratio=0.5, cog_active=0)
    for ps in (1, 2, 500):
        run_cstr_all(rng, f"[pop={ps}, n=40, sku=4]", ps, 40, 4, L, W, H,
                     cstr_args(rng, 40, pmw=200.0, mlot_val=50.0, rfs_frac=0.3,
                               support_ratio=0.5, cog_active=0))

    # --------------------------------------------------------------------
    # 2. n_boxes sweep 1..200 (geom). pop=8 (small to keep runtime down,
    #    but >1 so prange splits work).
    # --------------------------------------------------------------------
    print("\n=== 2. n_boxes sweep 1..200 (geom, all modes) ===")
    for nb in [1, 2, 3, 5, 8, 13, 25, 50, 100, 150, 200]:
        run_geom_all_modes(rng, f"[n={nb}, pop=8, sku=4]", 8, nb, min(4, nb), L, W, H)

    # --------------------------------------------------------------------
    # 3. SKU extremes — n_skus=1 (all same) and n_skus=n_boxes (all distinct)
    # --------------------------------------------------------------------
    print("\n=== 3. SKU extremes (geom) ===")
    run_geom_all_modes(rng, "[n=60, sku=1 ALL-SAME]", 16, 60, 1, L, W, H)
    run_geom_all_modes(rng, "[n=60, sku=60 ALL-DISTINCT]", 16, 60, 60, L, W, H)
    print("\n=== 3b. SKU extremes (cstr) ===")
    run_cstr_all(rng, "[n=60, sku=1 ALL-SAME]", 16, 60, 1, L, W, H,
                 cstr_args(rng, 60, pmw=300.0, mlot_val=50.0, rfs_frac=0.3,
                           support_ratio=0.6, cog_active=0))
    run_cstr_all(rng, "[n=60, sku=60 ALL-DISTINCT]", 16, 60, 60, L, W, H,
                 cstr_args(rng, 60, pmw=300.0, mlot_val=50.0, rfs_frac=0.3,
                           support_ratio=0.6, cog_active=0))

    # --------------------------------------------------------------------
    # 4. Degenerate dims — boxes bigger than pallet on an axis (unplaceable),
    #    plus tiny/odd footprints. Tests the unplaced (col5=0) bookkeeping.
    # --------------------------------------------------------------------
    print("\n=== 4. degenerate dims (geom + cstr) ===")
    nb = 30
    n_rots = np.full(nb, 6, dtype=np.int64)
    dims = np.zeros((nb, 6, 3), dtype=np.int64)
    for i in range(nb):
        if i % 3 == 0:
            # oversized on every axis -> cannot be placed
            bx, by, bz = L + 100, W + 100, H + 100
        elif i % 3 == 1:
            # tall, thin
            bx, by, bz = 20, 20, H - 1
        else:
            bx, by, bz = int(rng.integers(40, 200)), int(rng.integers(40, 200)), int(rng.integers(40, 200))
        rots = [(bx, by, bz), (bx, bz, by), (by, bx, bz),
                (by, bz, bx), (bz, bx, by), (bz, by, bx)]
        for r in range(6):
            dims[i, r] = rots[r]
    sku = rng.integers(0, 5, nb).astype(np.int64)
    n_skus_eff = int(sku.max()) + 1
    chroms = random_chroms(rng, 16, 2 * nb + 1)
    # geom modes 0/1/2/3
    for mode in (0, 1, 2):
        po_b = np.zeros((16, nb, 6), dtype=np.int64); nb_b = np.zeros(16, dtype=np.int64)
        geom.decode_batch_njit_mode(chroms, n_rots, dims, L, W, H, 1, mode, po_b, nb_b)
        po_s = np.zeros_like(po_b); nb_s = np.zeros_like(nb_b)
        for i in range(16):
            order = np.argsort(chroms[i, :nb]).astype(np.int64)
            nb_s[i] = geom.decode_njit_mode(order, n_rots, dims, L, W, H, 1, po_s[i], mode)
        cmp_buffers(f"[degenerate] geom mode {mode}", po_b, nb_b, po_s, nb_s)
    po_b = np.zeros((16, nb, 6), dtype=np.int64); nb_b = np.zeros(16, dtype=np.int64)
    geom.decode_batch_blocks_njit_mode(chroms, n_rots, dims, sku, L, W, H, 1, po_b, nb_b, n_skus_eff)
    po_s = np.zeros_like(po_b); nb_s = np.zeros_like(nb_b)
    for i in range(16):
        order = np.argsort(chroms[i, :nb]).astype(np.int64)
        nb_s[i] = geom.decode_blocks_njit_mode(order, n_rots, dims, sku, L, W, H, 1, po_s[i], n_skus_eff)
    cmp_buffers("[degenerate] geom mode 4 (blocks)", po_b, nb_b, po_s, nb_s)

    # --------------------------------------------------------------------
    # 5. multi-pallet (max_pallets > 1) — batch vs serial.
    # --------------------------------------------------------------------
    print("\n=== 5. multi-pallet max_pallets=5 (geom) ===")
    run_geom_all_modes(rng, "[n=80, pop=12, sku=6, mp=5]", 12, 80, 6, L, W, H, mp=5)

    # --------------------------------------------------------------------
    # 6. CSTR combination matrix — exercise every guard active/inactive.
    # --------------------------------------------------------------------
    print("\n=== 6. cstr constraint combinations ===")
    combos = [
        ("weight-cap-only", dict(pmw=120.0, mlot_val=_NO_LIMIT, rfs_frac=0.0, support_ratio=0.0, cog_active=0)),
        ("mlot-only",       dict(pmw=_NO_LIMIT, mlot_val=10.0, rfs_frac=0.0, support_ratio=0.0, cog_active=0)),
        ("rfs-only",        dict(pmw=_NO_LIMIT, mlot_val=_NO_LIMIT, rfs_frac=0.5, support_ratio=0.0, cog_active=0)),
        ("support-ratio",   dict(pmw=_NO_LIMIT, mlot_val=_NO_LIMIT, rfs_frac=0.0, support_ratio=0.75, cog_active=0)),
        ("cog-envelope",    dict(pmw=200.0, mlot_val=50.0, rfs_frac=0.0, support_ratio=0.5, cog_active=1,
                                 cog_x_min=L*0.25, cog_x_max=L*0.75, cog_y_min=W*0.25, cog_y_max=W*0.75,
                                 cog_min_load_frac=0.3, require_centroid=1)),
        ("ALL-active",      dict(pmw=150.0, mlot_val=20.0, rfs_frac=0.4, support_ratio=0.8, cog_active=1,
                                 cog_x_min=L*0.3, cog_x_max=L*0.7, cog_y_min=W*0.3, cog_y_max=W*0.7,
                                 cog_min_load_frac=0.25, require_centroid=1)),
    ]
    for name, kw in combos:
        run_cstr_all(rng, f"[cstr:{name}]", 16, 50, 5, L, W, H,
                     cstr_args(rng, 50, **kw))

    # --------------------------------------------------------------------
    # 7. decode_population_fitness array == per-chrom decode_chromosome loop
    #    Across geometric / multi-decoder / top-K / constrained inputs.
    # --------------------------------------------------------------------
    print("\n=== 7. decode_population_fitness vs per-chrom loop (INTEGER dims) ===")
    pal = Pallet(length=1200, width=1000, height=1500)
    cfg_geom = PackerConfig(support_ratio=1.0, require_centroid_supported=False,
                            cog_envelope_fraction=1.0)
    # single-mode (mode 0) geometric
    boxes = make_boxes(rng, 60, 1200, 1000, 1500, n_skus=5)
    fitness_consistency(rng, "[fit] geom single-mode pop=40", boxes, pal, cfg_geom, 40,
                        n_modes=6, use_multi_decoder=False, use_top_k=False)
    # multi-decoder geometric, best-block
    fitness_consistency(rng, "[fit] geom multi-decoder pop=40 (best-block)", boxes, pal, cfg_geom, 40,
                        n_modes=6, use_multi_decoder=True, use_top_k=False)
    # multi-decoder geometric, top-K (per-chrom mode-5 path)
    fitness_consistency(rng, "[fit] geom multi-decoder pop=40 (top-K)", boxes, pal, cfg_geom, 40,
                        n_modes=6, use_multi_decoder=True, use_top_k=True)
    # pop_size edge cases
    fitness_consistency(rng, "[fit] geom multi-decoder pop=1 (top-K)", boxes, pal, cfg_geom, 1,
                        n_modes=6, use_multi_decoder=True, use_top_k=True)
    fitness_consistency(rng, "[fit] geom multi-decoder pop=2 (top-K)", boxes, pal, cfg_geom, 2,
                        n_modes=6, use_multi_decoder=True, use_top_k=True)
    # all-same-SKU and all-distinct
    boxes_same = make_boxes(rng, 50, 1200, 1000, 1500, n_skus=1)
    fitness_consistency(rng, "[fit] geom all-same-SKU multi (top-K)", boxes_same, pal, cfg_geom, 24,
                        n_modes=6, use_multi_decoder=True, use_top_k=True)
    boxes_distinct = make_boxes(rng, 50, 1200, 1000, 1500, n_skus=50)
    fitness_consistency(rng, "[fit] geom all-distinct-SKU multi (top-K)", boxes_distinct, pal, cfg_geom, 24,
                        n_modes=6, use_multi_decoder=True, use_top_k=True)

    # constrained fitness consistency
    print("\n=== 7b. decode_population_fitness vs per-chrom loop (CONSTRAINED) ===")
    pal_w = Pallet(length=1200, width=1000, height=1500, max_weight=200.0)
    cfg_cstr = PackerConfig(support_ratio=0.75, require_centroid_supported=True,
                            cog_envelope_fraction=0.25)
    boxes_c = make_boxes(rng, 50, 1200, 1000, 1500, n_skus=5,
                         weighted=True, mlot=True, rfs=True)
    fitness_consistency(rng, "[fit] cstr single-mode pop=30", boxes_c, pal_w, cfg_cstr, 30,
                        n_modes=6, use_multi_decoder=False, use_top_k=False)
    fitness_consistency(rng, "[fit] cstr multi-decoder pop=30", boxes_c, pal_w, cfg_cstr, 30,
                        n_modes=6, use_multi_decoder=True, use_top_k=False)

    # --------------------------------------------------------------------
    # 8. NON-INTEGER box dims fitness consistency. This is the one place the
    #    batch int-volume vs _fitness_pallet1 float-volume may diverge.
    #    Reported separately as it may be a KNOWN architectural gap, NOT a
    #    batch/serial decode bug.
    # --------------------------------------------------------------------
    print("\n=== 8. fitness consistency with NON-INTEGER dims ===")
    boxes_f = make_boxes(rng, 50, 1200, 1000, 1500, n_skus=5, integer=False)
    fitness_consistency(rng, "[fit] NON-INT geom single-mode pop=30", boxes_f, pal, cfg_geom, 30,
                        n_modes=6, use_multi_decoder=False, use_top_k=False)
    fitness_consistency(rng, "[fit] NON-INT geom multi-decoder pop=30 (top-K)", boxes_f, pal, cfg_geom, 30,
                        n_modes=6, use_multi_decoder=True, use_top_k=True)

    # --------------------------------------------------------------------
    print("\n=== SUMMARY ===")
    print(f"  total checks: {N_CHECKS}")
    print(f"  failures:     {N_FAIL}")
    if FAILS:
        print("  WORST OFFENDERS:")
        for label, detail in FAILS[:10]:
            print(f"    {label}: {detail}")
    print(f"\n=== RESULT: {'PASS' if N_FAIL == 0 else 'FAIL'} ===")
    return 0 if N_FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
