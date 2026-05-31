"""Controlled timing + throughput suite for the Cython BRKGA pallet packer.

Run inside Docker. Designed to be invoked SEPARATELY per OMP setting so the
thread-scaling sweep is clean (OpenMP thread count is fixed at process start):

    # decode throughput (per-mode, batch vs per-chromosome)
    docker run --rm -e OMP_NUM_THREADS=8 -v "$(pwd)":/app -w /app pallet-packer:dev \
        python scripts/_verify/timing_suite.py decode-throughput

    # OMP thread scaling (run once per thread count; tabulate externally)
    for t in 1 2 4 8; do docker run --rm -e OMP_NUM_THREADS=$t ... \
        python scripts/_verify/timing_suite.py omp-scaling ; done

    # single-generation (hot-loop) time vs N — pure throughput at scale
    docker run --rm -e OMP_NUM_THREADS=8 ... \
        python scripts/_verify/timing_suite.py n-scaling

    # real-world wall-clock per solve (BR + industry)
    docker run --rm -e OMP_NUM_THREADS=8 ... \
        python scripts/_verify/timing_suite.py realworld --time 20

All timings use time.perf_counter and report median of repeats where relevant.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit
from pallet_packer import validate, Box, Pallet, PackerConfig, ALL_ROTATIONS
from pallet_packer._brkga_core import jit_decoders_geom_cy as geom
from pallet_packer._brkga_core import jit_decoders_cstr_cy as cstr
from pallet_packer._brkga_core.dispatch import decode_population_fitness
from pallet_packer._brkga_core.precompute import (
    precompute_box_dims_and_sku, precompute_constraint_arrays)
from benchmarks.br import parse_thpack, geometric_only_config, first_pallet_utilization
from benchmarks.industry import cases as industry_cases


def _omp():
    return os.environ.get("OMP_NUM_THREADS", "default")


def make_population(rng, pop_size, n_boxes, L, W, H):
    n_rots = np.full(n_boxes, 6, dtype=np.int64)
    dims = np.zeros((n_boxes, 6, 3), dtype=np.int64)
    for i in range(n_boxes):
        bx = int(rng.integers(40, max(41, L // 4)))
        by = int(rng.integers(40, max(41, W // 4)))
        bz = int(rng.integers(40, max(41, H // 4)))
        rots = [(bx, by, bz), (bx, bz, by), (by, bx, bz),
                (by, bz, bx), (bz, bx, by), (bz, by, bx)]
        for r in range(6):
            dims[i, r] = rots[r]
    chroms = rng.random((pop_size, n_boxes)).astype(np.float64)
    return chroms, n_rots, dims


def time_call(fn, n_runs=7, warmup=2):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), min(ts)


# ---------------------------------------------------------------------------
def cmd_decode_throughput(args):
    print(f"=== decode throughput (OMP={_omp()}) ===")
    rng = np.random.default_rng(7)
    pop, n, L, W, H, mp = 600, 100, 1000, 800, 600, 1
    chroms, n_rots, dims = make_population(rng, pop, n, L, W, H)
    sku = rng.integers(0, 5, n).astype(np.int64)
    sku_bb = np.array([[2, 2, 2, 0]] * 5, dtype=np.int64)
    warmup_jit()
    po = np.zeros((pop, n, 6), dtype=np.int64)
    nb = np.zeros(pop, dtype=np.int64)

    def batch_layer():
        geom.decode_batch_layer_njit(chroms, n_rots, dims, L, W, H, mp, po, nb)

    def batch_blocks():
        geom.decode_batch_blocks_njit_mode(chroms, n_rots, dims, sku, L, W, H, mp, po, nb, 5)

    def batch_precomp():
        geom.decode_batch_precomputed_blocks_njit_mode(
            chroms, n_rots, dims, sku, sku_bb, L, W, H, mp, po, nb, 5)

    for label, fn in [("mode3 layer", batch_layer),
                      ("mode4 blocks", batch_blocks),
                      ("mode5 precomp", batch_precomp)]:
        med, best = time_call(fn)
        dps = pop / med
        print(f"  batch {label:16s}: {med*1e3:8.2f} ms / {pop} chroms "
              f"-> {dps:10.0f} decodes/sec  (best {best*1e3:.2f} ms)")


def cmd_omp_scaling(args):
    """Time a fixed batch workload; caller varies OMP_NUM_THREADS."""
    rng = np.random.default_rng(11)
    pop, n, L, W, H, mp = 600, 120, 1000, 800, 600, 1
    chroms, n_rots, dims = make_population(rng, pop, n, L, W, H)
    warmup_jit()
    po = np.zeros((pop, n, 6), dtype=np.int64)
    nb = np.zeros(pop, dtype=np.int64)
    weights = rng.uniform(0.5, 5.0, n).astype(np.float64)
    mlot = np.full(n, 50.0, dtype=np.float64)
    rfs = np.zeros(n, dtype=np.int64)

    def batch_layer():
        geom.decode_batch_layer_njit(chroms, n_rots, dims, L, W, H, mp, po, nb)

    def batch_cstr0():
        cstr.decode_batch_njit_mode_cstr(
            chroms, n_rots, dims, L, W, H, mp, 0,
            weights, mlot, rfs, 1e18, 0.5, 0,
            -1e18, 1e18, -1e18, 1e18, 0.0, 0, po, nb)

    ml, _ = time_call(batch_layer, n_runs=9)
    mc, _ = time_call(batch_cstr0, n_runs=9)
    print(f"OMP={_omp():>3}  layer_batch={ml*1e3:8.2f}ms  "
          f"cstr0_batch={mc*1e3:8.2f}ms  (pop={pop} n={n})")


def gen_instance(n_boxes, rng, n_skus=8):
    """A realistic heterogeneous instance for scaling tests."""
    sku_dims = []
    for _ in range(n_skus):
        sku_dims.append((int(rng.integers(80, 400)),
                         int(rng.integers(80, 350)),
                         int(rng.integers(60, 300))))
    boxes = []
    for i in range(n_boxes):
        l, w, h = sku_dims[i % n_skus]
        boxes.append(Box(id=f"B{i}", length=l, width=w, height=h, weight=0.0,
                         allowed_rotations=ALL_ROTATIONS))
    pallet = Pallet(length=1200, width=1000, height=1500)
    return boxes, pallet


def cmd_n_scaling(args):
    """Pure hot-loop throughput: time ONE decode_population_fitness call vs N."""
    print(f"=== single-generation (hot loop) time vs N (OMP={_omp()}) ===")
    print(f"{'N':>6} {'pop':>5} {'gen_ms':>10} {'decodes/s':>12} {'setup_ms':>10}")
    rng = np.random.default_rng(23)
    cfg = geometric_only_config(max_pallets=1)
    warmup_jit()
    pop_size = 100
    # 5000 is intentionally excluded: a single generation-0 decode at N=5000
    # does not complete in <1h (O(N^2) decode). Measured separately with a
    # hard timeout via the 'one-decode' subcommand.
    for n_boxes in [50, 100, 200, 500, 1000, 2000]:
        boxes, pallet = gen_instance(n_boxes, rng)
        t0 = time.perf_counter()
        n_rots, dims, sku = precompute_box_dims_and_sku(boxes)
        w, m, r, pmw, hc = precompute_constraint_arrays(boxes, pallet)
        setup = time.perf_counter() - t0
        chrom_size = 2 * n_boxes + 1
        population = rng.random((pop_size, chrom_size))

        def one_gen():
            decode_population_fitness(
                population, boxes, pallet, cfg, n_rots, dims, max_pallets=1,
                use_multi_decoder=True, n_modes=6, sku_id_per_box=sku,
                weights=w, mlot=m, rfs=r, pallet_max_weight=pmw,
                has_constraints=hc)
        med, best = time_call(one_gen, n_runs=5, warmup=1)
        dps = pop_size / med
        print(f"{n_boxes:>6} {pop_size:>5} {med*1e3:>10.1f} {dps:>12.0f} {setup*1e3:>10.1f}",
              flush=True)


def cmd_realworld(args):
    """Per-instance wall-clock + quality on BR + industry at a fixed budget."""
    t_budget = args.time
    print(f"=== real-world wall-clock (budget={t_budget}s, OMP={_omp()}) ===")
    warmup_jit()
    cfg = geometric_only_config(max_pallets=1)
    print("\n-- BR (geometric, single pallet) --")
    print(f"{'inst':10} {'N':>5} {'wall_s':>8} {'util%':>7} {'errs':>5}")
    for sid, fn in [("BR1", "thpack1"), ("BR3", "thpack3"),
                    ("BR5", "thpack5"), ("BR7", "thpack7")]:
        insts = parse_thpack(f"benchmarks/data/{fn}.txt", set_id=sid)[:3]
        for inst in insts:
            t0 = time.perf_counter()
            r = brkga_pack_v35(inst.boxes, inst.pallet, cfg,
                               time_limit_s=t_budget, max_pallets=1,
                               population_size=600, n_populations=3, patience=200,
                               local_search_budget_s=4.0,
                               seed=42 + inst.instance_id, n_modes=6, verbose=False)
            wall = time.perf_counter() - t0
            util = first_pallet_utilization(r, inst.pallet) * 100
            errs = len(validate(r, inst.pallet, cfg))
            print(f"{sid}#{inst.instance_id:<7d} {len(inst.boxes):>5} {wall:>8.2f} "
                  f"{util:>7.2f} {errs:>5}", flush=True)

    print("\n-- Industry (constrained, multi-pallet) --")
    print(f"{'case':38} {'N':>5} {'wall_s':>8} {'pallets':>8} {'unpkd':>6} {'errs':>5}")
    for c in industry_cases():
        t0 = time.perf_counter()
        r = brkga_pack_v35(c.boxes, c.pallet, c.config,
                           time_limit_s=t_budget, max_pallets=10,
                           population_size=300, n_populations=3, patience=150,
                           seed=42, n_modes=6, verbose=False)
        wall = time.perf_counter() - t0
        errs = len(validate(r, c.pallet, c.config))
        print(f"{c.name[:36]:38} {len(c.boxes):>5} {wall:>8.2f} "
              f"{len(r.pallets):>8} {len(r.unpacked):>6} {errs:>5}", flush=True)


def cmd_one_decode(args):
    """Time a SINGLE decode at a given N (run under external `timeout`)."""
    print(f"=== single-decode latency at N={args.n} (OMP={_omp()}) ===", flush=True)
    rng = np.random.default_rng(31)
    cfg = geometric_only_config(max_pallets=200)
    warmup_jit()
    boxes, pallet = gen_instance(args.n, rng)
    from pallet_packer._brkga_core.dispatch import decode_chromosome
    n_rots, dims, sku = precompute_box_dims_and_sku(boxes)
    chrom = rng.random(2 * args.n + 1)
    print(f"  building one PackResult (decode_chromosome) for N={args.n}...", flush=True)
    t0 = time.perf_counter()
    decode_chromosome(chrom, boxes, pallet, cfg, n_rots, dims, mode=4,
                      max_pallets=200, sku_id_per_box=sku)
    dt = time.perf_counter() - t0
    print(f"  decode_chromosome(mode4) N={args.n}: {dt:.3f} s for ONE decode", flush=True)
    print(f"  => a single pop=100 generation would cost ~{dt*100:.0f} s", flush=True)


def cmd_float_int(args):
    """Independent minimal repro of the int-decode / float-validate defect."""
    print(f"=== float/int repro (OMP={_omp()}) ===")
    warmup_jit()
    cfg = geometric_only_config(max_pallets=1)
    for label, dim, n, pL in [("integer 100^3", 100.0, 64, 800),
                              ("fractional 100.49^3", 100.49, 64, 804),
                              ("realistic 333.33x250.7x124.9", None, 33, 1200)]:
        if dim is not None:
            boxes = [Box(id=f"b{i}", length=dim, width=dim, height=100.0,
                         allowed_rotations=ALL_ROTATIONS) for i in range(n)]
            pallet = Pallet(length=pL, width=pL, height=100)
        else:
            boxes = [Box(id=f"b{i}", length=333.33, width=250.7, height=124.9,
                         allowed_rotations=ALL_ROTATIONS) for i in range(n)]
            pallet = Pallet(length=1200, width=1000, height=1500)
        r = brkga_pack_v35(boxes, pallet, cfg, time_limit_s=6.0, max_pallets=1,
                           population_size=200, n_populations=2, patience=100,
                           seed=42, n_modes=6, verbose=False)
        errs = validate(r, pallet, cfg)
        # independent worst-overlap measure
        worst = 0.0
        pls = r.pallets[0].placements if r.pallets else []
        for i in range(len(pls)):
            for j in range(i + 1, len(pls)):
                a, b = pls[i], pls[j]
                ox = min(a.x2, b.x2) - max(a.x, b.x)
                oy = min(a.y2, b.y2) - max(a.y, b.y)
                oz = min(a.z2, b.z2) - max(a.z, b.z)
                if ox > 1e-6 and oy > 1e-6 and oz > 1e-6:
                    worst = max(worst, min(ox, oy, oz))
        print(f"  {label:34s}: validator_errors={len(errs):4d}  worst_overlap={worst:.3f}u  placed={len(pls)}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("decode-throughput")
    sub.add_parser("omp-scaling")
    sub.add_parser("n-scaling")
    rw = sub.add_parser("realworld")
    rw.add_argument("--time", type=float, default=20.0)
    od = sub.add_parser("one-decode")
    od.add_argument("--n", type=int, default=5000)
    sub.add_parser("float-int")
    args = ap.parse_args()
    {
        "decode-throughput": cmd_decode_throughput,
        "omp-scaling": cmd_omp_scaling,
        "n-scaling": cmd_n_scaling,
        "realworld": cmd_realworld,
        "one-decode": cmd_one_decode,
        "float-int": cmd_float_int,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
