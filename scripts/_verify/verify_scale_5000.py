"""verify_scale_5000.py — adversarial scaling probe for brkga_pack_v35.

Mandate: probe the UNTESTED real-world scale target. The service limit is
5000 boxes; the largest *tested* case is 200 (industry IND2/IND5). This probe
runs the full solver at N = 500, 1000, 2000, 5000 boxes with a realistic
multi-SKU mix and asks, adversarially:

  - Does it COMPLETE without crash / OOM within a generous time_limit_s?
  - Wall-clock, peak RSS (resource.getrusage), BRKGA generations reached
    (parsed from verbose stdout), final pallet-1 utilization, validator
    cleanliness (the INDEPENDENT ground-truth validator).
  - Does it DEGENERATE at scale (0 generations, all time in setup, trivial
    or empty packing)?
  - int64 overflow risk in the volume math at scale (sub-test 0, analytic).

Run (heavy — uses OMP_NUM_THREADS=4 per mandate):
    docker run --rm -e OMP_NUM_THREADS=4 -e PYTHONHASHSEED=0 \
        -v "$(pwd)":/app -w /app pallet-packer:dev \
        python scripts/_verify/verify_scale_5000.py
"""
from __future__ import annotations

import io
import os
import sys
import contextlib
import gc
import re
import resource
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from pallet_packer import Box, Pallet, PackerConfig, THIS_SIDE_UP, ALL_ROTATIONS, validate
from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit
from pallet_packer._brkga_core import dispatch
from pallet_packer._brkga_core.precompute import precompute_constraint_arrays


# Standard Euro pallet, 1.2 x 1.0 x 1.5 m, weight-capped (so a realistic
# multi-pallet logistics workload — constraints ON, exercises the cstr path).
EURO = Pallet(length=1200, width=1000, height=1500, max_weight=1000.0)

# Realistic SKU catalog (mm, kg). Mix of small/medium/large, varied mlot.
# Dims chosen to tile reasonably so packings aren't pathologically sparse.
SKU_CATALOG = [
    # (l,   w,   h,   weight, max_load_on_top)
    (200, 150, 100, 0.6, 15.0),
    (300, 250, 200, 1.5, 20.0),
    (400, 300, 300, 3.0, 10.0),
    (150, 150, 100, 0.8, 0.0),    # fragile
    (250, 200, 150, 1.2, 18.0),
    (380, 300, 250, 2.0, 50.0),
    (200, 200, 200, 1.0, 25.0),
    (350, 250, 300, 1.8, 30.0),
]


def maxrss_mb() -> float:
    """Peak resident set size in MB (Linux ru_maxrss is in KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def make_workload(n: int, seed: int = 7) -> list:
    """Deterministic realistic multi-SKU workload of exactly n boxes."""
    rng = np.random.default_rng(seed)
    boxes = []
    for i in range(n):
        l, w, h, wt, mlot = SKU_CATALOG[int(rng.integers(0, len(SKU_CATALOG)))]
        boxes.append(Box(
            id=f"B{i:05d}", length=l, width=w, height=h, weight=wt,
            allowed_rotations=THIS_SIDE_UP, max_load_on_top=mlot,
        ))
    return boxes


_GEN_RE = re.compile(r"gen (\d+)")


def parse_max_gen(verbose_text: str) -> int:
    """Highest BRKGA generation index seen in verbose output (-1 if none)."""
    gens = [int(m) for m in _GEN_RE.findall(verbose_text)]
    return max(gens) if gens else -1


# ===========================================================================
# Sub-test 0 — int64 overflow analysis for the batch volume math.
# ===========================================================================
def test_overflow():
    print("=== Sub-test 0: int64 overflow risk in volume math ===")
    ok = True
    INT64_MAX = np.iinfo(np.int64).max  # 9.223e18

    # (a) Single-box volume: dims are int64; product of 3. Worst realistic
    #     box ~ a pallet-filling 1200x1000x1500.
    worst_box_vol = 1200 * 1000 * 1500
    print(f"  worst single-box volume = {worst_box_vol:.3e} "
          f"(int64 max {INT64_MAX:.3e})")
    if worst_box_vol >= INT64_MAX:
        print("  FAIL: single box volume overflows int64"); ok = False

    # (b) Summed volume across N boxes (the (vols*mask).sum(axis=1) reduction).
    #     Cap at N=5000 each pallet-sized (absurd upper bound).
    summed = worst_box_vol * 5000
    print(f"  sum of 5000 pallet-vol boxes = {summed:.3e}")
    if summed >= INT64_MAX:
        print("  FAIL: summed volume can overflow int64 at N=5000"); ok = False
    else:
        margin = INT64_MAX / summed
        print(f"  no overflow; headroom factor = {margin:.1f}x  PASS")

    # (c) Empirically reproduce the exact reduction np uses, with a degenerate
    #     huge-dim workload, to confirm numpy does NOT silently wrap.
    n = 5000
    dims = np.zeros((n, 1, 3), dtype=np.int64)
    dims[:, 0, 0] = 1200; dims[:, 0, 1] = 1000; dims[:, 0, 2] = 1500
    orders = np.arange(n, dtype=np.int64)[None, :]
    rot_idx = np.zeros((1, n), dtype=np.int64)
    sel = dims[orders, rot_idx]
    vols = sel[:, :, 0] * sel[:, :, 1] * sel[:, :, 2]
    used = vols.sum(axis=1).astype(np.float64)
    expect = float(worst_box_vol) * n
    rel = abs(used[0] - expect) / expect
    print(f"  empirical int64 reduction: got {used[0]:.6e} expect {expect:.6e} "
          f"rel_err={rel:.2e}")
    if rel > 1e-9:
        print("  FAIL: int64 volume reduction wrapped/mismatched"); ok = False
    else:
        print("  empirical reduction exact  PASS")

    print(f"  [Sub-test 0] {'PASS' if ok else 'FAIL'}\n")
    return ok


# ===========================================================================
# Sub-test 1 — memory footprint of the hot-loop buffers at scale.
# ===========================================================================
def test_buffer_footprint():
    print("=== Sub-test 1: hot-loop buffer footprint estimate ===")
    # placements_out_all = (pop_size, n, 6) int64, plus a per-mode duplicate.
    for pop in (300, 600):
        for n in (500, 1000, 2000, 5000):
            mb = pop * n * 6 * 8 / 1e6
            tag = ""
            if mb > 500:
                tag = "  <-- >500MB single buffer"
            print(f"  pop={pop:4d} n={n:5d}: placements_out_all ~ {mb:8.1f} MB{tag}")
    print("  (informational — driver default population_size=600; this probe "
          "uses scale-aware sizing below)\n")
    return True


# ===========================================================================
# Sub-test 2 — full-solve scaling sweep.
# ===========================================================================
def run_scale(n, time_limit, pop, n_pop):
    boxes = make_workload(n)
    cfg = PackerConfig()
    # confirm this workload actually drives the constraint path
    _, _, _, _, has_cstr = precompute_constraint_arrays(boxes, EURO)

    gc.collect()
    rss0 = maxrss_mb()
    buf = io.StringIO()
    t0 = time.time()
    crashed = None
    r = None
    try:
        with contextlib.redirect_stdout(buf):
            r = brkga_pack_v35(
                boxes, EURO, cfg,
                time_limit_s=time_limit,
                max_pallets=200,          # generous — let it open as many as it needs
                seed=42,
                population_size=pop,
                n_populations=n_pop,
                patience=10_000,          # don't let patience cut a scale run short
                local_search_budget_s=min(8.0, time_limit / 4),
                use_v2_seed=None,         # auto (constraints present -> True)
                n_modes=6,
                verbose=True,
            )
    except MemoryError as e:
        crashed = f"MemoryError: {e}"
    except Exception as e:  # noqa: BLE001 — adversarial: catch anything
        crashed = f"{type(e).__name__}: {e}"
    wall = time.time() - t0
    rss_peak = maxrss_mb()
    vtext = buf.getvalue()
    max_gen = parse_max_gen(vtext)

    row = {
        "n": n, "has_cstr": has_cstr, "time_limit": time_limit,
        "pop": pop, "n_pop": n_pop, "wall": wall,
        "rss0_mb": rss0, "rss_peak_mb": rss_peak,
        "rss_delta_mb": rss_peak - rss0, "max_gen": max_gen,
        "crashed": crashed,
    }
    if r is not None:
        n_placed = sum(len(p.placements) for p in r.pallets)
        cap1 = EURO.length * EURO.width * EURO.height
        used1 = (sum(b.box.volume for b in r.pallets[0].placements)
                 if r.pallets else 0.0)
        util1 = used1 / cap1 if cap1 > 0 else 0.0
        errs = validate(r, EURO, cfg)
        row.update({
            "n_pallets": len(r.pallets), "n_placed": n_placed,
            "n_unpacked": len(r.unpacked), "util1": util1,
            "n_errors": len(errs),
            "first_errs": errs[:3],
        })
    return row, vtext


def test_scale():
    print("=== Sub-test 2: full-solve scaling sweep (OMP=%s) ===" %
          os.environ.get("OMP_NUM_THREADS", "default"))
    # Scale-aware sizing: keep population modest so buffers stay sane, but
    # large enough that BRKGA is meaningful. Generous time budgets.
    plan = [
        # n,    time_limit, pop, n_pop
        (500,   60.0, 300, 3),
        (1000,  90.0, 200, 2),
        (2000, 120.0, 150, 2),
        (5000, 120.0, 100, 2),
    ]
    rows = []
    for n, tl, pop, n_pop in plan:
        print(f"\n--- N={n}  time_limit={tl}s  pop={pop}  n_pop={n_pop} ---",
              flush=True)
        row, _ = run_scale(n, tl, pop, n_pop)
        rows.append(row)
        if row["crashed"]:
            print(f"  N={n}: CRASHED -> {row['crashed']}  "
                  f"wall={row['wall']:.1f}s peak_rss={row['rss_peak_mb']:.0f}MB",
                  flush=True)
            continue
        # degeneracy heuristics
        degenerate = []
        if row["max_gen"] < 0:
            degenerate.append("0 BRKGA generations (no gen line emitted)")
        if row["n_placed"] == 0:
            degenerate.append("nothing placed")
        # if wall is near time_limit but max_gen is 0 -> all time in setup
        if row["max_gen"] <= 0 and row["wall"] > 0.6 * row["time_limit"]:
            degenerate.append("time spent without completing a generation")
        print(f"  N={n}: wall={row['wall']:.1f}s  peak_rss={row['rss_peak_mb']:.0f}MB "
              f"(Δ{row['rss_delta_mb']:.0f}MB)  max_gen={row['max_gen']}  "
              f"pallets={row['n_pallets']}  placed={row['n_placed']}/{n}  "
              f"unpacked={row['n_unpacked']}  util_p1={row['util1']*100:.1f}%  "
              f"errs={row['n_errors']}", flush=True)
        if row["n_errors"]:
            print(f"    VALIDATOR ERRORS (first 3): {row['first_errs']}",
                  flush=True)
        if degenerate:
            print(f"    DEGENERACY FLAGS: {'; '.join(degenerate)}", flush=True)

    # Scaling table
    print("\n=== SCALING TABLE  N -> (wall, peak_rss, gens, util_p1, valid) ===")
    print(f"{'N':>6} {'wall_s':>8} {'peakRSS_MB':>11} {'gens':>6} "
          f"{'pallets':>8} {'placed':>8} {'unp':>6} {'util_p1':>8} "
          f"{'errs':>5} {'status':>10}")
    all_ok = True
    for row in rows:
        if row["crashed"]:
            print(f"{row['n']:>6} {row['wall']:>8.1f} "
                  f"{row['rss_peak_mb']:>11.0f} {'-':>6} {'-':>8} {'-':>8} "
                  f"{'-':>6} {'-':>8} {'-':>5} {'CRASH':>10}")
            all_ok = False
            continue
        # A run is "healthy" if: completed, validator-clean, made >=1 gen OR
        # placed a non-trivial fraction (smart-init/v2 can pack well even if
        # the BRKGA loop didn't get a full generation in).
        healthy = (row["n_errors"] == 0
                   and row["n_placed"] > 0
                   and not (row["max_gen"] < 0 and row["n_placed"] == 0))
        status = "OK" if healthy else "DEGRADED"
        if not healthy:
            all_ok = False
        print(f"{row['n']:>6} {row['wall']:>8.1f} {row['rss_peak_mb']:>11.0f} "
              f"{row['max_gen']:>6} {row['n_pallets']:>8} {row['n_placed']:>8} "
              f"{row['n_unpacked']:>6} {row['util1']*100:>7.1f}% "
              f"{row['n_errors']:>5} {status:>10}")
    print()
    return all_ok, rows


# ===========================================================================
# Sub-test 3 — per-decode cost scaling (isolates WHY full solve degrades).
#
# The driver checks the time budget only at the TOP of each BRKGA generation
# (driver.py:411). One generation evaluates the whole population via a single
# decode_population_fitness call. If ONE generation costs more than the budget,
# time_limit_s is silently blown and the solver makes <=1 generation. This
# sub-test times a single decode_population_fitness over pop_size=1 (one decode)
# and a small population, at each N, to show the per-decode cost growth.
# ===========================================================================
def test_decode_cost():
    print("=== Sub-test 3: per-decode cost scaling (mode 0 geom-ish path) ===")
    from pallet_packer._brkga_core.precompute import precompute_box_dims_and_sku
    rows = []
    for n in (500, 1000, 2000, 5000):
        boxes = make_workload(n)
        n_rots_arr, dims_all, sku_id = precompute_box_dims_and_sku(boxes)
        w, m, r, pmw, hc = precompute_constraint_arrays(boxes, EURO)
        chrom_size = 2 * n + 1
        for pop in (1, 8):
            rng = np.random.default_rng(0)
            population = rng.random((pop, chrom_size)).astype(np.float64)
            # warm + time
            n_skus = int(sku_id.max()) + 1
            t0 = time.time()
            fits = dispatch.decode_population_fitness(
                population, boxes, EURO, PackerConfig(),
                n_rots_arr, dims_all, max_pallets=200,
                use_multi_decoder=True, n_modes=6,
                sku_id_per_box=sku_id,
                weights=w, mlot=m, rfs=r, pallet_max_weight=pmw,
                has_constraints=hc, support_ratio=0.8,
            )
            dt = time.time() - t0
            per = dt / pop
            rows.append((n, pop, dt, per))
            print(f"  N={n:5d} pop={pop:2d}: total={dt:7.3f}s  per-decode={per:7.3f}s")
    # estimate: one generation of pop=100 at N=5000
    per5000 = next(p for (n, pop, dt, p) in rows if n == 5000 and pop == 8)
    est_gen = per5000 * 100
    print(f"  EST one generation (pop=100) @ N=5000 ~ {est_gen:.0f}s "
          f"({est_gen/60:.1f} min) -- vs typical time_limit_s=120s")
    if est_gen > 120:
        print("  -> a single generation EXCEEDS a 120s budget: time_limit_s "
              "cannot be honored at N=5000 (solver makes <=1 generation).")
    print()
    return rows


def main():
    print("backend: decode_njit_mode from", dispatch.decode_njit_mode.__module__,
          " _BATCH_AVAILABLE =", dispatch._BATCH_AVAILABLE)
    print("OMP_NUM_THREADS =", os.environ.get("OMP_NUM_THREADS", "default"))
    print("warming up JIT...", flush=True)
    warmup_jit()
    print()

    micro = "--micro" in sys.argv
    ok0 = test_overflow()
    ok1 = test_buffer_footprint()
    if micro:
        # Bounded probe: skip the multi-minute full solves, characterize the
        # per-decode cost that drives the degradation instead.
        test_decode_cost()
        print("=== (micro mode: full-solve sweep skipped) ===")
        return 0
    test_decode_cost()
    ok2, rows = test_scale()

    overall = ok0 and ok1 and ok2
    print(f"=== OVERALL: {'PASS' if overall else 'FAIL'} ===")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
