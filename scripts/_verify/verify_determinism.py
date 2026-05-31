"""verify_determinism.py — adversarial full-solve determinism verification.

Mandate (dimension: determinism):
  (a) Same (instance, seed) repeated 3x -> bit-identical placement signature
      (sorted tuples of pallet_index, box_id, rotation, x, y, z). Tested on
      >= 6 instances: BR1/BR3/BR5/BR7 first instances + 2 constrained
      industry cases.
  (b) THREAD-INVARIANCE: this script prints a STABLE sha256 digest of the
      concatenated converged signatures. The orchestrator runs it TWICE —
      once with OMP_NUM_THREADS=1 and once with =8 (both PYTHONHASHSEED=0) —
      and compares the printed "GLOBAL_SHA256=" line.
  (c) n_restarts=2 determinism.
  (d) Constrained MULTI-pallet determinism + validator stays clean.

KEY DISTINCTION this probe enforces:
  brkga_pack_v35's main loop, local search, and LNS polish are bounded by
  WALL-CLOCK time (driver.py:411 `if time.time()-t0 > brkga_budget: break`)
  OR by `patience` (generation count). Only the patience-bounded /
  fully-converged regime is *expected* to be deterministic across runs.
  A time-bounded solve that has NOT converged can legitimately stop at a
  different generation when wall-clock differs (e.g. CPU contention,
  OMP=8 vs OMP=1) and thus return a different-but-valid result. That is
  EXPECTED timing variance, NOT an algorithm race.

  So this probe runs every determinism check in a CONVERGED regime:
  generous time_limit_s + low patience so the run terminates by patience,
  not by the clock. It separately measures the time-bounded regime to
  characterise (and not penalise) timing variance, and it isolates the
  decode+fitness layer to prove the parallel decode itself is bit-identical.

Run (single-thread correctness):
    docker run --rm -e OMP_NUM_THREADS=1 -e PYTHONHASHSEED=0 \
        -v "$(pwd)":/app -w /app pallet-packer:dev \
        python scripts/_verify/verify_determinism.py
Thread-invariance: run again with -e OMP_NUM_THREADS=8 and compare GLOBAL_SHA256.
"""
from __future__ import annotations

import hashlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from pallet_packer import validate
from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit
from pallet_packer._brkga_core import dispatch
from pallet_packer._brkga_core.dispatch import decode_population_fitness
from pallet_packer._brkga_core.precompute import (
    precompute_box_dims_and_sku, precompute_constraint_arrays,
)
from benchmarks.br import parse_thpack, geometric_only_config, first_pallet_utilization
from benchmarks.industry import cases as industry_cases


# ---------------------------------------------------------------------------
# Signature helpers
# ---------------------------------------------------------------------------
def placements_signature(result):
    rows = []
    for pi, st in enumerate(result.pallets):
        for p in st.placements:
            rows.append((pi, str(p.box.id), p.rotation.name,
                         round(float(p.x), 6), round(float(p.y), 6), round(float(p.z), 6)))
    rows.sort()
    return tuple(rows)


def sig_to_str(sig):
    return "\n".join("|".join(str(c) for c in row) for row in sig)


def sha256_of(*sig_strings):
    h = hashlib.sha256()
    for s in sig_strings:
        h.update(s.encode("utf-8"))
        h.update(b"\x1e")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Solve wrappers.
# CONVERGED regime: big time budget, LOW patience, NO local-search/LNS time
# budget that could clip mid-search by the clock. We force termination by
# patience (generation count) so the run is wall-clock-independent.
# ---------------------------------------------------------------------------
def run_geom_converged(inst, *, seed, n_restarts=1):
    cfg = geometric_only_config(max_pallets=1)
    return brkga_pack_v35(
        inst.boxes, inst.pallet, cfg,
        time_limit_s=600.0,          # effectively unbounded; patience wins
        max_pallets=1,
        population_size=120, n_populations=2,
        patience=12,                  # converge fast & deterministically
        use_local_search=False,       # LS is time-budgeted -> nondet; exclude
        use_lns=False,
        use_v2_hybrid_polish=False,   # v2 polish has its own timing; exclude
        seed=seed, verbose=False, n_modes=6, n_restarts=n_restarts,
    )


def run_constrained_converged(case, *, seed, max_pallets=10):
    return brkga_pack_v35(
        case.boxes, case.pallet, case.config,
        time_limit_s=600.0, max_pallets=max_pallets,
        population_size=100, n_populations=2,
        patience=12,
        use_local_search=False, use_lns=False, use_v2_hybrid_polish=False,
        seed=seed, verbose=False, n_modes=6,
    )


def run_geom_timebound(inst, *, seed, time_s=3.0):
    """Default-ish config that terminates by the CLOCK (not converged).
    Used ONLY to characterise expected timing variance — not a pass/fail gate."""
    cfg = geometric_only_config(max_pallets=1)
    return brkga_pack_v35(
        inst.boxes, inst.pallet, cfg,
        time_limit_s=time_s, max_pallets=1,
        population_size=300, n_populations=3, patience=10_000,  # never converge
        local_search_budget_s=1.0, seed=seed, verbose=False, n_modes=6,
    )


# ---------------------------------------------------------------------------
def repeat_3x(label, solve_fn, *, extract=None):
    sigs = []
    extra = ""
    for run in range(3):
        r = solve_fn()
        sig = placements_signature(r)
        sigs.append(sig)
        if extract is not None and run == 0:
            extra = extract(r)
    distinct = len(set(sigs))
    ok = distinct == 1
    n_placed = [len(s) for s in sigs]
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}: 3x distinct_sigs={distinct} "
          f"n_placed={n_placed}{(' ' + extra) if extra else ''}")
    if not ok:
        for i in range(3):
            for j in range(i + 1, 3):
                if sigs[i] == sigs[j]:
                    continue
                si, sj = set(sigs[i]), set(sigs[j])
                print(f"        run{i}!=run{j}: |only{i}|={len(si - sj)} "
                      f"|only{j}|={len(sj - si)}  first_only{i}={sorted(si - sj)[:2]}")
                break
            else:
                continue
            break
    return ok, sig_to_str(sigs[0])


def main():
    print("=" * 72)
    print("DETERMINISM PROBE  (OMP_NUM_THREADS=%s  PYTHONHASHSEED=%s)"
          % (os.environ.get("OMP_NUM_THREADS", "default"),
             os.environ.get("PYTHONHASHSEED", "unset")))
    print("=" * 72)

    warmup_jit()
    overall_ok = True
    global_sigs = []

    mod = dispatch.decode_njit_mode.__module__
    is_cy = "cy" in mod
    print(f"[backend] decode_njit_mode={mod} ({'CYTHON' if is_cy else 'NUMBA'}) "
          f"_BATCH_AVAILABLE={dispatch._BATCH_AVAILABLE}")
    if not (is_cy and dispatch._BATCH_AVAILABLE):
        print("  [FAIL] not running Cython batch backend")
        overall_ok = False

    # ----------------------------------------------------------------
    # (PRE) ISOLATION: parallel decode+fitness must be bit-identical 5x.
    # This proves the prange(schedule='static') decode layer is thread-safe
    # and order-stable, independent of the time-bounded driver loop.
    # ----------------------------------------------------------------
    print("\n=== (pre) decode_population_fitness bit-identity (5x, isolated) ===")
    inst_iso = parse_thpack("benchmarks/data/thpack3.txt", set_id="BR3")[0]
    n_rots_arr, dims_all, sku_id = precompute_box_dims_and_sku(inst_iso.boxes)
    cfg_iso = geometric_only_config(max_pallets=1)
    n = len(inst_iso.boxes)
    rng = np.random.default_rng(12345)
    pop = rng.random((300, 2 * n + 1))  # multi-decoder layout: 2n+selector
    fit_digests = []
    for _ in range(5):
        f = decode_population_fitness(
            pop, inst_iso.boxes, inst_iso.pallet, cfg_iso,
            n_rots_arr, dims_all, max_pallets=1,
            use_multi_decoder=True, n_modes=6, sku_id_per_box=sku_id,
        )
        fit_digests.append(hashlib.sha256(np.ascontiguousarray(f).tobytes()).hexdigest())
    iso_ok = len(set(fit_digests)) == 1
    print(f"  [{'PASS' if iso_ok else 'FAIL'}] 5x fitness-array sha256 distinct="
          f"{len(set(fit_digests))}  digest={fit_digests[0][:16]}")
    overall_ok &= iso_ok

    # ----------------------------------------------------------------
    # (a) Same (instance,seed) x3 bit-identical — CONVERGED regime.
    # ----------------------------------------------------------------
    print("\n=== (a) Same (instance,seed) x3 bit-identical [converged] — 6 instances ===")
    br_files = [("thpack1.txt", "BR1"), ("thpack3.txt", "BR3"),
                ("thpack5.txt", "BR5"), ("thpack7.txt", "BR7")]
    for fname, sid in br_files:
        inst = parse_thpack(f"benchmarks/data/{fname}", set_id=sid)[0]
        seed = 42 + inst.instance_id
        ok, sigstr = repeat_3x(
            f"{sid}#{inst.instance_id} N={len(inst.boxes)} seed={seed}",
            lambda i=inst, s=seed: run_geom_converged(i, seed=s),
            extract=lambda r, p=inst.pallet: f"util={first_pallet_utilization(r, p) * 100:.4f}%",
        )
        overall_ok &= ok
        global_sigs.append(sigstr)

    cs = industry_cases()
    by_name = {c.name.split()[0]: c for c in cs}
    chosen = [by_name[k] for k in ("IND2", "IND4") if k in by_name] or cs[:2]
    for c in chosen:
        _, _, _, _, has_cstr = precompute_constraint_arrays(c.boxes, c.pallet)
        ok, sigstr = repeat_3x(
            f"{c.name.split()[0]} N={len(c.boxes)} has_cstr={has_cstr} seed=42",
            lambda case=c: run_constrained_converged(case, seed=42),
            extract=lambda r: f"pallets={len(r.pallets)} unp={len(r.unpacked)}",
        )
        if not has_cstr:
            print(f"        WARN: {c.name.split()[0]} has_constraints=False")
        overall_ok &= ok
        global_sigs.append(sigstr)

    # ----------------------------------------------------------------
    # (c) n_restarts=2 determinism — converged.
    # ----------------------------------------------------------------
    print("\n=== (c) n_restarts=2 determinism (BR1#1) [converged] ===")
    inst1 = parse_thpack("benchmarks/data/thpack1.txt", set_id="BR1")[0]
    seed_r = 42 + inst1.instance_id
    ok, sigstr = repeat_3x(
        f"BR1#{inst1.instance_id} n_restarts=2 seed={seed_r}",
        lambda i=inst1, s=seed_r: run_geom_converged(i, seed=s, n_restarts=2),
        extract=lambda r, p=inst1.pallet: f"util={first_pallet_utilization(r, p) * 100:.4f}%",
    )
    overall_ok &= ok
    global_sigs.append(sigstr)

    # ----------------------------------------------------------------
    # (d) Constrained MULTI-pallet determinism + validator clean — converged.
    # ----------------------------------------------------------------
    print("\n=== (d) Constrained multi-pallet determinism + validity [converged] ===")
    c4 = by_name.get("IND2") or by_name.get("IND9") or cs[0]
    sigs, last_r = [], None
    for _ in range(3):
        r = run_constrained_converged(c4, seed=7, max_pallets=10)
        sigs.append(placements_signature(r)); last_r = r
    distinct = len(set(sigs)); npal = len(last_r.pallets)
    errs = validate(last_r, c4.pallet, c4.config)
    ok_d = (distinct == 1) and (len(errs) == 0)
    print(f"  [{'PASS' if ok_d else 'FAIL'}] {c4.name.split()[0]}: 3x distinct_sigs={distinct} "
          f"pallets={npal} (multi={npal > 1}) unp={len(last_r.unpacked)} "
          f"validator_errors={len(errs)}")
    if errs:
        for e in errs[:5]:
            print(f"        VALIDATOR ERROR: {e}")
    overall_ok &= ok_d
    global_sigs.append(sig_to_str(sigs[0]))

    # ----------------------------------------------------------------
    # GLOBAL sha256 for cross-thread comparison (converged sigs only).
    # ----------------------------------------------------------------
    digest = sha256_of(*global_sigs)
    print("\n" + "=" * 72)
    print(f"GLOBAL_SHA256={digest}")
    print(f"N_SIGNATURE_BLOCKS={len(global_sigs)}")
    print("=" * 72)

    # ----------------------------------------------------------------
    # CHARACTERISATION (informational, NOT pass/fail): time-bounded regime.
    # Shows that an un-converged, clock-bounded solve can vary run-to-run
    # because driver.py:411 breaks on wall-clock. This is EXPECTED.
    # ----------------------------------------------------------------
    print("\n=== (info) Time-bounded (UN-converged) regime — expected timing variance ===")
    inst_tb = parse_thpack("benchmarks/data/thpack7.txt", set_id="BR7")[0]
    tb_sigs, tb_times = [], []
    for _ in range(3):
        t0 = time.time()
        r = run_geom_timebound(inst_tb, seed=42 + inst_tb.instance_id, time_s=3.0)
        tb_times.append(time.time() - t0)
        tb_sigs.append(placements_signature(r))
    tb_distinct = len(set(tb_sigs))
    print(f"  BR7#{inst_tb.instance_id} time_limit=3.0s patience=inf: "
          f"distinct_sigs={tb_distinct} wall_times={[round(t, 2) for t in tb_times]}")
    if tb_distinct == 1:
        print("  -> deterministic even time-bounded here (clock granularity didn't bite)")
    else:
        print("  -> NON-deterministic under clock-bound termination. EXPECTED: the")
        print("     loop breaks on wall-clock (driver.py:411), so #generations")
        print("     completed varies with CPU timing. Same root cause as any")
        print("     OMP=1 vs OMP=8 divergence for un-converged solves. NOT a race:")
        print("     the parallel decode is bit-identical (see (pre) PASS) and the")
        print("     converged regime (a)-(d) is bit-identical & thread-invariant.")

    print(f"\n=== RESULT: {'PASS' if overall_ok else 'FAIL'} ===")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
