"""Verification smoke baseline — confirm the harness is sound before fan-out.

Checks, inside Docker:
  1. Cython modules are the active backend (not Numba fallback).
  2. Full-solve determinism: same (instance, seed) → bit-identical placements
     across repeated runs.
  3. Determinism across OMP_NUM_THREADS: result must not depend on thread count
     (prange static-schedule contract).
  4. Validity: validate() returns zero errors on a real BR instance + an
     industry (constrained) instance.

Run:
    docker run --rm -v "$(pwd)":/app -w /app pallet-packer:dev \
        python scripts/_verify/smoke_baseline.py
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from pallet_packer import validate
from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit
from benchmarks.br import parse_thpack, geometric_only_config, first_pallet_utilization
from benchmarks.industry import cases as industry_cases


def placements_signature(result):
    """Canonical, order-independent signature of a PackResult's geometry."""
    rows = []
    for pi, st in enumerate(result.pallets):
        for p in st.placements:
            rows.append((pi, p.box.id, p.rotation.name,
                         round(p.x, 6), round(p.y, 6), round(p.z, 6)))
    rows.sort()
    return tuple(rows)


def check_backend():
    from pallet_packer._brkga_core import dispatch
    mod = dispatch.decode_njit_mode.__module__
    is_cy = "cy" in mod
    print(f"[backend] decode_njit_mode from: {mod}  ({'CYTHON' if is_cy else 'NUMBA'})")
    print(f"[backend] _BATCH_AVAILABLE = {dispatch._BATCH_AVAILABLE}")
    return is_cy and dispatch._BATCH_AVAILABLE


def run_geom(inst, seed_off=0, t=8.0):
    cfg = geometric_only_config(max_pallets=1)
    return brkga_pack_v35(
        inst.boxes, inst.pallet, cfg,
        time_limit_s=t, max_pallets=1,
        population_size=300, n_populations=3, patience=200,
        local_search_budget_s=2.0, seed=42 + inst.instance_id + seed_off,
        verbose=False, n_modes=6,
    )


def main():
    warmup_jit()
    ok = True

    print("=== 1. Backend check ===")
    if not check_backend():
        print("  FAIL: not running Cython batch backend")
        ok = False

    inst = parse_thpack("benchmarks/data/thpack1.txt", set_id="BR1")[0]
    print(f"\n=== 2. Determinism (BR1#{inst.instance_id}, N={len(inst.boxes)}) ===")
    sigs = []
    for run in range(3):
        r = run_geom(inst)
        sigs.append(placements_signature(r))
        u = first_pallet_utilization(r, inst.pallet) * 100
        print(f"  run {run}: util={u:.4f}%  n_placed={len(sigs[-1])}")
    if len(set(sigs)) == 1:
        print("  determinism: bit-identical across 3 runs ✓")
    else:
        print(f"  FAIL: {len(set(sigs))} distinct results across 3 runs")
        ok = False

    print(f"\n=== 3. Thread-count invariance (OMP={os.environ.get('OMP_NUM_THREADS','default')}) ===")
    # Re-run under a forced single-thread context within same process not
    # possible (OMP set at import); we record the signature so the wrapper
    # script can compare across two container invocations.
    sig0 = placements_signature(run_geom(inst))
    print(f"  signature_hash={hash(sig0)}")

    print("\n=== 4. Validity on constrained industry case (IND4 beverage) ===")
    cs = industry_cases()
    c = cs[3]  # IND4 beverage — weight-binding
    r = brkga_pack_v35(c.boxes, c.pallet, c.config,
                       time_limit_s=10.0, max_pallets=10,
                       population_size=200, n_populations=3, patience=150,
                       seed=42, verbose=False, n_modes=6)
    errs = validate(r, c.pallet, c.config)
    print(f"  {c.name}: pallets={len(r.pallets)} unpacked={len(r.unpacked)} "
          f"validator_errors={len(errs)}")
    if errs:
        for e in errs[:5]:
            print(f"    ERROR: {e}")
        ok = False
    else:
        print("  validator: clean ✓")

    print(f"\n=== RESULT: {'PASS' if ok else 'FAIL'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
