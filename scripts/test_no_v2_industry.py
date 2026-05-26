"""Test impact of use_v2_seed=False on industry workloads (constrained)."""
from __future__ import annotations
import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.industry import cases as industry_cases
from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit
from pallet_packer import validate

warmup_jit()
all_cases = industry_cases()
print(f"{'case':46s} {'v2':>22s} {'no-v2':>22s}")
for c in all_cases:
    boxes, pallet, config = c.boxes, c.pallet, c.config
    s_v2 = s_nv = ""
    for use_v2 in (True, False):
        t0 = time.time()
        r = brkga_pack_v35(
            boxes, pallet, config,
            time_limit_s=15, max_pallets=4,
            population_size=400, n_populations=3,
            patience=200, local_search_budget_s=2.0,
            seed=42, verbose=False, n_modes=6,
            use_v2_seed=use_v2,
        )
        rt = time.time() - t0
        errs = validate(r, pallet, config)
        cap = pallet.length * pallet.width * pallet.height
        u1 = (sum(p.box.volume for p in r.pallets[0].placements) / cap * 100
              if r.pallets else 0.0)
        n_unp = len(r.unpacked) if r.unpacked else 0
        s = f"{len(r.pallets)}p {n_unp}u u1={u1:5.1f}% e={len(errs)}"
        if use_v2:
            s_v2 = s
        else:
            s_nv = s
    print(f"{c.name[:46]:46s} {s_v2:>22s} {s_nv:>22s}", flush=True)
