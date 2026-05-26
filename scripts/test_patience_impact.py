"""Test patience parameter impact. If patience=200 is causing early
termination at the saturation point, raising it should use the full
budget and (maybe) find escapes."""
from __future__ import annotations
import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.br import parse_thpack, geometric_only_config, first_pallet_utilization
from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit

instances = parse_thpack('benchmarks/data/thpack1.txt', set_id='BR1')[:5]
warmup_jit()
print(f"{'inst':6s} {'N':>4s} {'p=200':>9s} {'p=2000':>10s} {'p=2000+LNS':>12s} {'rt p=200':>9s} {'rt p=2000':>10s}")
for inst in instances:
    cfg = geometric_only_config(max_pallets=1)
    common = dict(
        time_limit_s=30, max_pallets=1,
        population_size=600, n_populations=3,
        local_search_budget_s=4.0,
        seed=42 + inst.instance_id, verbose=False, n_modes=6,
    )
    # Baseline: patience=200, LS only.
    t0 = time.time()
    r0 = brkga_pack_v35(inst.boxes, inst.pallet, cfg,
                        patience=200, use_lns=False, **common)
    rt0 = time.time() - t0
    u0 = first_pallet_utilization(r0, inst.pallet) * 100
    # Variant A: patience=2000.
    t0 = time.time()
    r1 = brkga_pack_v35(inst.boxes, inst.pallet, cfg,
                        patience=2000, use_lns=False, **common)
    rt1 = time.time() - t0
    u1 = first_pallet_utilization(r1, inst.pallet) * 100
    # Variant B: patience=2000 + LNS.
    r2 = brkga_pack_v35(inst.boxes, inst.pallet, cfg,
                        patience=2000, use_lns=True, lns_budget_s=2.0, **common)
    u2 = first_pallet_utilization(r2, inst.pallet) * 100
    print(f"BR1#{inst.instance_id:<3d} {len(inst.boxes):4d} {u0:8.2f}% {u1:9.2f}% {u2:11.2f}% "
          f"{rt0:7.1f}s {rt1:8.1f}s", flush=True)
