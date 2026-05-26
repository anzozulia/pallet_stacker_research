"""Test impact of enabling LNS on BR1 instances. Quick A/B."""
from __future__ import annotations
import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.br import parse_thpack, geometric_only_config, first_pallet_utilization
from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit

instances = parse_thpack('benchmarks/data/thpack1.txt', set_id='BR1')[:5]
warmup_jit()
print(f"{'inst':6s} {'N':>4s} {'LS only':>9s} {'LS+LNS':>9s} {'Δ':>6s}")
for inst in instances:
    cfg = geometric_only_config(max_pallets=1)
    common = dict(
        time_limit_s=30, max_pallets=1,
        population_size=600, n_populations=3,
        patience=200, local_search_budget_s=4.0,
        seed=42 + inst.instance_id, verbose=False, n_modes=6,
    )
    # LS only (current default).
    r0 = brkga_pack_v35(inst.boxes, inst.pallet, cfg, use_lns=False, **common)
    u0 = first_pallet_utilization(r0, inst.pallet) * 100
    # LS + LNS (use_lns=True, lns_budget_s=2 carved from BRKGA budget).
    r1 = brkga_pack_v35(inst.boxes, inst.pallet, cfg, use_lns=True,
                        lns_budget_s=2.0, **common)
    u1 = first_pallet_utilization(r1, inst.pallet) * 100
    print(f"BR1#{inst.instance_id:<3d} {len(inst.boxes):4d} {u0:8.2f}% {u1:8.2f}% "
          f"{u1-u0:+5.2f}pp", flush=True)
