"""A/B test block-aware LS on the 6 hard-stuck BR1 instances.

The hard-stuck instances (BR1#3/4/6/7/9/10) don't respond to patience
tuning, LNS, multi-restart, or no-v2-seed. They're the remaining ~1pp
of the SOTA gap. Block-aware LS should help these by allowing same-SKU
clusters to be swapped or consolidated as a unit.
"""
from __future__ import annotations
import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.br import parse_thpack, geometric_only_config, first_pallet_utilization
from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit

instances_all = parse_thpack('benchmarks/data/thpack1.txt', set_id='BR1')[:10]
warmup_jit()
print(f"{'inst':6s} {'N':>4s} {'baseline':>9s}")
for inst in instances_all:
    cfg = geometric_only_config(max_pallets=1)
    common = dict(
        time_limit_s=30, max_pallets=1,
        population_size=600, n_populations=3,
        patience=200, local_search_budget_s=4.0,
        seed=42 + inst.instance_id, verbose=False, n_modes=6,
    )
    r = brkga_pack_v35(inst.boxes, inst.pallet, cfg, **common)
    u = first_pallet_utilization(r, inst.pallet) * 100
    print(f"BR1#{inst.instance_id:<3d} {len(inst.boxes):4d} {u:8.2f}%", flush=True)
