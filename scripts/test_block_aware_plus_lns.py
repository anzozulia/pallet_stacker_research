"""Last-ditch test: block-aware LS + LNS combined on hard BR1 instances."""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.br import parse_thpack, geometric_only_config, first_pallet_utilization
from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit

instances = parse_thpack('benchmarks/data/thpack1.txt', set_id='BR1')[:10]
warmup_jit()
print(f"{'inst':6s} {'N':>4s} {'baseline':>9s} {'+LNS':>8s} {'+LNS,LS=8s':>11s}")
for inst in instances:
    cfg = geometric_only_config(max_pallets=1)
    common = dict(
        time_limit_s=30, max_pallets=1,
        population_size=600, n_populations=3,
        patience=200,
        seed=42 + inst.instance_id, verbose=False, n_modes=6,
    )
    # baseline (current default)
    r0 = brkga_pack_v35(inst.boxes, inst.pallet, cfg,
                        local_search_budget_s=4.0, use_lns=False, **common)
    u0 = first_pallet_utilization(r0, inst.pallet) * 100
    # +LNS
    r1 = brkga_pack_v35(inst.boxes, inst.pallet, cfg,
                        local_search_budget_s=4.0, use_lns=True,
                        lns_budget_s=2.0, **common)
    u1 = first_pallet_utilization(r1, inst.pallet) * 100
    # +LNS, LS=8s
    r2 = brkga_pack_v35(inst.boxes, inst.pallet, cfg,
                        local_search_budget_s=8.0, use_lns=True,
                        lns_budget_s=3.0, **common)
    u2 = first_pallet_utilization(r2, inst.pallet) * 100
    print(f"BR1#{inst.instance_id:<3d} {len(inst.boxes):4d} {u0:8.2f}% {u1:7.2f}% {u2:10.2f}%", flush=True)
