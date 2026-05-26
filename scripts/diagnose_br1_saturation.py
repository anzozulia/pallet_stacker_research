"""Run BR1 instance #1 at varied budgets to see where saturation hits."""
from __future__ import annotations
import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.br import parse_thpack, geometric_only_config, first_pallet_utilization
from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit

set_name = "thpack1"
set_id = "BR1"
instances = parse_thpack(f"benchmarks/data/{set_name}.txt", set_id=set_id)
warmup_jit()
print(f"{'inst':5s} {'N':>4s} {'budget':>7s} {'util':>7s} {'rt':>6s}")
for inst_id in (1, 5, 8):  # easy / mid-load / heavy
    inst = instances[inst_id - 1]
    cfg = geometric_only_config(max_pallets=1)
    for budget in (30, 60, 120, 240):
        t0 = time.time()
        r = brkga_pack_v35(
            inst.boxes, inst.pallet, cfg,
            time_limit_s=budget, max_pallets=1,
            population_size=600, n_populations=3,
            patience=200, local_search_budget_s=4.0,
            seed=42 + inst.instance_id, verbose=False,
            n_modes=6,
        )
        rt = time.time() - t0
        util = first_pallet_utilization(r, inst.pallet)
        print(f"BR1#{inst_id:<2d} {len(inst.boxes):4d} {budget:>6d}s "
              f"{util*100:6.2f}% {rt:5.1f}s", flush=True)
