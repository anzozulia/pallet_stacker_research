"""Run brkga_pack_v35 on one BR instance with canonical config; print util."""
from __future__ import annotations
import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.br import parse_thpack, geometric_only_config, first_pallet_utilization
from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit

set_name = sys.argv[1] if len(sys.argv) > 1 else "thpack3"
inst_id = int(sys.argv[2]) if len(sys.argv) > 2 else 3
budget = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0

set_id = set_name.upper().replace("THPACK", "BR")
instances = parse_thpack(f"benchmarks/data/{set_name}.txt", set_id=set_id)
inst = instances[inst_id - 1]
cfg = geometric_only_config(max_pallets=1)
warmup_jit()
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
print(f"{set_id}#{inst.instance_id} N={len(inst.boxes)} budget={budget}s "
      f"util={util*100:.2f}% rt={rt:.1f}s")
