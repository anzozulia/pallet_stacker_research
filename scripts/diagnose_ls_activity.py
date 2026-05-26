"""Diagnose what LS is doing on a hard-stuck instance.

Run BR1#3 with verbose LS to see how many moves fire, how many accept,
and whether block-aware moves are being tried.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.br import parse_thpack, geometric_only_config, first_pallet_utilization
from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit

instances = parse_thpack('benchmarks/data/thpack1.txt', set_id='BR1')
inst = instances[2]  # BR1#3
print(f"=== BR1#3 N={len(inst.boxes)} ===")
warmup_jit()
cfg = geometric_only_config(max_pallets=1)
r = brkga_pack_v35(
    inst.boxes, inst.pallet, cfg,
    time_limit_s=30, max_pallets=1,
    population_size=600, n_populations=3,
    patience=200, local_search_budget_s=4.0,
    seed=42 + inst.instance_id, verbose=True, n_modes=6,
)
u = first_pallet_utilization(r, inst.pallet) * 100
print(f"\nFinal util: {u:.2f}%")
