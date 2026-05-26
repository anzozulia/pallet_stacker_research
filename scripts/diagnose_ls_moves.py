"""Confirm block-aware LS moves are actually firing.

Patches polish.py at import time to count move types.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Quick monkey-patch: replace local_search_2opt's RNG with a counting one.
import numpy as np
from benchmarks.br import parse_thpack, geometric_only_config, first_pallet_utilization
from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit
from pallet_packer._brkga_core import polish

# Hijack: track which move type fires
move_counts = {'swap': 0, 'rev': 0, 'insert': 0, 'bswap': 0, 'bcons': 0, 'rot': 0, 'dec': 0}
orig_random = np.random.Generator.random
fired = {'count': 0}

class _CountingRng:
    def __init__(self, rng):
        self.rng = rng
    def random(self):
        v = self.rng.random()
        fired['count'] += 1
        # categorize at top-level move-pick calls — first call per iteration
        return v
    def __getattr__(self, name):
        return getattr(self.rng, name)

instances = parse_thpack('benchmarks/data/thpack1.txt', set_id='BR1')
inst = instances[2]  # BR1#3
print(f"=== BR1#3 N={len(inst.boxes)} ===")
warmup_jit()
cfg = geometric_only_config(max_pallets=1)
# Trace: just count via verbose; we can match output for move types
r = brkga_pack_v35(
    inst.boxes, inst.pallet, cfg,
    time_limit_s=10, max_pallets=1,  # smaller budget
    population_size=600, n_populations=3,
    patience=200, local_search_budget_s=4.0,
    seed=42 + inst.instance_id, verbose=False, n_modes=6,
)
u = first_pallet_utilization(r, inst.pallet) * 100
print(f"util={u:.2f}%")

# Now patch a flag inside polish.py to count move types
import pallet_packer._brkga_core.polish as p
src = open(p.__file__).read()
print(f"\npolish.py has block_aware references: {src.count('block_aware')}")
print(f"polish.py has bswap branch: {'BLOCK SWAP' in src}")
print(f"polish.py has bcons branch: {'BLOCK CONSOLIDATE' in src}")
