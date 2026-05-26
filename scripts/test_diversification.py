"""Test diversification strategies on BR1: multi-restart, no v2 seed."""
from __future__ import annotations
import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.br import parse_thpack, geometric_only_config, first_pallet_utilization
from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit

instances = parse_thpack('benchmarks/data/thpack1.txt', set_id='BR1')[:5]
warmup_jit()
print(f"{'inst':6s} {'N':>4s} {'base':>8s} {'r=2':>8s} {'r=3':>8s} {'no-v2':>8s} "
      f"{'r2+LNS':>9s} {'best Δ':>8s}")
for inst in instances:
    cfg = geometric_only_config(max_pallets=1)
    common = dict(
        time_limit_s=30, max_pallets=1,
        population_size=600, n_populations=3,
        patience=200, local_search_budget_s=4.0,
        seed=42 + inst.instance_id, verbose=False, n_modes=6,
    )
    # baseline
    r0 = brkga_pack_v35(inst.boxes, inst.pallet, cfg, n_restarts=1,
                        use_v2_seed=True, use_lns=False, **common)
    u0 = first_pallet_utilization(r0, inst.pallet) * 100
    # n_restarts=2 (15s each, fresh seeds)
    r1 = brkga_pack_v35(inst.boxes, inst.pallet, cfg, n_restarts=2,
                        use_v2_seed=True, use_lns=False, **common)
    u1 = first_pallet_utilization(r1, inst.pallet) * 100
    # n_restarts=3 (10s each)
    r2 = brkga_pack_v35(inst.boxes, inst.pallet, cfg, n_restarts=3,
                        use_v2_seed=True, use_lns=False, **common)
    u2 = first_pallet_utilization(r2, inst.pallet) * 100
    # no v2 seed (full 30s)
    r3 = brkga_pack_v35(inst.boxes, inst.pallet, cfg, n_restarts=1,
                        use_v2_seed=False, use_lns=False, **common)
    u3 = first_pallet_utilization(r3, inst.pallet) * 100
    # n_restarts=2 + LNS
    r4 = brkga_pack_v35(inst.boxes, inst.pallet, cfg, n_restarts=2,
                        use_v2_seed=True, use_lns=True, lns_budget_s=2.0,
                        **common)
    u4 = first_pallet_utilization(r4, inst.pallet) * 100
    best = max(u1, u2, u3, u4)
    print(f"BR1#{inst.instance_id:<3d} {len(inst.boxes):4d} "
          f"{u0:7.2f}% {u1:7.2f}% {u2:7.2f}% {u3:7.2f}% {u4:8.2f}% "
          f"{best-u0:+7.2f}pp", flush=True)
