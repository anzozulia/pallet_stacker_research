"""BR n=10 at 30s budget without v2 seed."""
from __future__ import annotations
import sys, time, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.br import parse_thpack, geometric_only_config, first_pallet_utilization
from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit

results = []
warmup_jit()
for set_name in ("thpack1", "thpack3", "thpack5", "thpack7"):
    set_id = set_name.upper().replace("THPACK", "BR")
    for inst in parse_thpack(f"benchmarks/data/{set_name}.txt", set_id=set_id)[:10]:
        cfg = geometric_only_config(max_pallets=1)
        t0 = time.time()
        r = brkga_pack_v35(
            inst.boxes, inst.pallet, cfg,
            time_limit_s=30, max_pallets=1,
            population_size=600, n_populations=3,
            patience=200, local_search_budget_s=4.0,
            seed=42 + inst.instance_id, verbose=False, n_modes=6,
            use_v2_seed=False,
        )
        rt = time.time() - t0
        u = first_pallet_utilization(r, inst.pallet)
        results.append({"set_id": set_id, "instance_id": inst.instance_id,
                        "n_boxes": len(inst.boxes),
                        "util_pallet1": u, "runtime_s": rt})
        print(f"{set_id}#{inst.instance_id:<3d} N={len(inst.boxes):4d} "
              f"util={u*100:6.2f}% rt={rt:.1f}s", flush=True)

with open("results/checkpoints/br_n10_no_v2.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {len(results)} rows.")
