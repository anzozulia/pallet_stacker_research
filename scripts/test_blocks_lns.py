"""Test blocks (mode 4) and LNS impact across BR sets.

3 configs × 4 sets × 5 instances × 30s = ~30 min total.
"""
import sys, time, statistics
sys.path.insert(0, '/Users/anzozulia/Desktop/pallet_stacker_research')

from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit
from benchmarks.br import parse_thpack, geometric_only_config, first_pallet_utilization

warmup_jit()
cfg = geometric_only_config(max_pallets=1)

CONFIGS = [
    ('baseline_4modes',  dict(n_modes=4, use_lns=False)),
    ('blocks_5modes',    dict(n_modes=5, use_lns=False)),
    ('blocks+lns',       dict(n_modes=5, use_lns=True, lns_budget_s=3.0,
                              local_search_budget_s=2.0)),
]

results = {}  # (set_id, config_name) -> list of utils
print(f"BR sets test: 5 inst/set, 30s/inst, 3 configs")
print(f"{'config':<20} {'set':<5} {'inst':>4} {'util%':>7} {'time':>6}", flush=True)
print("-" * 50)

for set_name, set_id in [('thpack1', 'BR1'), ('thpack3', 'BR3'),
                          ('thpack5', 'BR5'), ('thpack7', 'BR7')]:
    instances = parse_thpack(f'benchmarks/data/{set_name}.txt', set_id=set_id)[:5]
    for cname, kw in CONFIGS:
        utils = []
        for inst in instances:
            t0 = time.time()
            res = brkga_pack_v35(
                inst.boxes, inst.pallet, cfg,
                time_limit_s=30.0, max_pallets=1,
                population_size=600, n_populations=3,
                patience=200, seed=42 + inst.instance_id,
                verbose=False, **kw,
            )
            elapsed = time.time() - t0
            u = first_pallet_utilization(res, inst.pallet) * 100
            utils.append(u)
            print(f"{cname:<20} {set_id:<5} {inst.instance_id:>4} {u:>6.2f} {elapsed:>5.1f}s", flush=True)
        results[(set_id, cname)] = utils
        m = statistics.mean(utils)
        print(f"  → {cname}/{set_id} mean={m:.2f}% stdev={statistics.stdev(utils) if len(utils)>1 else 0:.2f}", flush=True)

print("\n=== Summary ===", flush=True)
print(f"{'set':<5}", end='')
for cname, _ in CONFIGS:
    print(f" {cname:>17}", end='')
print(flush=True)
for sid in ['BR1', 'BR3', 'BR5', 'BR7']:
    print(f"{sid:<5}", end='')
    for cname, _ in CONFIGS:
        m = statistics.mean(results[(sid, cname)])
        print(f" {m:>17.2f}%", end='')
    print(flush=True)

print("\n=== Deltas vs baseline ===", flush=True)
for sid in ['BR1', 'BR3', 'BR5', 'BR7']:
    base = statistics.mean(results[(sid, 'baseline_4modes')])
    for cname, _ in CONFIGS[1:]:
        d = statistics.mean(results[(sid, cname)]) - base
        print(f"  {sid} {cname}: {d:+.2f}pp", flush=True)
