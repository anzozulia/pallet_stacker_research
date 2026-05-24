"""run_v39_adaptive_eval.py — Compare v3.9 (adaptive selector) vs v3.8 baseline.

Runs BR1/3/5/7 with n=5 instances each, 30s budget, comparing
  v3.8: use_adaptive_mode_selector=False, n_modes=6
  v3.9: use_adaptive_mode_selector=True,  n_modes=6
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, json, statistics, time
from dataclasses import asdict, dataclass

from pallet_packer import validate
from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit
from benchmarks.br import parse_thpack, geometric_only_config, first_pallet_utilization


@dataclass
class Row:
    set_id: str
    instance_id: int
    n_boxes: int
    config: str
    util_pallet1: float
    runtime_s: float
    validator_errors: int


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+",
                    default=["thpack1", "thpack3", "thpack5", "thpack7"])
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--time", type=float, default=30.0)
    ap.add_argument("--out", default="results/checkpoints/v39_adaptive_eval.json")
    args = ap.parse_args()

    cfg = geometric_only_config(max_pallets=1)
    CONFIGS = [
        ("v3.8", dict(use_adaptive_mode_selector=False, n_modes=6)),
        ("v3.9", dict(use_adaptive_mode_selector=True,  n_modes=6)),
    ]

    print(f"=== BR n={args.n}/set, time={args.time}s ===", flush=True)
    print("warming up JIT...", flush=True)
    warmup_jit()

    results = []
    if os.path.exists(args.out):
        with open(args.out) as f:
            results = json.load(f)
    done = {(r["set_id"], r["instance_id"], r["config"]) for r in results}

    for set_name in args.sets:
        set_id = set_name.upper().replace("THPACK", "BR")
        path = f"benchmarks/data/{set_name}.txt"
        instances = parse_thpack(path, set_id=set_id)[:args.n]
        for inst in instances:
            for cname, kw in CONFIGS:
                key = (set_id, inst.instance_id, cname)
                if key in done:
                    continue
                t0 = time.time()
                r = brkga_pack_v35(
                    inst.boxes, inst.pallet, cfg,
                    time_limit_s=args.time, max_pallets=1,
                    population_size=600, n_populations=3,
                    patience=200, local_search_budget_s=4.0,
                    seed=42 + inst.instance_id, verbose=False, **kw,
                )
                rt = time.time() - t0
                errs = validate(r, inst.pallet, cfg)
                row = asdict(Row(
                    set_id=set_id, instance_id=inst.instance_id,
                    n_boxes=len(inst.boxes), config=cname,
                    util_pallet1=first_pallet_utilization(r, inst.pallet),
                    runtime_s=rt, validator_errors=len(errs),
                ))
                results.append(row)
                done.add(key)
                os.makedirs(os.path.dirname(args.out), exist_ok=True)
                with open(args.out, "w") as f:
                    json.dump(results, f, indent=1)
                print(f"  {set_id}#{inst.instance_id:>2}  {cname}: "
                      f"util={row['util_pallet1']*100:.2f}%  {rt:.1f}s",
                      flush=True)

    # Summary
    print("\n=== Summary ===", flush=True)
    by = {}
    for r in results:
        by.setdefault((r["set_id"], r["config"]), []).append(r["util_pallet1"])
    print(f"{'set':<5}  {'v3.8 (mean±sd)':<18} {'v3.9 (mean±sd)':<18} {'Δ':>7}  W/T/L")
    for sid in ['BR1', 'BR3', 'BR5', 'BR7']:
        v38 = by.get((sid, 'v3.8'), [])
        v39 = by.get((sid, 'v3.9'), [])
        if not v38 or not v39:
            continue
        m38, m39 = statistics.mean(v38)*100, statistics.mean(v39)*100
        s38 = statistics.stdev(v38)*100 if len(v38)>1 else 0
        s39 = statistics.stdev(v39)*100 if len(v39)>1 else 0
        # W/T/L
        per_inst = {}
        for r in results:
            if r["set_id"] != sid:
                continue
            per_inst.setdefault(r["instance_id"], {})[r["config"]] = r["util_pallet1"]
        w = t = l = 0
        for iid, cfgs in per_inst.items():
            if 'v3.8' in cfgs and 'v3.9' in cfgs:
                d_pp = (cfgs['v3.9'] - cfgs['v3.8']) * 100
                if d_pp > 0.3: w += 1
                elif d_pp < -0.3: l += 1
                else: t += 1
        print(f"{sid:<5}  {m38:.2f}±{s38:.2f}     {m39:.2f}±{s39:.2f}     "
              f"{m39-m38:+5.2f}pp  {w}/{t}/{l}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
