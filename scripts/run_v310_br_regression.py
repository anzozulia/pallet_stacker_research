"""run_v310_br_regression.py — verify v3.10 hasn't degraded BR performance.

Re-runs BR1/3/5/7 n=10/set with the v3.10 code (constraint-aware path
inactive on BR because boxes have no weight/mlot constraints) using
identical params + seeds to the prior v3.8 full eval. Compares head-
to-head against the stored v3.8 results in v38_full_eval.json.
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, json, statistics, time
from dataclasses import asdict, dataclass
from typing import Dict, List

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
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--time", type=float, default=30.0)
    ap.add_argument("--out", default="results/checkpoints/v310_br_regression.json")
    ap.add_argument("--v38-baseline",
                    default="results/checkpoints/v38_full_eval.json")
    args = ap.parse_args()

    cfg = geometric_only_config(max_pallets=1)
    print(f"=== v3.10 BR regression  n={args.n}/set  time={args.time}s ===",
          flush=True)
    print("warming up JIT...", flush=True)
    warmup_jit()

    # Load v3.8 baseline from prior run
    v38: Dict[tuple, dict] = {}
    if os.path.exists(args.v38_baseline):
        for r in json.load(open(args.v38_baseline)):
            if r.get("config") == "v3.8" and "set_id" in r:
                v38[(r["set_id"], r["instance_id"])] = r

    results: List[dict] = []
    if os.path.exists(args.out):
        with open(args.out) as f:
            results = json.load(f)
    done = {(r["set_id"], r["instance_id"]) for r in results}

    for set_name in args.sets:
        set_id = set_name.upper().replace("THPACK", "BR")
        instances = parse_thpack(f"benchmarks/data/{set_name}.txt",
                                  set_id=set_id)[:args.n]
        for inst in instances:
            if (set_id, inst.instance_id) in done:
                continue
            t0 = time.time()
            r = brkga_pack_v35(
                inst.boxes, inst.pallet, cfg,
                time_limit_s=args.time, max_pallets=1,
                population_size=600, n_populations=3,
                patience=200, local_search_budget_s=4.0,
                seed=42 + inst.instance_id, verbose=False,
                n_modes=6,
            )
            rt = time.time() - t0
            errs = validate(r, inst.pallet, cfg)
            row = asdict(Row(
                set_id=set_id, instance_id=inst.instance_id,
                n_boxes=len(inst.boxes), config="v3.10",
                util_pallet1=first_pallet_utilization(r, inst.pallet),
                runtime_s=rt, validator_errors=len(errs),
            ))
            results.append(row)
            done.add((set_id, inst.instance_id))
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(results, f, indent=1)
            baseline = v38.get((set_id, inst.instance_id))
            delta = ""
            if baseline:
                d_pp = (row["util_pallet1"] - baseline["util_pallet1"]) * 100
                delta = f" (v3.8={baseline['util_pallet1']*100:.2f}%  Δ={d_pp:+.2f}pp)"
            print(f"  {set_id}#{inst.instance_id:>2} N={len(inst.boxes):>4d}  "
                  f"v3.10: util={row['util_pallet1']*100:>5.2f}%  "
                  f"{rt:>5.1f}s  errs={len(errs):>2d}{delta}", flush=True)

    # Summary
    print("\n=== Summary: v3.8 → v3.10 (n=10 each) ===", flush=True)
    print(f"{'set':<5}  {'v3.8 mean±sd':<18} {'v3.10 mean±sd':<18} "
          f"{'Δ mean':>8}  W/T/L")
    by_v38 = {}
    by_v310 = {}
    for k, r in v38.items():
        by_v38.setdefault(k[0], []).append(r["util_pallet1"])
    for r in results:
        by_v310.setdefault(r["set_id"], []).append(r["util_pallet1"])
    for sid in ["BR1", "BR3", "BR5", "BR7"]:
        a = by_v38.get(sid, [])
        b = by_v310.get(sid, [])
        if not a or not b:
            continue
        ma, sa = statistics.mean(a)*100, (statistics.stdev(a)*100 if len(a)>1 else 0)
        mb, sb = statistics.mean(b)*100, (statistics.stdev(b)*100 if len(b)>1 else 0)
        # Per-instance comparison
        per = {}
        for r in v38.values():
            if r["set_id"] == sid:
                per.setdefault(r["instance_id"], {})["v3.8"] = r["util_pallet1"]
        for r in results:
            if r["set_id"] == sid:
                per.setdefault(r["instance_id"], {})["v3.10"] = r["util_pallet1"]
        w = t = l = 0
        for iid, cfgs in per.items():
            if "v3.8" in cfgs and "v3.10" in cfgs:
                d_pp = (cfgs["v3.10"] - cfgs["v3.8"]) * 100
                if d_pp > 0.3: w += 1
                elif d_pp < -0.3: l += 1
                else: t += 1
        print(f"{sid:<5}  {ma:>5.2f}±{sa:>4.2f}     "
              f"{mb:>5.2f}±{sb:>4.2f}     {mb-ma:>+6.2f}pp  {w}/{t}/{l}")

    # Per-instance lift table (only items in both)
    print("\n=== Per-instance deltas ===", flush=True)
    print(f"{'set':<5} {'#':>3} {'N':>4} {'v3.8':>7} {'v3.10':>7} {'Δ':>7}")
    for r in results:
        b = v38.get((r["set_id"], r["instance_id"]))
        if not b:
            continue
        d_pp = (r["util_pallet1"] - b["util_pallet1"]) * 100
        marker = " ←" if abs(d_pp) > 1.0 else ""
        print(f"{r['set_id']:<5} {r['instance_id']:>3d} {r['n_boxes']:>4d} "
              f"{b['util_pallet1']*100:>6.2f}% {r['util_pallet1']*100:>6.2f}% "
              f"{d_pp:>+6.2f}pp{marker}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
