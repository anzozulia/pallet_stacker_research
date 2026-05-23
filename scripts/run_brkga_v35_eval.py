"""Run v3.5 (hybrid BRKGA) on BR instances."""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, json, statistics, time
from dataclasses import asdict, dataclass
from typing import Dict, List

from pallet_packer import validate
from pallet_packer.brkga_v3_5 import brkga_pack_v35
from benchmarks.br import parse_thpack, geometric_only_config, first_pallet_utilization


@dataclass
class V35Result:
    set_id: str
    instance_id: int
    n_boxes: int
    util_pallet1: float
    runtime_s: float
    validator_errors: int


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+",
                    default=["thpack1", "thpack3", "thpack5", "thpack7"])
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--pop", type=int, default=600)
    ap.add_argument("--n-pops", type=int, default=3)
    ap.add_argument("--patience", type=int, default=200)
    ap.add_argument("--n-restarts", type=int, default=1)
    ap.add_argument("--ls-budget", type=float, default=4.0)
    ap.add_argument("--max-instances", type=int, default=10)
    ap.add_argument("--skip-huge-n", type=int, default=250)
    ap.add_argument("--out", default="results/checkpoints/brkga_v35_results.json")
    args = ap.parse_args()

    cfg = geometric_only_config(max_pallets=1)
    print(f"v3.5: pop={args.pop}/{args.n_pops}pops, restarts={args.n_restarts}, "
          f"LS={args.ls_budget}s, time={args.time_limit}s/instance", file=sys.stderr)

    results: List[Dict] = []
    if os.path.exists(args.out):
        with open(args.out) as f:
            results = json.load(f)
    done = {(r["set_id"], r["instance_id"]) for r in results}

    for set_name in args.sets:
        path = f"benchmarks/data/{set_name}.txt"
        set_id = set_name.upper().replace("THPACK", "BR")
        instances = parse_thpack(path, set_id=set_id)
        if args.max_instances:
            instances = instances[:args.max_instances]
        print(f"\n=== {set_name} ({len(instances)} inst) ===", file=sys.stderr)
        t_set = time.time()
        for inst in instances:
            if (set_id, inst.instance_id) in done:
                continue
            if len(inst.boxes) > args.skip_huge_n:
                continue
            t0 = time.time()
            r = brkga_pack_v35(
                inst.boxes, inst.pallet, cfg,
                time_limit_s=args.time_limit, max_pallets=1,
                population_size=args.pop, n_populations=args.n_pops,
                patience=args.patience, local_search_budget_s=args.ls_budget,
                n_restarts=args.n_restarts,
                seed=42 + inst.instance_id, verbose=False,
            )
            rt = time.time() - t0
            errs = validate(r, inst.pallet, cfg)
            results.append(asdict(V35Result(
                set_id=set_id, instance_id=inst.instance_id,
                n_boxes=len(inst.boxes),
                util_pallet1=first_pallet_utilization(r, inst.pallet),
                runtime_s=rt, validator_errors=len(errs),
            )))
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(results, f, indent=1)
        set_r = [r for r in results if r["set_id"] == set_id]
        if set_r:
            u = [r["util_pallet1"] for r in set_r]
            print(f"  {set_id}: n={len(set_r)} mean={statistics.mean(u)*100:.1f}% "
                  f"stdev={statistics.stdev(u)*100 if len(u)>1 else 0:.1f} "
                  f"elapsed={time.time()-t_set:.0f}s", file=sys.stderr)

    print("\n=== v3.5 Summary ===", file=sys.stderr)
    by_set = {}
    for r in results:
        by_set.setdefault(r["set_id"], []).append(r["util_pallet1"])
    for sid, u in sorted(by_set.items()):
        print(f"{sid}: n={len(u):3d} mean={statistics.mean(u)*100:5.1f}% "
              f"stdev={statistics.stdev(u)*100 if len(u)>1 else 0:.1f}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
