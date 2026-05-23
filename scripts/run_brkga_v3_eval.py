"""
run_brkga_v3_eval.py — BRKGA-2013 v3 evaluation on BR1-7.

Runs the new pallet_packer.brkga_v3 implementation across BR instances
with configurable compute budget. Output is per-instance util + runtime
plus a per-set summary.

Usage:
    python3 scripts/run_brkga_v3_eval.py \\
        --sets thpack1 thpack3 thpack5 thpack7 \\
        --time-limit 30 --pop 80 --gens 50 \\
        --skip-huge-n 200
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Dict, List

from pallet_packer import validate
from pallet_packer.brkga_v3 import brkga_pack
from benchmarks.br import parse_thpack, geometric_only_config, first_pallet_utilization


@dataclass
class V3Result:
    set_id: str
    instance_id: int
    n_boxes: int
    util_pallet1: float
    util_total: float
    runtime_s: float
    validator_errors: int


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+",
                    default=["thpack1", "thpack3", "thpack5", "thpack7"])
    ap.add_argument("--time-limit", type=float, default=30.0,
                    help="Per-instance time budget (seconds)")
    ap.add_argument("--pop", type=int, default=80, help="BRKGA population")
    ap.add_argument("--gens", type=int, default=50, help="Max generations")
    ap.add_argument("--elite-frac", type=float, default=0.25)
    ap.add_argument("--mutant-frac", type=float, default=0.20)
    ap.add_argument("--p-elite", type=float, default=0.70)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--max-instances", type=int, default=None)
    ap.add_argument("--skip-huge-n", type=int, default=200)
    ap.add_argument("--out", default="results/checkpoints/brkga_v3_results.json")
    args = ap.parse_args()

    cfg = geometric_only_config(max_pallets=1)
    print(f"BRKGA-v3 eval: pop={args.pop}, gens={args.gens}, "
          f"time={args.time_limit}s/instance", file=sys.stderr)

    all_results: List[Dict] = []
    if os.path.exists(args.out):
        with open(args.out) as f:
            all_results = json.load(f)
    done = {(r["set_id"], r["instance_id"]) for r in all_results}

    for set_name in args.sets:
        path = f"benchmarks/data/{set_name}.txt"
        set_id = set_name.upper().replace("THPACK", "BR")
        instances = parse_thpack(path, set_id=set_id)
        if args.max_instances:
            instances = instances[:args.max_instances]
        print(f"\n=== {set_name}: {len(instances)} instances ===", file=sys.stderr)
        t_set = time.time()
        n_done = 0
        for inst in instances:
            if (set_id, inst.instance_id) in done:
                continue
            if len(inst.boxes) > args.skip_huge_n:
                print(f"  SKIP {set_id}#{inst.instance_id} N={len(inst.boxes)}",
                      file=sys.stderr)
                continue
            t0 = time.time()
            result = brkga_pack(
                inst.boxes, inst.pallet, cfg,
                population_size=args.pop,
                generations=args.gens,
                elite_fraction=args.elite_frac,
                mutant_fraction=args.mutant_frac,
                p_elite_inherit=args.p_elite,
                time_limit_s=args.time_limit,
                max_pallets=1,
                seed=42 + inst.instance_id,
                patience=args.patience,
                try_all_rotations=True,
                verbose=False,
            )
            rt = time.time() - t0
            errs = validate(result, inst.pallet, cfg)
            res = V3Result(
                set_id=set_id,
                instance_id=inst.instance_id,
                n_boxes=len(inst.boxes),
                util_pallet1=first_pallet_utilization(result, inst.pallet),
                util_total=result.total_volume_utilisation,
                runtime_s=rt,
                validator_errors=len(errs),
            )
            all_results.append(asdict(res))
            n_done += 1
            if n_done % 5 == 0:
                os.makedirs(os.path.dirname(args.out), exist_ok=True)
                with open(args.out, "w") as f:
                    json.dump(all_results, f, indent=1)
                set_results = [r for r in all_results if r["set_id"] == set_id]
                utils = [r["util_pallet1"] for r in set_results]
                print(f"  [{n_done}/{len(instances)}] {set_id} mean={statistics.mean(utils)*100:.1f}% "
                      f"elapsed={time.time()-t_set:.0f}s", file=sys.stderr)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(all_results, f, indent=1)
        # Per-set summary
        set_results = [r for r in all_results if r["set_id"] == set_id]
        if set_results:
            utils = [r["util_pallet1"] for r in set_results]
            rts = [r["runtime_s"] for r in set_results]
            print(f"  {set_id}: n={len(set_results)} "
                  f"mean={statistics.mean(utils)*100:.1f}% "
                  f"stdev={statistics.stdev(utils)*100 if len(utils)>1 else 0:.1f} "
                  f"rt_avg={statistics.mean(rts):.1f}s", file=sys.stderr)

    # Final summary
    print("\n=== BRKGA-v3 Summary ===", file=sys.stderr)
    by_set: Dict[str, List[float]] = {}
    for r in all_results:
        by_set.setdefault(r["set_id"], []).append(r["util_pallet1"])
    for set_id, utils in sorted(by_set.items()):
        print(f"{set_id}: n={len(utils):3d} mean={statistics.mean(utils)*100:5.1f}% "
              f"stdev={statistics.stdev(utils)*100 if len(utils)>1 else 0:.1f}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
