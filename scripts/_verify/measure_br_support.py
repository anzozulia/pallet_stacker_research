"""Honest BR utilization vs support_ratio.

Re-measures BR1/3/5/7 with support enforcement at several support_ratio levels,
so we can see what enforcing physical stability (which the BR literature is said
to assume) costs in utilization — and whether the prior "beats 2013 SOTA" claim
(measured with support OFF) holds once support is ON.

Protocol — CONVERGED BRKGA (contention-immune, so it can run in parallel without
the wall-clock/CPU contention biasing the result):
  - patience-bounded convergence (not a wall-clock cutoff)
  - local search / LNS / v2-hybrid / v2-seed OFF -> deterministic, thread-
    invariant result (verified: converged runs are bit-identical across threads)
  - population_size=600, n_populations=3, n_modes=6, seed=42+instance_id
This differs from the prior 30s-time-bounded protocol; the SUPPORT COST (same
protocol, sr=0 vs sr=1.0) is the robust headline. Absolute sr=0 is shown next to
the prior reference number to expose the protocol delta.

Usage:
    python scripts/_verify/measure_br_support.py --set thpack1 --srs 0.0 0.8 1.0 --n 10
Writes results/checkpoints/br_support_<set>.json and prints a JSON summary line.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pallet_packer import validate
from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit
from benchmarks.br import parse_thpack, geometric_only_config, first_pallet_utilization


def run_instance(inst, sr, patience, time_cap):
    cfg = replace(geometric_only_config(max_pallets=1),
                  support_ratio=sr, require_centroid_supported=False)
    t0 = time.perf_counter()
    r = brkga_pack_v35(
        inst.boxes, inst.pallet, cfg,
        time_limit_s=time_cap, max_pallets=1,
        population_size=600, n_populations=3, patience=patience,
        n_modes=6, seed=42 + inst.instance_id,
        use_local_search=False, use_lns=False,
        use_v2_hybrid_polish=False, use_v2_seed=False,
        verbose=False,
    )
    rt = time.perf_counter() - t0
    util = first_pallet_utilization(r, inst.pallet) * 100
    errs = len(validate(r, inst.pallet, cfg))
    return util, errs, rt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)           # thpack1/3/5/7
    ap.add_argument("--srs", nargs="+", type=float, default=[0.0, 0.8, 1.0])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--time", type=float, default=180.0)
    args = ap.parse_args()

    set_id = args.set.upper().replace("THPACK", "BR")
    instances = parse_thpack(f"benchmarks/data/{args.set}.txt", set_id=set_id)[:args.n]
    warmup_jit()

    out = {"set": set_id, "n": len(instances), "patience": args.patience,
           "protocol": "converged_LSoff", "results": []}
    for sr in args.srs:
        rows = []
        for inst in instances:
            util, errs, rt = run_instance(inst, sr, args.patience, args.time)
            rows.append({"id": inst.instance_id, "util": round(util, 4),
                         "errs": errs, "rt": round(rt, 1)})
            print(f"  {set_id} sr={sr:<4} #{inst.instance_id:>2} N={len(inst.boxes):>4} "
                  f"util={util:6.2f}% errs={errs} {rt:5.1f}s", file=sys.stderr, flush=True)
        utils = [r["util"] for r in rows]
        out["results"].append({
            "sr": sr,
            "mean_util": round(statistics.mean(utils), 3),
            "stdev": round(statistics.stdev(utils), 3) if len(utils) > 1 else 0.0,
            "clean": sum(1 for r in rows if r["errs"] == 0),
            "instances": rows,
        })
        print(f"=== {set_id} sr={sr}: mean_util={statistics.mean(utils):.2f}% "
              f"clean={out['results'][-1]['clean']}/{len(rows)} ===", file=sys.stderr, flush=True)

    os.makedirs("results/checkpoints", exist_ok=True)
    path = f"results/checkpoints/br_support_{args.set}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out))   # stdout = single JSON line for the caller
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
