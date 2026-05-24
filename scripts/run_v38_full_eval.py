"""
run_v38_full_eval.py — Comprehensive v3.8 evaluation.

Runs:
  1. BR1/3/5/7 with n=10 instances each, comparing
       baseline = v3.6 (n_modes=5, top-K disabled in selector range)
       v3.8     = top-K (n_modes=6, mode 5 = pre-computed top-K blocks)
  2. All 10 industry scenarios with v3.8 (multi-pallet).

Persists everything to results/checkpoints/v38_full_eval.json incrementally
so a Ctrl-C / OOM kill doesn't lose progress.
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
from typing import Dict, List, Optional

from pallet_packer import validate
from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit
from benchmarks.br import parse_thpack, geometric_only_config, first_pallet_utilization
from benchmarks.industry import cases as industry_cases


@dataclass
class BRRow:
    set_id: str
    instance_id: int
    n_boxes: int
    config: str
    util_pallet1: float
    runtime_s: float
    validator_errors: int


@dataclass
class INDRow:
    case_name: str
    n_boxes: int
    config: str
    pallets_used: int
    unpacked: int
    util_pallet1: float
    util_total: float
    runtime_s: float
    validator_errors: int


def first_pallet_util(result, pallet):
    if not result.pallets:
        return 0.0
    st = result.pallets[0]
    used = sum(p.box.volume for p in st.placements)
    cap = pallet.length * pallet.width * pallet.height
    return used / cap if cap > 0 else 0.0


def total_util(result, pallet):
    if not result.pallets:
        return 0.0
    used = sum(p.box.volume for st in result.pallets for p in st.placements)
    cap = pallet.length * pallet.width * pallet.height * len(result.pallets)
    return used / cap if cap > 0 else 0.0


def run_br(args, results: List[Dict], outpath: str) -> None:
    cfg = geometric_only_config(max_pallets=1)
    done = {(r["set_id"], r["instance_id"], r["config"])
            for r in results if "set_id" in r}

    CONFIGS = [
        ("v3.6", dict(n_modes=5)),       # selector picks 0..4 (no top-K)
        ("v3.8", dict(n_modes=6)),       # selector picks 0..5 (top-K active)
    ]

    print(f"\n=== BR n={args.br_n} per set, time={args.br_time}s ===", flush=True)
    for set_name in args.sets:
        set_id = set_name.upper().replace("THPACK", "BR")
        path = f"benchmarks/data/{set_name}.txt"
        instances = parse_thpack(path, set_id=set_id)[:args.br_n]
        for inst in instances:
            for cname, kw in CONFIGS:
                key = (set_id, inst.instance_id, cname)
                if key in done:
                    continue
                t0 = time.time()
                r = brkga_pack_v35(
                    inst.boxes, inst.pallet, cfg,
                    time_limit_s=args.br_time, max_pallets=1,
                    population_size=600, n_populations=3,
                    patience=200, local_search_budget_s=4.0,
                    seed=42 + inst.instance_id, verbose=False,
                    **kw,
                )
                rt = time.time() - t0
                errs = validate(r, inst.pallet, cfg)
                row = asdict(BRRow(
                    set_id=set_id, instance_id=inst.instance_id,
                    n_boxes=len(inst.boxes),
                    config=cname,
                    util_pallet1=first_pallet_utilization(r, inst.pallet),
                    runtime_s=rt, validator_errors=len(errs),
                ))
                results.append(row)
                done.add(key)
                _save(results, outpath)
                print(f"  {set_id}#{inst.instance_id:>2} N={len(inst.boxes):>4d}  "
                      f"{cname}: util={row['util_pallet1']*100:>5.2f}%  "
                      f"{rt:>5.1f}s  errs={len(errs)}", flush=True)


def run_industry(args, results: List[Dict], outpath: str) -> None:
    cases = industry_cases()
    done = {(r["case_name"], r["config"])
            for r in results if "case_name" in r}
    print(f"\n=== Industry: {len(cases)} scenarios, "
          f"time={args.ind_time}s/pallet ===", flush=True)
    cname = "v3.8"
    for case in cases:
        key = (case.name, cname)
        if key in done:
            continue
        # Industry cases have full constraints; run multi-pallet.
        # Cap at min_pallets * 1.5 as max so we don't search forever.
        total_vol = sum(b.length * b.width * b.height for b in case.boxes)
        pallet_vol = case.pallet.length * case.pallet.width * case.pallet.height
        min_p_vol = max(1, int(total_vol / pallet_vol + 0.9999))
        total_wt = sum(b.weight for b in case.boxes)
        min_p_wt = max(1, int(total_wt / max(case.pallet.max_weight, 1) + 0.9999))
        min_pallets = max(min_p_vol, min_p_wt)
        max_pallets = min_pallets + 2  # margin

        t0 = time.time()
        r = brkga_pack_v35(
            case.boxes, case.pallet, case.config,
            time_limit_s=args.ind_time, max_pallets=max_pallets,
            population_size=400, n_populations=3,
            patience=150, local_search_budget_s=2.0,
            seed=42, verbose=False, n_modes=6,
        )
        rt = time.time() - t0
        errs = validate(r, case.pallet, case.config)
        row = asdict(INDRow(
            case_name=case.name, n_boxes=len(case.boxes), config=cname,
            pallets_used=r.num_pallets,
            unpacked=len(r.unpacked),
            util_pallet1=first_pallet_util(r, case.pallet),
            util_total=total_util(r, case.pallet),
            runtime_s=rt, validator_errors=len(errs),
        ))
        results.append(row)
        done.add(key)
        _save(results, outpath)
        print(f"  {case.name[:45]:<45}  N={len(case.boxes):>3d}  "
              f"pals={r.num_pallets:>2d}/{max_pallets}  "
              f"unp={len(r.unpacked):>3d}  "
              f"util1={row['util_pallet1']*100:>5.2f}%  "
              f"{rt:>5.1f}s  errs={len(errs)}", flush=True)


def _save(results: List[Dict], outpath: str) -> None:
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(results, f, indent=1)


def summarize(results: List[Dict]) -> None:
    br_rows = [r for r in results if "set_id" in r]
    ind_rows = [r for r in results if "case_name" in r]
    if br_rows:
        print("\n=== BR Summary ===", flush=True)
        by = {}
        for r in br_rows:
            by.setdefault((r["set_id"], r["config"]), []).append(r["util_pallet1"])
        sets = sorted({k[0] for k in by})
        cfgs = sorted({k[1] for k in by})
        print(f"{'set':<5}  " + "  ".join(f"{c:<14}" for c in cfgs) + "  Δ(v3.8-v3.6)")
        for sid in sets:
            cells = []
            vals = {}
            for c in cfgs:
                xs = by.get((sid, c), [])
                m = statistics.mean(xs) * 100 if xs else 0
                s = statistics.stdev(xs) * 100 if len(xs) > 1 else 0
                vals[c] = m
                cells.append(f"{m:>5.2f}±{s:>4.2f} (n={len(xs):<2d})")
            delta = vals.get("v3.8", 0) - vals.get("v3.6", 0)
            print(f"{sid:<5}  " + "  ".join(cells) + f"  {delta:+5.2f}pp")
    if ind_rows:
        print("\n=== Industry Summary ===", flush=True)
        print(f"{'case':<50}  {'N':>4}  {'pals':>4}  {'unp':>4}  {'util1%':>6}  {'time':>6}")
        for r in ind_rows:
            print(f"{r['case_name'][:50]:<50}  {r['n_boxes']:>4d}  "
                  f"{r['pallets_used']:>4d}  {r['unpacked']:>4d}  "
                  f"{r['util_pallet1']*100:>6.2f}  {r['runtime_s']:>5.1f}s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+",
                    default=["thpack1", "thpack3", "thpack5", "thpack7"])
    ap.add_argument("--br-n", type=int, default=10)
    ap.add_argument("--br-time", type=float, default=30.0)
    ap.add_argument("--ind-time", type=float, default=30.0)
    ap.add_argument("--skip-br", action="store_true")
    ap.add_argument("--skip-ind", action="store_true")
    ap.add_argument("--out", default="results/checkpoints/v38_full_eval.json")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    results: List[Dict] = []
    if os.path.exists(args.out):
        with open(args.out) as f:
            results = json.load(f)
        print(f"resumed: {len(results)} prior rows in {args.out}", flush=True)

    if args.summary_only:
        summarize(results)
        return 0

    print("warming up JIT...", flush=True)
    warmup_jit()

    if not args.skip_br:
        run_br(args, results, args.out)
    if not args.skip_ind:
        run_industry(args, results, args.out)
    summarize(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
