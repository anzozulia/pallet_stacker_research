"""run_v310_industry_eval.py — v3.10 constraint-aware decoder on industry.

Runs all 10 industry scenarios with v3.10 (constraint-aware via
Box.max_load_on_top + Pallet.max_weight enforcement). Compares against
the prior v3.8 results that ignored constraints.
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, json, time
from dataclasses import asdict, dataclass

from pallet_packer import validate
from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit
from benchmarks.industry import cases as industry_cases


@dataclass
class Row:
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", type=float, default=30.0)
    ap.add_argument("--out", default="results/checkpoints/v310_industry_eval.json")
    args = ap.parse_args()

    print("warming up JIT...", flush=True)
    warmup_jit()
    cases = industry_cases()

    results = []
    if os.path.exists(args.out):
        with open(args.out) as f:
            results = json.load(f)
    done = {(r["case_name"], r["config"]) for r in results}

    print(f"\n=== v3.10 constraint-aware industry eval (10 cases, "
          f"time={args.time}s/pallet) ===", flush=True)
    for case in cases:
        if (case.name, "v3.10") in done:
            continue
        total_vol = sum(b.length * b.width * b.height for b in case.boxes)
        pallet_vol = case.pallet.length * case.pallet.width * case.pallet.height
        min_p_vol = max(1, int(total_vol / pallet_vol + 0.9999))
        total_wt = sum(b.weight for b in case.boxes)
        min_p_wt = max(1, int(total_wt / max(case.pallet.max_weight, 1) + 0.9999))
        min_pallets = max(min_p_vol, min_p_wt)
        max_pallets = min_pallets + 2

        t0 = time.time()
        r = brkga_pack_v35(
            case.boxes, case.pallet, case.config,
            time_limit_s=args.time, max_pallets=max_pallets,
            population_size=400, n_populations=3,
            patience=150, local_search_budget_s=2.0,
            seed=42, verbose=False, n_modes=6,
        )
        rt = time.time() - t0
        errs = validate(r, case.pallet, case.config)
        row = asdict(Row(
            case_name=case.name, n_boxes=len(case.boxes), config="v3.10",
            pallets_used=r.num_pallets, unpacked=len(r.unpacked),
            util_pallet1=first_pallet_util(r, case.pallet),
            util_total=total_util(r, case.pallet),
            runtime_s=rt, validator_errors=len(errs),
        ))
        results.append(row)
        done.add((case.name, "v3.10"))
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
        print(f"  {case.name[:48]:<48}  pals={r.num_pallets:>2d}/{max_pallets}  "
              f"unp={len(r.unpacked):>3d}  u1={row['util_pallet1']*100:>5.2f}%  "
              f"utT={row['util_total']*100:>5.2f}%  {rt:>5.1f}s  errs={len(errs)}",
              flush=True)

    # Comparison with v3.8 prior results
    v38_path = "results/checkpoints/v38_full_eval.json"
    v38 = {}
    if os.path.exists(v38_path):
        old = json.load(open(v38_path))
        v38 = {r["case_name"]: r for r in old if "case_name" in r}
    print("\n=== v3.8 (geometric) vs v3.10 (constraint-aware) ===", flush=True)
    print(f"{'case':<48} {'v3.8':<25} {'v3.10':<25} {'Δ pals':>6}  {'Δ errs':>6}")
    for r in results:
        old = v38.get(r["case_name"])
        if not old:
            continue
        old_str = (f"{old['pallets_used']}p u1={old['util_pallet1']*100:.1f}% "
                   f"errs={old['validator_errors']}")
        new_str = (f"{r['pallets_used']}p u1={r['util_pallet1']*100:.1f}% "
                   f"errs={r['validator_errors']}")
        dp = r['pallets_used'] - old['pallets_used']
        de = r['validator_errors'] - old['validator_errors']
        print(f"{r['case_name'][:48]:<48} {old_str:<25} {new_str:<25} "
              f"{dp:>+6d}  {de:>+6d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
