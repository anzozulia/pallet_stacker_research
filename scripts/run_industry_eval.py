"""
run_industry_eval.py — Evaluate the packer on realistic industry scenarios.

Runs every case in benchmarks/industry.py under multiple configs and
reports pallets used / utilization / runtime / validator-clean.

Configs:
  v1        — current default (no Phase 2+ features)
  +mip      — adds use_mip_polish=True (with adaptive budget per Option B)
  +layer    — adds use_layer_building=True
  quality   — combined: mip + layer + sku_consistent_rotation + ejection

Output:
  results/checkpoints/industry_results.json
  Console summary table
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
from dataclasses import asdict, dataclass, replace
from typing import Dict, List

from pallet_packer import PackerConfig, PalletPacker, validate
from benchmarks.industry import cases


@dataclass
class InstanceResult:
    case_name: str
    n_boxes: int
    config: str
    pallets_used: int
    unpacked: int
    util_pallet1: float
    util_total: float
    runtime_s: float
    validator_errors: int


def make_configs(base: PackerConfig) -> Dict[str, PackerConfig]:
    return {
        "v1": base,
        "+mip": replace(base, use_mip_polish=True),
        "+layer": replace(base, use_layer_building=True),
        "quality": replace(
            base,
            use_mip_polish=True,
            use_layer_building=True,
            use_block_building=True,
            sku_consistent_rotation=True,
            use_ejection_chains=True,
            ejection_max_depth=2,
            ejection_max_iters=50,
        ),
    }


def first_pallet_util(result, pallet):
    if not result.pallets:
        return 0.0
    st = result.pallets[0]
    used = sum(p.box.volume for p in st.placements)
    cap = pallet.length * pallet.width * pallet.height
    return used / cap if cap > 0 else 0.0


def run_case(case, cfg: PackerConfig, cfg_name: str) -> InstanceResult:
    packer = PalletPacker(case.pallet, cfg)
    t0 = time.time()
    result = packer.pack(case.boxes)
    rt = time.time() - t0
    errs = validate(result, case.pallet, cfg)
    return InstanceResult(
        case_name=case.name,
        n_boxes=len(case.boxes),
        config=cfg_name,
        pallets_used=result.num_pallets,
        unpacked=len(result.unpacked),
        util_pallet1=first_pallet_util(result, case.pallet),
        util_total=result.total_volume_utilisation,
        runtime_s=rt,
        validator_errors=len(errs),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+",
                    default=["v1", "+mip", "+layer", "quality"])
    ap.add_argument("--out",
                    default="results/checkpoints/industry_results.json")
    args = ap.parse_args()

    all_cases = cases()
    print(f"Industry cases: {len(all_cases)}", file=sys.stderr)
    cfg_map = make_configs(all_cases[0].config)
    selected = {k: v for k, v in cfg_map.items() if k in args.configs}
    print(f"Configs: {list(selected.keys())}", file=sys.stderr)

    results: List[Dict] = []
    if os.path.exists(args.out):
        with open(args.out) as f:
            results = json.load(f)
    done = {(r["case_name"], r["config"]) for r in results}

    print(f"\n{'Case':<46} {'Cfg':<10} {'N':>4} {'pals':>5} {'unp':>4} "
          f"{'util%':>6} {'rt':>7}")
    print("-" * 90)
    for case in all_cases:
        for cfg_name, cfg in selected.items():
            if (case.name, cfg_name) in done:
                continue
            # Each case has its own config; merge with the eval config.
            merged = replace(case.config,
                             use_mip_polish=cfg.use_mip_polish,
                             use_layer_building=cfg.use_layer_building,
                             use_block_building=cfg.use_block_building,
                             sku_consistent_rotation=cfg.sku_consistent_rotation,
                             use_ejection_chains=cfg.use_ejection_chains,
                             ejection_max_depth=cfg.ejection_max_depth,
                             ejection_max_iters=cfg.ejection_max_iters)
            res = run_case(case, merged, cfg_name)
            results.append(asdict(res))
            print(f"{case.name:<46} {cfg_name:<10} {res.n_boxes:>4d} "
                  f"{res.pallets_used:>5d} {res.unpacked:>4d} "
                  f"{res.util_total*100:>5.1f}% {res.runtime_s:>6.2f}s")
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(results, f, indent=1)

    # Summary
    print("\n=== Comparison (default config vs best of new configs) ===\n")
    by_case = {}
    for r in results:
        by_case.setdefault(r["case_name"], {})[r["config"]] = r
    for case_name in sorted(by_case.keys()):
        rows = by_case[case_name]
        v1 = rows.get("v1")
        best_cfg = min(rows.values(),
                       key=lambda r: (r["pallets_used"] + r["unpacked"],
                                      -r["util_total"]))
        if v1 is None:
            continue
        delta_pallets = best_cfg["pallets_used"] - v1["pallets_used"]
        delta_util = (best_cfg["util_total"] - v1["util_total"]) * 100
        delta_str = "—" if best_cfg["config"] == "v1" else \
            f"{delta_pallets:+d}p / {delta_util:+.1f}pp util via {best_cfg['config']}"
        print(f"{case_name:<46} v1={v1['pallets_used']}p/{v1['util_total']*100:.1f}%  "
              f"best={best_cfg['pallets_used']}p/{best_cfg['util_total']*100:.1f}%  "
              f"{delta_str}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
