"""
run_full_br_eval.py — Comprehensive BR1/3/5/7 evaluation across configs.

Runs all 100 instances of each BR set under multiple configs and reports
per-set + per-instance utilization, runtime, validator-clean status.

Configs:
  v1         — current default (no Phase 2+ features)
  v1+layer   — adds use_layer_building=True (the BR-quality driver per
               docs/reports/08_br_deep_eval.md)
  v1+lksku   — layer + sku_consistent_rotation (Bortfeldt-Gehring 2001)

Outputs:
  results/checkpoints/br_full_results.json  (per-instance raw data)
  Console summary
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

from pallet_packer import PackerConfig, PalletPacker, validate
from benchmarks.br import parse_thpack, geometric_only_config, first_pallet_utilization


@dataclass
class InstanceResult:
    set_id: str
    instance_id: int
    n_boxes: int
    config: str
    pallets_used: int
    unpacked: int
    util_pallet1: float
    util_total: float
    runtime_s: float
    validator_errors: int


def make_configs() -> Dict[str, PackerConfig]:
    base = geometric_only_config(max_pallets=1)
    layer = PackerConfig(
        support_ratio=base.support_ratio,
        require_centroid_supported=base.require_centroid_supported,
        allow_pallet_overhang=base.allow_pallet_overhang,
        heavy_on_bottom=base.heavy_on_bottom,
        cog_envelope_fraction=base.cog_envelope_fraction,
        cog_check_min_load_fraction=base.cog_check_min_load_fraction,
        enforce_load_bearing=base.enforce_load_bearing,
        multi_start_trials=1,
        seed=42,
        max_pallets=1,
        optimize="max_util",
        use_layer_building=True,
    )
    layer_sku = PackerConfig(
        support_ratio=base.support_ratio,
        require_centroid_supported=base.require_centroid_supported,
        allow_pallet_overhang=base.allow_pallet_overhang,
        heavy_on_bottom=base.heavy_on_bottom,
        cog_envelope_fraction=base.cog_envelope_fraction,
        cog_check_min_load_fraction=base.cog_check_min_load_fraction,
        enforce_load_bearing=base.enforce_load_bearing,
        multi_start_trials=1,
        seed=42,
        max_pallets=1,
        optimize="max_util",
        use_layer_building=True,
        sku_consistent_rotation=True,
    )
    return {"v1": base, "v1+layer": layer, "v1+lksku": layer_sku}


def run_instance(instance, cfg: PackerConfig) -> InstanceResult:
    packer = PalletPacker(instance.pallet, cfg)
    t0 = time.time()
    result = packer.pack(instance.boxes)
    rt = time.time() - t0
    errs = validate(result, instance.pallet, cfg)
    return InstanceResult(
        set_id=instance.set_id,
        instance_id=instance.instance_id,
        n_boxes=len(instance.boxes),
        config="",
        pallets_used=result.num_pallets,
        unpacked=len(result.unpacked),
        util_pallet1=first_pallet_utilization(result, instance.pallet),
        util_total=result.total_volume_utilisation,
        runtime_s=rt,
        validator_errors=len(errs),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+",
                    default=["thpack1", "thpack3", "thpack5", "thpack7"])
    ap.add_argument("--configs", nargs="+",
                    default=["v1", "v1+layer"])
    ap.add_argument("--max-instances", type=int, default=None,
                    help="Cap instances per set (default: all 100)")
    ap.add_argument("--skip-huge-n", type=int, default=300,
                    help="Skip instances with N > this")
    ap.add_argument("--out",
                    default="results/checkpoints/br_full_results.json")
    args = ap.parse_args()

    configs = make_configs()
    selected = {k: v for k, v in configs.items() if k in args.configs}
    print(f"Configs to run: {list(selected.keys())}", file=sys.stderr)

    all_results: List[Dict] = []
    if os.path.exists(args.out):
        with open(args.out) as f:
            all_results = json.load(f)
        print(f"Resuming with {len(all_results)} existing results",
              file=sys.stderr)
    done = {(r["set_id"], r["instance_id"], r["config"]) for r in all_results}

    for set_name in args.sets:
        path = f"benchmarks/data/{set_name}.txt"
        instances = parse_thpack(path, set_id=set_name.upper().replace("THPACK", "BR"))
        if args.max_instances:
            instances = instances[:args.max_instances]
        print(f"\n=== {set_name}: {len(instances)} instances ===",
              file=sys.stderr)
        for cfg_name, cfg in selected.items():
            t_set = time.time()
            n_done = 0
            for inst in instances:
                if (inst.set_id, inst.instance_id, cfg_name) in done:
                    continue
                if len(inst.boxes) > args.skip_huge_n:
                    print(f"  SKIP {inst.set_id}#{inst.instance_id} "
                          f"N={len(inst.boxes)} (> --skip-huge-n)",
                          file=sys.stderr)
                    continue
                res = run_instance(inst, cfg)
                res.config = cfg_name
                all_results.append(asdict(res))
                n_done += 1
                if n_done % 10 == 0:
                    os.makedirs(os.path.dirname(args.out), exist_ok=True)
                    with open(args.out, "w") as f:
                        json.dump(all_results, f, indent=1)
                    print(f"  [{cfg_name}] {n_done}/{len(instances)} "
                          f"elapsed={time.time()-t_set:.1f}s",
                          file=sys.stderr)
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(all_results, f, indent=1)
            relevant = [r for r in all_results
                        if r["set_id"].endswith(set_name[-1])
                        and r["config"] == cfg_name]
            if relevant:
                utils = [r["util_pallet1"] for r in relevant]
                print(f"  [{cfg_name}] {set_name}: n={len(relevant)} "
                      f"mean={statistics.mean(utils)*100:.1f}% "
                      f"stdev={statistics.stdev(utils)*100 if len(utils)>1 else 0:.1f} "
                      f"range=[{min(utils)*100:.1f}, {max(utils)*100:.1f}] "
                      f"set_time={time.time()-t_set:.1f}s",
                      file=sys.stderr)

    print("\n=== Summary ===", file=sys.stderr)
    by_key = {}
    for r in all_results:
        key = (r["set_id"], r["config"])
        by_key.setdefault(key, []).append(r["util_pallet1"])
    for (set_id, cfg), utils in sorted(by_key.items()):
        print(f"{set_id:6s} {cfg:12s}: n={len(utils):3d} "
              f"mean={statistics.mean(utils)*100:5.1f}% "
              f"stdev={statistics.stdev(utils)*100 if len(utils)>1 else 0:.1f}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
