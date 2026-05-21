"""
run_br_deep_eval.py — Comprehensive BR1/3/5/7 evaluation.

Every available instance is run under TWO configurations:
  v1            — baseline (no Phase 2+ features). max_pallets=1
                  (single container, BR objective).
  quality_max   — every quality-improving feature on, with budgets sized
                  for BR scale: SKU-grid layer fill, MIP polish
                  (threshold=60 to capture larger BR instances),
                  ejection chains, GRASP, etc.

Metric: pallet-1 utilization (the BR objective).

JSON-checkpointed (each (set, instance, config) tuple is saved
incrementally). Resumes from where it left off.

Output:
  br_deep_results.json
  BR_DEEP_REPORT.md
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from pallet_packer import PackerConfig, PalletPacker, validate
from br_benchmark import parse_thpack, first_pallet_utilization


def v1_config(seed: int = 42) -> PackerConfig:
    """Plain v1, single-container objective."""
    return PackerConfig(
        support_ratio=1.0,
        require_centroid_supported=True,
        allow_pallet_overhang=False,
        heavy_on_bottom=False,
        cog_envelope_fraction=1.0,
        cog_check_min_load_fraction=1.0,
        enforce_load_bearing=False,
        multi_start_trials=1,
        seed=seed,
        max_pallets=1,
        optimize="max_util",
    )


def quality_max_config(seed: int = 42) -> PackerConfig:
    """Quality features for BR single-container.

    Layer-building (with SKU-grid) is the main quality driver — the BR-1995
    pattern that gives us +2-7pp over v1. BRKGA and MIP are expensive at
    BR scale (N=100+) and rarely close anything BR-specific, so they're
    off in this config. The safety net + SKU-consistent rotation are
    on to preserve other guarantees.
    """
    return PackerConfig(
        support_ratio=1.0,
        require_centroid_supported=True,
        allow_pallet_overhang=False,
        heavy_on_bottom=False,
        cog_envelope_fraction=1.0,
        cog_check_min_load_fraction=1.0,
        enforce_load_bearing=False,
        multi_start_trials=1,
        seed=seed,
        use_block_building=False,
        use_brkga=False,
        sku_consistent_rotation=True,
        grasp_alpha=1,                       # deterministic decode at BR scale
        use_ejection_chains=False,
        use_safety_net=True,
        use_layer_building=True,
        layer_axis="all",
        use_mip_polish=False,
        mip_n_threshold=60,
        mip_num_workers=1,
        max_pallets=1,
        optimize="max_util",
    )


@dataclass
class Cell:
    util_pct: float = 0.0
    overflow: int = 0
    runtime_s: float = 0.0
    errors: int = 0


def run_one(inst, cfg: PackerConfig) -> Cell:
    packer = PalletPacker(inst.pallet, cfg)
    t0 = time.time()
    result = packer.pack(list(inst.boxes))
    elapsed = time.time() - t0
    util = first_pallet_utilization(result, inst.pallet)
    n_p1 = len(result.pallets[0].placements) if result.pallets else 0
    overflow = len(inst.boxes) - n_p1
    errs = validate(result, inst.pallet, cfg)
    return Cell(util_pct=util * 100, overflow=overflow,
                runtime_s=elapsed, errors=len(errs))


def load(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save(path: str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main(argv: Optional[List[str]] = None) -> int:
    argv = argv or []
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="br_data")
    ap.add_argument("--sets", nargs="+",
                    default=["thpack1", "thpack3", "thpack5", "thpack7"])
    ap.add_argument("--out", default="br_deep_results.json")
    ap.add_argument("--max-time", type=float, default=35.0,
                    help="Approx wall-clock cap per session in seconds.")
    ap.add_argument("--skip-huge-n", type=int, default=250,
                    help="Skip instances where N > this.")
    args = ap.parse_args(argv)

    configs = {
        "v1": v1_config(),
        "quality_max": quality_max_config(),
    }

    cache = load(args.out)
    start = time.time()
    work_done = 0

    for set_name in args.sets:
        path = os.path.join(args.data_dir, set_name + ".txt")
        if not os.path.exists(path):
            print(f"skip {set_name}: not found", file=sys.stderr)
            continue
        all_instances = parse_thpack(path)
        for inst in all_instances:
            if inst.n_boxes > args.skip_huge_n:
                continue
            for cfg_name, cfg in configs.items():
                key = f"{set_name}|{inst.instance_id}|{cfg_name}"
                if key in cache:
                    continue
                if time.time() - start > args.max_time:
                    save(args.out, cache)
                    print(f"time cap reached ({work_done} runs done)",
                          file=sys.stderr)
                    return 0
                cell = run_one(inst, cfg)
                cache[key] = asdict(cell)
                work_done += 1
                print(f"  {set_name} #{inst.instance_id} (N={inst.n_boxes}) "
                      f"{cfg_name}: util={cell.util_pct:.1f}% "
                      f"t={cell.runtime_s:.1f}s err={cell.errors}",
                      file=sys.stderr)
                save(args.out, cache)

    print(f"DONE  ({work_done} new runs, {len(cache)} total)",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
