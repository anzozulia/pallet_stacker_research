"""
run_evaluation.py — Quality-focused evaluation of the current packer.

Runs every case in the 41-case suite under TWO configurations:
  v1            — baseline (no Phase 2+ features, multi_start_trials=1).
  quality_max   — every quality-improving feature enabled with generous
                  compute budgets. The point is to measure what the packer
                  can achieve when speed is not a constraint.

Reports per-case (vs LB):
  v1 result, quality_max result, gap-to-best-LB, mechanism that closed
  the case (or "still open" if neither hit LB).

JSON-checkpointed — runs in chunks, resumes from where it left off.

Outputs:
  evaluation_results.json  per-case raw results
  EVALUATION_REPORT.md     consolidated report
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
from typing import Dict, List, Optional, Tuple

from pallet_packer import (
    Box, Pallet, PalletPacker, PackerConfig, validate,
)
from pallet_packer.lower_bounds import compute_lower_bounds
from benchmarks import internal as benchmark
from benchmarks import failure_cases


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------
def v1_config(seed: int = 42) -> PackerConfig:
    return PackerConfig(seed=seed, multi_start_trials=1)


def quality_max_config(seed: int = 42) -> PackerConfig:
    """Every quality-improving feature on, with moderate budgets sized to
    keep evaluation tractable. Calibrated so each case completes in <30s
    while still exercising every quality-improving feature.
    """
    return PackerConfig(
        seed=seed,
        multi_start_trials=4,
        use_block_building=True,
        block_threshold=4,
        use_brkga=True,
        brkga_population_size=20,
        brkga_generations=6,
        brkga_elite_fraction=0.20,
        brkga_mutant_fraction=0.15,
        brkga_p_elite=0.70,
        brkga_n_threshold=None,
        sku_consistent_rotation=True,
        grasp_alpha=3,
        use_ejection_chains=True,
        ejection_max_depth=2,
        ejection_max_iters=100,
        use_safety_net=True,
        use_layer_building=True,
        layer_axis="all",
        use_mip_polish=True,
        mip_n_threshold=25,                 # MIP only on small instances
        mip_time_limit_s=10.0,              # tight cap; CP-SAT converges fast at N<=25
        mip_num_workers=4,
    )


def merged_config(case_cfg: PackerConfig, abl: PackerConfig) -> PackerConfig:
    """Merge ablation features on top of case-specific physical constraints."""
    return PackerConfig(
        support_ratio=case_cfg.support_ratio,
        require_centroid_supported=case_cfg.require_centroid_supported,
        allow_pallet_overhang=case_cfg.allow_pallet_overhang,
        heavy_on_bottom=case_cfg.heavy_on_bottom,
        cog_envelope_fraction=case_cfg.cog_envelope_fraction,
        cog_check_min_load_fraction=case_cfg.cog_check_min_load_fraction,
        enforce_load_bearing=case_cfg.enforce_load_bearing,
        multi_start_trials=abl.multi_start_trials,
        seed=abl.seed,
        use_block_building=abl.use_block_building,
        block_threshold=abl.block_threshold,
        use_brkga=abl.use_brkga,
        brkga_population_size=abl.brkga_population_size,
        brkga_generations=abl.brkga_generations,
        brkga_elite_fraction=abl.brkga_elite_fraction,
        brkga_mutant_fraction=abl.brkga_mutant_fraction,
        brkga_p_elite=abl.brkga_p_elite,
        brkga_n_threshold=abl.brkga_n_threshold,
        sku_consistent_rotation=abl.sku_consistent_rotation,
        grasp_alpha=abl.grasp_alpha,
        use_ejection_chains=abl.use_ejection_chains,
        ejection_max_depth=abl.ejection_max_depth,
        ejection_max_iters=abl.ejection_max_iters,
        use_safety_net=abl.use_safety_net,
        use_layer_building=abl.use_layer_building,
        layer_axis=abl.layer_axis,
        use_mip_polish=abl.use_mip_polish,
        mip_n_threshold=abl.mip_n_threshold,
        mip_time_limit_s=abl.mip_time_limit_s,
        mip_num_workers=abl.mip_num_workers,
    )


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------
def all_cases():
    for c in benchmark.cases():
        yield "bench", c.name, list(c.boxes), c.pallet, c.config
    for factory in [
        failure_cases.case_F1_pareto_continuum,
        failure_cases.case_F2_block_of_identicals,
        failure_cases.case_F3_interlock_pattern,
        failure_cases.case_F4_height_mismatch,
        failure_cases.case_F5_must_rotate_early,
        failure_cases.case_F6_constraint_cascade,
        failure_cases.case_F7_layered_pyramid,
        failure_cases.case_F8_strongly_heterogeneous,
        failure_cases.case_F9_long_unrotatable,
        failure_cases.case_F10_overhang_required,
        failure_cases.case_F11_spurious_unpacked,
        failure_cases.case_F12_strongly_heterogeneous_at_scale,
        failure_cases.case_F13_layer_advantage,
    ]:
        c = factory()
        yield "fail", c.name, c.boxes_factory(), c.pallet, c.config


@dataclass
class Cell:
    pallets: int = 0
    unpacked: int = 0
    util_pct: float = 0.0
    runtime_s: float = 0.0
    errors: int = 0


def run_case(boxes: List[Box], pallet: Pallet, cfg: PackerConfig) -> Cell:
    packer = PalletPacker(pallet, cfg)
    t0 = time.time()
    result = packer.pack(list(boxes))
    elapsed = time.time() - t0
    errs = validate(result, pallet, cfg)
    return Cell(
        pallets=result.num_pallets,
        unpacked=len(result.unpacked),
        util_pct=result.total_volume_utilisation * 100,
        runtime_s=elapsed,
        errors=len(errs),
    )


def load_existing(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save(path: str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    argv = argv or []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/checkpoints/evaluation_results.json")
    ap.add_argument("--time-cap", type=float, default=35,
                    help="Approximate wall-clock cap per session.")
    ap.add_argument("--skip-huge", type=int, default=120,
                    help="Skip cases with N > this (keep eval tractable).")
    ap.add_argument("--only-quality-max", action="store_true",
                    help="Only run quality_max (assumes v1 already cached).")
    args = ap.parse_args(argv)

    cache = load_existing(args.out)

    configs = {}
    if not args.only_quality_max:
        configs["v1"] = v1_config()
    configs["quality_max"] = quality_max_config()

    cases = list(all_cases())
    start = time.time()

    for suite, name, boxes, pallet, case_cfg in cases:
        if len(boxes) > args.skip_huge:
            continue
        key = f"{suite}|{name}"
        row = cache.setdefault(key, {})
        # Compute LBs once per case.
        if "lb" not in row:
            lbs = compute_lower_bounds(boxes, pallet, case_cfg)
            row["lb"] = {
                "volume": lbs.volume,
                "per_sku_max": lbs.per_sku_max,
                "geometric": lbs.geometric,
                "lp": lbs.lp,
                "best": lbs.best,
            }
            row["n_boxes"] = len(boxes)
            save(args.out, cache)
        for cfg_name, abl_cfg in configs.items():
            if cfg_name in row:
                continue
            if time.time() - start > args.time_cap:
                print(f"time cap reached", file=sys.stderr)
                save(args.out, cache)
                return 0
            cfg = merged_config(case_cfg, abl_cfg)
            cell = run_case(boxes, pallet, cfg)
            row[cfg_name] = asdict(cell)
            print(f"  {suite}|{name:<40} {cfg_name:<12} "
                  f"{cell.pallets}p/{cell.unpacked}u/{cell.util_pct:.1f}% "
                  f"t={cell.runtime_s:.1f}s err={cell.errors}",
                  file=sys.stderr)
            save(args.out, cache)
    print("DONE", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
