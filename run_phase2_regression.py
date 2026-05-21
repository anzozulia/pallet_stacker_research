"""
run_phase2_regression.py — Regression + ablation across the 41-case suite.

Writes incremental JSON so the run can be inspected mid-flight or split into
chunks. Final summary printed at end.

Usage:
    python3 run_phase2_regression.py [--full] [--ablation] [--out FILE]
        --full       include the two slow cases (B4 + F2) — adds ~30s
        --ablation   run 6 configs per case for feature isolation (slow)
                     default is 3 configs (v1, +grasp+ejection, all_on)
        --out        results JSON path (default phase2_results.json)
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

from pallet_packer import (
    Box, Pallet, PalletPacker, PackerConfig, validate,
)
import benchmark
import failure_cases


def make_configs(ablation: bool = False) -> Dict[str, PackerConfig]:
    common = dict(seed=42, multi_start_trials=1)
    configs = {
        "v1": PackerConfig(**common),
        "v1+layer": PackerConfig(**common, use_layer_building=True),
        "v1+mip": PackerConfig(**common, use_mip_polish=True,
                               mip_n_threshold=50, mip_time_limit_s=30),
        "all_on": PackerConfig(
            **common,
            use_block_building=True, use_brkga=True,
            brkga_population_size=16, brkga_generations=4,
            brkga_n_threshold=None,
            grasp_alpha=3,
            sku_consistent_rotation=True,
            use_ejection_chains=True,
            ejection_max_depth=2,
            ejection_max_iters=50,
            use_layer_building=True,
            use_mip_polish=True,
            mip_n_threshold=50, mip_time_limit_s=30,
        ),
    }
    if ablation:
        configs.update({
            "+grasp": PackerConfig(**common, grasp_alpha=3,
                                   use_brkga=True, brkga_population_size=16,
                                   brkga_generations=4, brkga_n_threshold=200),
            "+sku_lock": PackerConfig(**common,
                                      sku_consistent_rotation=True),
            "+ejection": PackerConfig(**common,
                                      use_ejection_chains=True,
                                      ejection_max_depth=2,
                                      ejection_max_iters=50),
        })
    return configs


@dataclass
class RunCell:
    pallets: int = 0
    unpacked: int = 0
    util: float = 0.0
    runtime: float = 0.0
    errors: int = 0


def _benchmark_cases():
    for c in benchmark.cases():
        yield "bench", c.name, list(c.boxes), c.pallet, c.config


def _failure_cases():
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


def merged_config(case_cfg: PackerConfig, ablation: PackerConfig) -> PackerConfig:
    return PackerConfig(
        support_ratio=case_cfg.support_ratio,
        require_centroid_supported=case_cfg.require_centroid_supported,
        allow_pallet_overhang=case_cfg.allow_pallet_overhang,
        heavy_on_bottom=case_cfg.heavy_on_bottom,
        cog_envelope_fraction=case_cfg.cog_envelope_fraction,
        cog_check_min_load_fraction=case_cfg.cog_check_min_load_fraction,
        enforce_load_bearing=case_cfg.enforce_load_bearing,
        multi_start_trials=ablation.multi_start_trials,
        seed=ablation.seed,
        use_block_building=ablation.use_block_building,
        block_threshold=ablation.block_threshold,
        use_brkga=ablation.use_brkga,
        brkga_population_size=ablation.brkga_population_size,
        brkga_generations=ablation.brkga_generations,
        brkga_elite_fraction=ablation.brkga_elite_fraction,
        brkga_mutant_fraction=ablation.brkga_mutant_fraction,
        brkga_p_elite=ablation.brkga_p_elite,
        brkga_n_threshold=ablation.brkga_n_threshold,
        sku_consistent_rotation=ablation.sku_consistent_rotation,
        grasp_alpha=ablation.grasp_alpha,
        use_ejection_chains=ablation.use_ejection_chains,
        ejection_max_depth=ablation.ejection_max_depth,
        ejection_max_iters=ablation.ejection_max_iters,
        use_layer_building=getattr(ablation, "use_layer_building", False),
        layer_axis=getattr(ablation, "layer_axis", "all"),
        use_safety_net=getattr(ablation, "use_safety_net", True),
        max_pallets=getattr(ablation, "max_pallets", None),
        use_mip_polish=getattr(ablation, "use_mip_polish", False),
        mip_n_threshold=getattr(ablation, "mip_n_threshold", 25),
        mip_time_limit_s=getattr(ablation, "mip_time_limit_s", 30.0),
        mip_num_workers=getattr(ablation, "mip_num_workers", 4),
        optimize=getattr(ablation, "optimize", "min_unpacked"),
    )


def run_case(boxes: List[Box], pallet: Pallet, cfg: PackerConfig) -> RunCell:
    packer = PalletPacker(pallet, cfg)
    t0 = time.time()
    result = packer.pack(list(boxes))
    elapsed = time.time() - t0
    errs = validate(result, pallet, cfg)
    return RunCell(
        pallets=result.num_pallets,
        unpacked=len(result.unpacked),
        util=result.total_volume_utilisation,
        runtime=elapsed,
        errors=len(errs),
    )


def fmt_cell(c: RunCell, ref: Optional[RunCell] = None) -> str:
    s = f"{c.pallets}p/{c.unpacked}u/{c.util*100:.0f}%"
    if ref is not None:
        dp = c.pallets - ref.pallets
        du = c.unpacked - ref.unpacked
        if dp == 0 and du == 0:
            return s
        marker = "↓" if (dp < 0 or du < 0) else "↑"
        return f"{s}{marker}"
    return s


def load_existing(path: str) -> Dict[str, Dict[str, dict]]:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save(path: str, data: Dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main(argv: Optional[List[str]] = None) -> int:
    argv = argv or []
    full = "--full" in argv
    ablation = "--ablation" in argv
    out_path = "phase2_results.json"
    if "--out" in argv:
        out_path = argv[argv.index("--out") + 1]

    configs = make_configs(ablation=ablation)
    config_names = list(configs.keys())

    skip_slow = set() if full else {
        "B4 100 tiny boxes -> 1 pallet",          # 100 tiny boxes — EP explosion
        "F2 pharma: many fragile",                # N=120
        "F1 e-commerce: 50 small + 3 big",        # N=53, slow with safety net
        "C4 stress 80 boxes",                     # N=80, very slow
        "F12 strongly heterogeneous at scale",    # N=72
    }

    cases = []
    for tup in _benchmark_cases():
        cases.append(tup)
    for tup in _failure_cases():
        cases.append(tup)

    existing = load_existing(out_path)
    print(f"Resuming with {len(existing)} prior case rows.", file=sys.stderr)

    for suite, name, boxes, pallet, case_cfg in cases:
        if name in skip_slow:
            continue
        key = f"{suite}|{name}"
        row = existing.setdefault(key, {})
        for cfg_name, abl_cfg in configs.items():
            if cfg_name in row:
                continue
            cfg = merged_config(case_cfg, abl_cfg)
            cell = run_case(boxes, pallet, cfg)
            row[cfg_name] = asdict(cell)
            print(f"  {suite:<5} {name:<38} {cfg_name:<13} "
                  f"{fmt_cell(cell)} t={cell.runtime:.2f}s err={cell.errors}",
                  file=sys.stderr)
            # Checkpoint after every config.
            save(out_path, existing)

    # Summary table.
    print("\n" + "=" * 130)
    print("PHASE 2 REGRESSION + ABLATION")
    print("=" * 130)
    header = f"{'suite':<5} {'case':<38}"
    for cn in config_names:
        header += f" {cn:>14}"
    print(header)
    print("-" * len(header))
    for key, row in existing.items():
        suite, name = key.split("|", 1)
        v1 = RunCell(**row["v1"]) if "v1" in row else None
        line = f"{suite:<5} {name:<38}"
        for cn in config_names:
            if cn not in row:
                line += f" {'—':>14}"
                continue
            cell = RunCell(**row[cn])
            line += f" {fmt_cell(cell, v1):>14}"
        print(line)

    # Tally.
    regressions = 0
    improvements = 0
    val_errors = 0
    for key, row in existing.items():
        if "v1" not in row:
            continue
        v1 = RunCell(**row["v1"])
        for cn in config_names:
            if cn == "v1" or cn not in row:
                continue
            cell = RunCell(**row[cn])
            val_errors += cell.errors
            if (cell.unpacked > v1.unpacked or
                (cell.unpacked == v1.unpacked and cell.pallets > v1.pallets)):
                regressions += 1
                print(f"  REGRESSION: {key} [{cn}] {fmt_cell(cell)} vs v1 {fmt_cell(v1)}",
                      file=sys.stderr)
            elif (cell.unpacked < v1.unpacked or
                  (cell.unpacked == v1.unpacked and cell.pallets < v1.pallets)):
                improvements += 1

    print("\nSUMMARY")
    print(f"  cases: {len(existing)}")
    print(f"  configs: {config_names}")
    print(f"  validator errors: {val_errors}")
    print(f"  improvements vs v1: {improvements}")
    print(f"  regressions vs v1:  {regressions}")
    return 0 if regressions == 0 and val_errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
