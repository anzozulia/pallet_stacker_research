"""
run_deep_br.py — Deep Bischoff-Ratcliff benchmark with seed variance + Phase 2.

Per-set we sample N instances and run them under K seeds × M configs. Results
are checkpointed to JSON so the test can run in chunks (each chunk processes
one (set, instance, seed, config) tuple at a time).

Configs evaluated:
    v1                — no Phase 2 features (single trial), no v2.
    v2_legacy         — block + BRKGA at n_threshold=40 (old default).
    v2_phase2         — Phase 2 features all on, BRKGA threshold=200.
    all_on            — Phase 2 + block + BRKGA threshold=None.

Output:
    deep_br_results.json   per-instance, per-seed, per-config results
    DEEP_BR_REPORT.md      consolidated headline numbers w/ confidence intervals
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

from pallet_packer import PalletPacker, PackerConfig, validate
from br_benchmark import parse_thpack, BRInstance, first_pallet_utilization


def make_configs() -> Dict[str, dict]:
    """Config templates (seed filled in per-run)."""
    return {
        "v1": dict(),
        "v2_legacy": dict(
            use_block_building=True, use_brkga=True,
            brkga_population_size=16, brkga_generations=4,
            brkga_n_threshold=40,
        ),
        "v2_phase2": dict(
            grasp_alpha=3,
            sku_consistent_rotation=True,
            use_ejection_chains=True,
            ejection_max_depth=2, ejection_max_iters=50,
            use_brkga=True, brkga_population_size=16, brkga_generations=4,
            brkga_n_threshold=200,
        ),
        "v2_phase2_lite": dict(
            grasp_alpha=3,
            sku_consistent_rotation=True,
            use_ejection_chains=True,
            ejection_max_depth=2, ejection_max_iters=30,
            use_brkga=True, brkga_population_size=12, brkga_generations=3,
            brkga_n_threshold=200,
            use_safety_net=False,            # benchmark mode
        ),
        "all_on": dict(
            use_block_building=True, use_brkga=True,
            brkga_population_size=16, brkga_generations=4,
            brkga_n_threshold=None,
            grasp_alpha=3,
            sku_consistent_rotation=True,
            use_ejection_chains=True,
            ejection_max_depth=2, ejection_max_iters=50,
        ),
    }


def br_geometric_config(seed: int, overrides: dict) -> PackerConfig:
    """Construct a BR-style geometric-only config with given overrides."""
    base = dict(
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
        block_threshold=4,
        use_brkga=False,
        brkga_population_size=30,
        brkga_generations=8,
        brkga_elite_fraction=0.20,
        brkga_mutant_fraction=0.15,
        brkga_p_elite=0.70,
        brkga_n_threshold=200,
        sku_consistent_rotation=False,
        grasp_alpha=1,
        use_ejection_chains=False,
        ejection_max_depth=2,
        ejection_max_iters=50,
        use_safety_net=True,
        max_pallets=1,           # BR: single-container packing
    )
    base.update(overrides)
    return PackerConfig(**base)


@dataclass
class Cell:
    util_pct: float = 0.0
    overflow: int = 0
    runtime: float = 0.0
    errors: int = 0


def run_one(instance: BRInstance, cfg: PackerConfig) -> Cell:
    packer = PalletPacker(instance.pallet, cfg)
    t0 = time.time()
    result = packer.pack(list(instance.boxes))
    elapsed = time.time() - t0
    util = first_pallet_utilization(result, instance.pallet)
    n_p0 = len(result.pallets[0].placements) if result.pallets else 0
    overflow = len(instance.boxes) - n_p0
    errs = validate(result, instance.pallet, cfg)
    return Cell(util_pct=util * 100, overflow=overflow,
                runtime=elapsed, errors=len(errs))


def load(path: str) -> Dict[str, Dict]:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save(path: str, data: Dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main(argv: Optional[List[str]] = None) -> int:
    argv = argv or []
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="br_data")
    ap.add_argument("--sample", type=int, default=10,
                    help="Instances per BR set (default 10).")
    ap.add_argument("--seeds", type=int, default=5,
                    help="Number of seeds per (instance, config). Default 5.")
    ap.add_argument("--sets", nargs="+",
                    default=["thpack1", "thpack3", "thpack5"])
    ap.add_argument("--configs", nargs="+",
                    default=["v1", "v2_phase2"],
                    help="Which configs to run (subset of make_configs).")
    ap.add_argument("--out", default="deep_br_results.json")
    ap.add_argument("--max-time", type=float, default=35,
                    help="Approximate wall-clock cap in seconds before exit.")
    ap.add_argument("--skip-huge-n", type=int, default=200,
                    help="Skip instances where N > this. Default 200.")
    args = ap.parse_args(argv)

    configs_all = make_configs()
    for c in args.configs:
        if c not in configs_all:
            print(f"unknown config: {c}", file=sys.stderr)
            return 1

    cache = load(args.out)
    start = time.time()
    work_done = 0

    for set_name in args.sets:
        path = os.path.join(args.data_dir, set_name + ".txt")
        if not os.path.exists(path):
            print(f"skip {set_name}: not found", file=sys.stderr)
            continue
        all_instances = parse_thpack(path)
        instances = [i for i in all_instances if i.n_boxes <= args.skip_huge_n][: args.sample]
        skipped = len(all_instances) - len(instances)
        if skipped > 0:
            print(f"  {set_name}: skipping {skipped} instances with N > {args.skip_huge_n}",
                  file=sys.stderr)
        for inst in instances:
            for seed_idx in range(args.seeds):
                seed = 42 + seed_idx
                for cfg_name in args.configs:
                    key = f"{set_name}|{inst.instance_id}|{seed}|{cfg_name}"
                    if key in cache:
                        continue
                    if time.time() - start > args.max_time:
                        print(f"time cap reached ({work_done} runs done)",
                              file=sys.stderr)
                        save(args.out, cache)
                        return 0
                    cfg = br_geometric_config(seed, configs_all[cfg_name])
                    cell = run_one(inst, cfg)
                    cache[key] = asdict(cell)
                    work_done += 1
                    print(f"  {set_name} inst{inst.instance_id} seed{seed} "
                          f"{cfg_name}: util={cell.util_pct:.1f}% "
                          f"t={cell.runtime:.1f}s err={cell.errors}",
                          file=sys.stderr)
                    save(args.out, cache)

    print(f"DONE  ({work_done} new runs, {len(cache)} total)",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
