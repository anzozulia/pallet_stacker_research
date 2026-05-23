"""
benchmarks.br — Bischoff-Ratcliff BR1-BR7 benchmark harness.

Reads thpack*.txt files from `benchmarks/data/`, decodes them into Box/Pallet objects,
runs the packer in pure-geometric mode (no weight/fragility/CoG/load-bearing —
the published benchmarks don't include those constraints), and reports mean
volume utilization across instances.

Format reminder (OR-Library, Bischoff-Ratcliff 1995)
---------------------------------------------------
First line: number of instances P.
Per instance:
  line: instance_id seed
  line: container L W H
  line: number of box types n
  n lines: type_idx L flag_L W flag_W H flag_H quantity
    flag_X = 1 if X-axis is allowed to be the vertical orientation,
             0 otherwise.

Published baselines (mean volume utilization across 100 instances)
-------------------------------------------------------------------
                BR1     BR3     BR5     BR7
Bischoff-Ratcliff 1995    83.1%   79.5%   76.3%   73.2%
Bortfeldt 2000            87.8%   85.6%   83.0%   80.1%
Crainic-Perboli-Tadei 08  87.9%   86.4%   84.0%   80.5%
Gonçalves-Resende 2013    92.6%   90.5%   88.7%   85.4%
Lim et al. 2013           93.0%   91.0%   89.3%   86.0%

Our metric: mean utilization of pallet #1 across the sampled instances. Items
that didn't fit on pallet #1 (overflow to pallet #2+) count as "leftover" for
the BR metric, mirroring the literature's single-container objective.
"""
from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from pallet_packer import (
    Box, Pallet, PalletPacker, PackerConfig, Rotation,
    ALL_ROTATIONS, validate,
)


# ---------------------------------------------------------------------------
# Geometric-only config (matches the published benchmark setup)
# ---------------------------------------------------------------------------
def geometric_only_config(
    *, use_brkga: bool = False, use_block_building: bool = False,
    multi_start_trials: int = 1, seed: int = 42,
    brkga_population_size: int = 16, brkga_generations: int = 4,
    max_pallets: Optional[int] = 1,
) -> PackerConfig:
    """Config with all physical constraints relaxed.

    - support_ratio = 1.0 — Bischoff–Ratcliff assumes full support (no overhang
      of one box over another).
    - enforce_load_bearing = False — boxes are weightless geometrically.
    - cog_envelope_fraction = 1.0 — disables CoG check (envelope = full pallet).
    - cog_check_min_load_fraction = 1.0 — never trigger CoG either way.
    - allow_pallet_overhang = False — BR strictly inside container.
    """
    return PackerConfig(
        support_ratio=1.0,
        require_centroid_supported=True,
        allow_pallet_overhang=False,
        heavy_on_bottom=False,                # weights are zero anyway
        cog_envelope_fraction=1.0,
        cog_check_min_load_fraction=1.0,
        enforce_load_bearing=False,
        multi_start_trials=multi_start_trials,
        seed=seed,
        use_block_building=use_block_building,
        block_threshold=4,
        use_brkga=use_brkga,
        brkga_population_size=brkga_population_size,
        brkga_generations=brkga_generations,
        brkga_n_threshold=40,
        max_pallets=max_pallets,
        optimize="max_util" if max_pallets == 1 else "min_unpacked",
    )


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
@dataclass
class BRInstance:
    set_id: str                 # e.g., "BR1"
    instance_id: int
    seed: int
    pallet: Pallet
    boxes: List[Box]

    @property
    def n_boxes(self) -> int:
        return len(self.boxes)


def _rotations_from_flags(flag_L: int, flag_W: int, flag_H: int) -> List[Rotation]:
    """BR rotation flags → our Rotation set.

    flag_X = 1 means dim X is permitted to be the vertical axis.

    Mapping: Rotation R yields (dx, dy, dz) where dz is the vertical extent.
      LWH/WLH → H vertical  → enabled iff flag_H = 1
      LHW/HLW → W vertical  → enabled iff flag_W = 1
      HWL/WHL → L vertical  → enabled iff flag_L = 1
    """
    out: List[Rotation] = []
    if flag_H:
        out += [Rotation.LWH, Rotation.WLH]
    if flag_W:
        out += [Rotation.LHW, Rotation.HLW]
    if flag_L:
        out += [Rotation.HWL, Rotation.WHL]
    # If no flag set (shouldn't happen in real data), fall back to all.
    if not out:
        out = list(ALL_ROTATIONS)
    return out


def parse_thpack(path: str, set_id: Optional[str] = None) -> List[BRInstance]:
    """Parse a thpack*.txt file into a list of BRInstance objects.

    Resilient to varied whitespace and partial files: stops cleanly at EOF.
    """
    if set_id is None:
        set_id = os.path.splitext(os.path.basename(path))[0].upper().replace("THPACK", "BR")
    with open(path) as f:
        toks = f.read().split()
    i = 0
    n_problems = int(toks[i]); i += 1
    instances: List[BRInstance] = []
    while i < len(toks) and len(instances) < n_problems:
        try:
            instance_id = int(toks[i]); i += 1
            seed = int(toks[i]); i += 1
            L = int(toks[i]); i += 1
            W = int(toks[i]); i += 1
            H = int(toks[i]); i += 1
            n_types = int(toks[i]); i += 1
        except (IndexError, ValueError):
            break  # partial file: stop cleanly
        pallet = Pallet(length=L, width=W, height=H)  # weightless
        boxes: List[Box] = []
        for _ in range(n_types):
            try:
                type_idx = int(toks[i]); i += 1
                lL = int(toks[i]); i += 1
                fL = int(toks[i]); i += 1
                lW = int(toks[i]); i += 1
                fW = int(toks[i]); i += 1
                lH = int(toks[i]); i += 1
                fH = int(toks[i]); i += 1
                qty = int(toks[i]); i += 1
            except (IndexError, ValueError):
                return instances  # truncated; bail
            rotations = _rotations_from_flags(fL, fW, fH)
            for k in range(qty):
                boxes.append(Box(
                    id=f"{set_id}.{instance_id}.T{type_idx}.{k:03d}",
                    length=lL, width=lW, height=lH,
                    weight=0.0,
                    allowed_rotations=rotations,
                ))
        instances.append(BRInstance(set_id=set_id, instance_id=instance_id,
                                    seed=seed, pallet=pallet, boxes=boxes))
    return instances


# ---------------------------------------------------------------------------
# Pallet-1 utilization metric
# ---------------------------------------------------------------------------
def first_pallet_utilization(result, pallet: Pallet) -> float:
    """% of pallet 1's volume filled. Items that overflowed to pallet 2+
    count as 'leftover' for the BR metric."""
    if not result.pallets:
        return 0.0
    p0 = result.pallets[0]
    used = sum(b.box.volume for b in p0.placements)
    cap = pallet.length * pallet.width * pallet.height
    return used / cap if cap > 0 else 0.0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
@dataclass
class RunResult:
    instance: BRInstance
    util_pct: float
    runtime: float
    overflow_boxes: int
    validator_errors: List[str]


def run_one(instance: BRInstance, config: PackerConfig) -> RunResult:
    packer = PalletPacker(instance.pallet, config)
    t0 = time.time()
    result = packer.pack(list(instance.boxes))
    elapsed = time.time() - t0
    util = first_pallet_utilization(result, instance.pallet)
    n_p0 = len(result.pallets[0].placements) if result.pallets else 0
    overflow = len(instance.boxes) - n_p0
    errs = validate(result, instance.pallet, config)
    return RunResult(
        instance=instance,
        util_pct=util * 100,
        runtime=elapsed,
        overflow_boxes=overflow,
        validator_errors=errs,
    )


@dataclass
class SetSummary:
    set_id: str
    n_instances: int
    util_mean: float
    util_stdev: float
    util_min: float
    util_max: float
    total_runtime: float
    validator_clean: int            # count of instances with zero errors


def summarize(set_id: str, runs: List[RunResult]) -> SetSummary:
    utils = [r.util_pct for r in runs]
    return SetSummary(
        set_id=set_id,
        n_instances=len(runs),
        util_mean=statistics.mean(utils) if utils else 0.0,
        util_stdev=statistics.stdev(utils) if len(utils) > 1 else 0.0,
        util_min=min(utils) if utils else 0.0,
        util_max=max(utils) if utils else 0.0,
        total_runtime=sum(r.runtime for r in runs),
        validator_clean=sum(1 for r in runs if not r.validator_errors),
    )


def run_set(
    path: str,
    *,
    sample: Optional[int] = None,
    use_v2: bool = False,
    set_id: Optional[str] = None,
) -> Tuple[SetSummary, List[RunResult]]:
    instances = parse_thpack(path, set_id=set_id)
    if sample:
        instances = instances[:sample]
    cfg = geometric_only_config(
        use_brkga=use_v2, use_block_building=use_v2,
        multi_start_trials=1, seed=42,
    )
    runs: List[RunResult] = []
    for idx, inst in enumerate(instances, 1):
        print(f"  [{idx:>2}/{len(instances)}] {inst.set_id} #{inst.instance_id} "
              f"(N={inst.n_boxes})…", file=sys.stderr, end="", flush=True)
        r = run_one(inst, cfg)
        print(f"  util={r.util_pct:.1f}%  t={r.runtime:.2f}s", file=sys.stderr)
        runs.append(r)
    return summarize(instances[0].set_id if instances else "?", runs), runs


# ---------------------------------------------------------------------------
# Published baselines for the report (mean volume util %, 100 instances each)
# ---------------------------------------------------------------------------
PUBLISHED_BASELINES = {
    # set: (Bischoff-Ratcliff 1995, Bortfeldt 2000, CPT 2008, BRKGA G&R 2013, Lim 2013)
    "BR1": (83.1, 87.8, 87.9, 92.6, 93.0),
    "BR3": (79.5, 85.6, 86.4, 90.5, 91.0),
    "BR5": (76.3, 83.0, 84.0, 88.7, 89.3),
    "BR7": (73.2, 80.1, 80.5, 85.4, 86.0),
}
BASELINE_LABELS = ("BR-1995", "Bortfeldt-2000", "CPT-2008", "BRKGA-2013", "Lim-2013")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def write_report(
    v1_sums: List[SetSummary],
    v2_sums: List[SetSummary],
    sample: int,
    path: str = "BR_REPORT.md",
) -> None:
    lines: List[str] = []
    ap = lines.append
    ap("# Phase 1 — Bischoff-Ratcliff Benchmark Report\n\n")
    ap(f"Sampled {sample} instances per set from the OR-Library thpack files. "
       "Pure-geometric mode: no weight, fragility, CoG, or load-bearing — "
       "matching the published benchmark setup. Metric is mean volume "
       "utilization on pallet #1 across the sample.\n\n")
    ap("## Headline numbers\n\n")
    ap("| Set | N inst. | N boxes (mean) | v1 util mean ± stdev | v2 util mean ± stdev | "
       "BR-1995 | Bortfeldt-2000 | CPT-2008 | BRKGA-2013 | Lim-2013 |\n")
    ap("|---|---|---|---|---|---|---|---|---|---|\n")
    for v1, v2 in zip(v1_sums, v2_sums):
        baselines = PUBLISHED_BASELINES.get(v1.set_id, (0, 0, 0, 0, 0))
        ap(f"| {v1.set_id} | {v1.n_instances} | — | "
           f"{v1.util_mean:.1f}% ± {v1.util_stdev:.1f} | "
           f"{v2.util_mean:.1f}% ± {v2.util_stdev:.1f} | "
           f"{baselines[0]:.1f}% | {baselines[1]:.1f}% | "
           f"{baselines[2]:.1f}% | {baselines[3]:.1f}% | "
           f"{baselines[4]:.1f}% |\n")
    ap("\n## Where we stand\n\n")
    for v1, v2 in zip(v1_sums, v2_sums):
        b = PUBLISHED_BASELINES.get(v1.set_id, (0, 0, 0, 0, 0))
        ap(f"- **{v1.set_id}** — v1 at {v1.util_mean:.1f}%, v2 at "
           f"{v2.util_mean:.1f}%. "
           f"Gap to Bortfeldt-2000 baseline ({b[1]:.1f}%): "
           f"{v2.util_mean - b[1]:+.1f} pp. "
           f"Gap to BRKGA-2013 ({b[3]:.1f}%): "
           f"{v2.util_mean - b[3]:+.1f} pp.\n")
    ap("\n## Runtime\n\n")
    ap("| Set | v1 total | v2 total | v1 per-instance | v2 per-instance |\n")
    ap("|---|---|---|---|---|\n")
    for v1, v2 in zip(v1_sums, v2_sums):
        v1_per = v1.total_runtime / v1.n_instances if v1.n_instances else 0
        v2_per = v2.total_runtime / v2.n_instances if v2.n_instances else 0
        ap(f"| {v1.set_id} | {v1.total_runtime:.1f}s | "
           f"{v2.total_runtime:.1f}s | {v1_per:.2f}s | {v2_per:.2f}s |\n")
    ap("\n## Validator correctness\n\n")
    ap("| Set | v1 clean / total | v2 clean / total |\n")
    ap("|---|---|---|\n")
    for v1, v2 in zip(v1_sums, v2_sums):
        ap(f"| {v1.set_id} | {v1.validator_clean} / {v1.n_instances} | "
           f"{v2.validator_clean} / {v2.n_instances} |\n")
    with open(path, "w") as f:
        f.write("".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="benchmarks/data",
                   help="Directory containing thpack*.txt files.")
    p.add_argument("--sample", type=int, default=10,
                   help="Number of instances per set to run.")
    p.add_argument("--sets", nargs="+", default=["thpack1", "thpack3", "thpack5"],
                   help="Which thpack files (basename, no extension) to run.")
    p.add_argument("--out", default="docs/reports/04_br_baseline.md")
    p.add_argument("--skip-v2", action="store_true",
                   help="Skip v2 run (saves runtime; v2 is slower).")
    args = p.parse_args(argv)

    v1_sums: List[SetSummary] = []
    v2_sums: List[SetSummary] = []
    for set_name in args.sets:
        path = os.path.join(args.data_dir, set_name + ".txt")
        if not os.path.exists(path):
            print(f"SKIP {set_name}: not found at {path}", file=sys.stderr)
            continue
        print(f"\n=== {set_name} — v1 ===", file=sys.stderr)
        v1_sum, _v1_runs = run_set(path, sample=args.sample, use_v2=False)
        v1_sums.append(v1_sum)
        if not args.skip_v2:
            print(f"\n=== {set_name} — v2 (block + BRKGA) ===", file=sys.stderr)
            v2_sum, _v2_runs = run_set(path, sample=args.sample, use_v2=True)
            v2_sums.append(v2_sum)
        else:
            v2_sums.append(v1_sum)  # placeholder

    print("\n=== Summary ===")
    for v1, v2 in zip(v1_sums, v2_sums):
        print(f"  {v1.set_id}: v1 {v1.util_mean:5.1f}% ± {v1.util_stdev:4.1f}  "
              f"| v2 {v2.util_mean:5.1f}% ± {v2.util_stdev:4.1f}  "
              f"({v1.n_instances} instances)")

    write_report(v1_sums, v2_sums, args.sample, path=args.out)
    print(f"\nWrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
