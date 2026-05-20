"""Compare v1 (baseline) vs v2 (block-building + BRKGA) across all benchmark
and failure cases. Side-by-side, single report.
"""
from __future__ import annotations
import time
import argparse
import math
from typing import List, Tuple, Any

from pallet_packer import (
    Pallet, PackerConfig, PalletPacker, validate, PackResult, Box,
)
from benchmark import cases as benchmark_cases
from failure_cases import (
    case_F1_pareto_continuum, case_F2_block_of_identicals,
    case_F3_interlock_pattern, case_F4_height_mismatch,
    case_F5_must_rotate_early, case_F6_constraint_cascade,
    case_F7_layered_pyramid, case_F8_strongly_heterogeneous,
    case_F9_long_unrotatable, case_F10_overhang_required,
    case_F11_spurious_unpacked, case_F12_strongly_heterogeneous_at_scale,
    case_F13_layer_advantage,
)


def _failure_cases():
    return [
        case_F1_pareto_continuum(), case_F2_block_of_identicals(),
        case_F3_interlock_pattern(), case_F4_height_mismatch(),
        case_F5_must_rotate_early(), case_F6_constraint_cascade(),
        case_F7_layered_pyramid(), case_F8_strongly_heterogeneous(),
        case_F9_long_unrotatable(), case_F10_overhang_required(),
        case_F11_spurious_unpacked(), case_F12_strongly_heterogeneous_at_scale(),
        case_F13_layer_advantage(),
    ]


def _clone_box(b: Box) -> Box:
    return Box(
        id=b.id, length=b.length, width=b.width, height=b.height,
        weight=b.weight, max_load_on_top=b.max_load_on_top,
        allowed_rotations=list(b.allowed_rotations), group=b.group,
        requires_full_support=b.requires_full_support,
    )


def _make_boxes(case: Any) -> List[Box]:
    if hasattr(case, "boxes_factory"):
        return case.boxes_factory()
    return [_clone_box(b) for b in case.boxes]


def _volume_lb(boxes: List[Box], pallet: Pallet) -> int:
    total = sum(b.volume for b in boxes)
    pv = pallet.length * pallet.width * pallet.height
    return max(1, math.ceil(total / pv)) if boxes else 0


def _run(case: Any, kwargs: dict, brkga_pop: int, brkga_gen: int):
    cfg = PackerConfig(seed=42, **kwargs)
    src = case.config
    for attr in ("support_ratio", "cog_envelope_fraction", "multi_start_trials",
                 "heavy_on_bottom", "enforce_load_bearing",
                 "require_centroid_supported", "allow_pallet_overhang",
                 "cog_active_load_fraction"):
        if hasattr(src, attr):
            setattr(cfg, attr, getattr(src, attr))
    if kwargs.get("use_brkga"):
        cfg.brkga_population_size = brkga_pop
        cfg.brkga_generations = brkga_gen
    t0 = time.time()
    r = PalletPacker(case.pallet, cfg).pack(_make_boxes(case))
    elapsed = time.time() - t0
    errs = validate(r, case.pallet, cfg)
    return r, elapsed, len(errs)


def run_all(label: str, cases: List[Any], brkga_pop: int, brkga_gen: int):
    print(f"\n{'='*100}")
    print(f"{label}  (BRKGA pop={brkga_pop}, gens={brkga_gen})")
    print(f"{'='*100}")
    header = (f"{'case':<40} | {'LB':>3} | "
              f"{'v1':>4} {'util':>5} {'  t':>6} | "
              f"{'v2':>4} {'util':>5} {'  t':>6} | "
              f"{'Δp':>3} {'Δu':>6}")
    print(header)
    print("-" * len(header))

    tot = {"v1p": 0, "v2p": 0, "v1u": 0.0, "v2u": 0.0,
           "v1t": 0.0, "v2t": 0.0, "imp": 0, "reg": 0, "v1e": 0, "v2e": 0}
    for c in cases:
        r1, t1, e1 = _run(c, dict(), brkga_pop, brkga_gen)
        r2, t2, e2 = _run(c, dict(use_block_building=True, use_brkga=True),
                          brkga_pop, brkga_gen)
        lb = _volume_lb(_make_boxes(c), c.pallet)
        if hasattr(c, "expected_pallets") and c.expected_pallets:
            lb = max(lb, c.expected_pallets)

        dp = r2.num_pallets - r1.num_pallets
        du = r2.total_volume_utilisation - r1.total_volume_utilisation
        flag = ""
        if dp < 0:
            flag = "[OK] better"; tot["imp"] += 1
        elif dp > 0:
            flag = "[X]  worse"; tot["reg"] += 1
        elif du > 0.005:
            flag = "+util"

        tot["v1p"] += r1.num_pallets; tot["v2p"] += r2.num_pallets
        tot["v1u"] += r1.total_volume_utilisation
        tot["v2u"] += r2.total_volume_utilisation
        tot["v1t"] += t1; tot["v2t"] += t2
        tot["v1e"] += 1 if e1 else 0; tot["v2e"] += 1 if e2 else 0

        name = c.name[:38]
        print(f"{name:<40} | {lb:>3} | "
              f"{r1.num_pallets:>3}p {r1.total_volume_utilisation:>4.0%} {t1:>5.2f}s | "
              f"{r2.num_pallets:>3}p {r2.total_volume_utilisation:>4.0%} {t2:>5.2f}s | "
              f"{dp:>+3d} {du:>+6.1%} {flag}", flush=True)

    n = len(cases)
    print("-" * len(header))
    print(f"{'totals (' + str(n) + ' cases)':<40} | {'':3} | "
          f"{tot['v1p']:>3}   {tot['v1u']/n:>4.0%} {tot['v1t']:>5.1f}s | "
          f"{tot['v2p']:>3}   {tot['v2u']/n:>4.0%} {tot['v2t']:>5.1f}s | "
          f"{tot['v2p']-tot['v1p']:>+3d}")
    print(f"  v2 vs v1: {tot['imp']} improvements, {tot['reg']} regressions, "
          f"{n - tot['imp'] - tot['reg']} unchanged")
    if tot["v1e"] or tot["v2e"]:
        print(f"  validator errors: v1={tot['v1e']}, v2={tot['v2e']}")
    return tot


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pop", type=int, default=16)
    parser.add_argument("--gen", type=int, default=4)
    parser.add_argument("--suite", choices=["benchmark", "failure", "both"],
                        default="both")
    args = parser.parse_args()

    overall_t0 = time.time()
    if args.suite in ("benchmark", "both"):
        run_all("BENCHMARK SUITE", benchmark_cases(), args.pop, args.gen)
    if args.suite in ("failure", "both"):
        run_all("FAILURE-MODE SUITE", _failure_cases(), args.pop, args.gen)
    print(f"\nTotal runtime: {time.time()-overall_t0:.1f}s")
