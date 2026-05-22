"""
run_lb_diagnostic.py — Support-aware LB diagnostic.

Runs MIP polish with max_pallets = LB on each open-headroom case and a
long time budget. If MIP returns INFEASIBLE (no feasible solution exists
at that pallet count), then v1's pallet count is the true optimum under
support_ratio=0.8, and our reported "open" status was a false positive
from a loose LB.

If MIP returns FEASIBLE / OPTIMAL → there IS a tighter packing we
haven't found. The gap is real.

If MIP returns UNKNOWN (timed out without proving infeasibility OR
finding a solution) → inconclusive but suggests the LB might be tight.
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
from typing import Dict, List, Optional

from pallet_packer import PalletPacker, PackerConfig, validate
from benchmarks import internal as benchmark
from benchmarks import failure_cases

try:
    from pallet_packer.mip import mip_polish as _mip_polish
    _CP_SAT_AVAILABLE = True
except ImportError:
    _CP_SAT_AVAILABLE = False


def diagnose(case_name: str, case_cfg: PackerConfig, boxes, pallet,
             target_pallets: int, time_limit_s: float = 60.0) -> Dict:
    """Run MIP with max_pallets = target_pallets. Return result summary."""
    cfg = PackerConfig(
        support_ratio=case_cfg.support_ratio,
        require_centroid_supported=case_cfg.require_centroid_supported,
        allow_pallet_overhang=case_cfg.allow_pallet_overhang,
        heavy_on_bottom=case_cfg.heavy_on_bottom,
        cog_envelope_fraction=case_cfg.cog_envelope_fraction,
        cog_check_min_load_fraction=case_cfg.cog_check_min_load_fraction,
        enforce_load_bearing=case_cfg.enforce_load_bearing,
        multi_start_trials=1,
        seed=42,
    )
    # Get a v1 warm-start.
    packer = PalletPacker(pallet, cfg)
    t0 = time.time()
    warm = packer.pack(list(boxes))
    warm_t = time.time() - t0
    print(f"  v1 warm-start: {warm.num_pallets}p/{len(warm.unpacked)}u "
          f"util={warm.total_volume_utilisation*100:.1f}% t={warm_t:.1f}s",
          file=sys.stderr)
    if warm.num_pallets <= target_pallets and not warm.unpacked:
        return {"verdict": "v1 already at target", "warm": warm.num_pallets,
                "target": target_pallets}

    if not _CP_SAT_AVAILABLE:
        return {"verdict": "CP-SAT unavailable", "warm": warm.num_pallets}

    print(f"  Running MIP with max_pallets={target_pallets}, "
          f"time={time_limit_s}s…", file=sys.stderr)
    t0 = time.time()
    mip_result = _mip_polish(
        boxes, pallet, cfg,
        time_limit_s=time_limit_s,
        num_workers=1,
        max_pallets=target_pallets,
        warm_start=warm,
    )
    mip_t = time.time() - t0

    last_status = getattr(_mip_polish, "last_status", "?")
    if mip_result is None:
        verdict = last_status
        return {"verdict": verdict, "warm": warm.num_pallets,
                "target": target_pallets, "mip_runtime_s": mip_t}

    errs = validate(mip_result, pallet, cfg)
    return {
        "verdict": last_status,
        "warm": warm.num_pallets,
        "target": target_pallets,
        "mip_pallets": mip_result.num_pallets,
        "mip_unpacked": len(mip_result.unpacked),
        "mip_util_pct": mip_result.total_volume_utilisation * 100,
        "mip_validator_errors": len(errs),
        "mip_runtime_s": mip_t,
    }


def main(argv: Optional[List[str]] = None) -> int:
    argv = argv or []
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True,
                    help="Case name prefix (e.g. C1, C2, C3)")
    ap.add_argument("--target", type=int, default=None,
                    help="Target pallet count. Default = v1 - 1.")
    ap.add_argument("--time-limit", type=float, default=60.0)
    args = ap.parse_args(argv)

    # Find the case.
    found = None
    for c in benchmark.cases():
        if c.name.startswith(args.case):
            found = ("bench", c.name, c.boxes, c.pallet, c.config)
            break
    if found is None:
        for factory in [
            failure_cases.case_F1_pareto_continuum,
            failure_cases.case_F2_block_of_identicals,
            failure_cases.case_F3_interlock_pattern,
            failure_cases.case_F12_strongly_heterogeneous_at_scale,
        ]:
            c = factory()
            if c.name.startswith(args.case):
                found = ("fail", c.name, c.boxes_factory(), c.pallet, c.config)
                break
    if found is None:
        print(f"Case {args.case} not found", file=sys.stderr)
        return 1

    suite, name, boxes, pallet, case_cfg = found
    print(f"Diagnosing {suite}|{name} (N={len(boxes)})", file=sys.stderr)
    # Default target = v1 - 1.
    if args.target is None:
        packer = PalletPacker(pallet, PackerConfig(seed=42, multi_start_trials=1))
        v1 = packer.pack(list(boxes))
        args.target = max(1, v1.num_pallets - 1)
        print(f"  v1 = {v1.num_pallets} pallets; target = {args.target}",
              file=sys.stderr)

    result = diagnose(name, case_cfg, boxes, pallet, args.target,
                      time_limit_s=args.time_limit)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
