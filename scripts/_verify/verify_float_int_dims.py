"""verify_float_int_dims.py — adversarial probe for the INT-decode / FLOAT-validate gap.

THE RISK: decoders work in INTEGER coords (`int(round())` on pallet dims;
int64 `dims_all` for boxes), but `Placement` stores the ORIGINAL float `Box`
and the validator (`validate`) recomputes geometry from those float dims.
A box with width 250.4 is decoded as dim 250; the next box is butted against
it at integer x=250; but in float the first box ends at x2=250.4 → a 0.4 unit
overlap the decoder never saw. Symmetric story for boxes that round UP
(spurious gaps / lost density) and for the pallet ceiling (round-up can let a
stack poke through the float pallet height).

This probe:
  * builds matched FLOAT vs ROUNDED-INTEGER workloads,
  * solves each with brkga_pack_v35,
  * runs the ground-truth validate(),
  * ALSO runs an INDEPENDENT float-geometry checker (overlap + bounds, using a
    tight numeric tolerance, not the 1e-6 validator EPS) to see *how big* the
    float-overlap actually is,
  * quantifies validator-error RATE float vs int, and the worst offender.

Run:
  docker run --rm -e OMP_NUM_THREADS=1 -e PYTHONHASHSEED=0 \
    -v "$(pwd)":/app -w /app pallet-packer:dev \
    python scripts/_verify/verify_float_int_dims.py
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import math
import random
from typing import List, Tuple

from pallet_packer import (
    Box, Pallet, PackerConfig, validate,
    ALL_ROTATIONS,
)
from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit


# ---------------------------------------------------------------------------
# Independent float-geometry checker (does NOT trust the validator's EPS).
# Returns the worst overlap depth and worst out-of-bounds excess found.
# ---------------------------------------------------------------------------
def float_geometry_audit(result, pallet) -> dict:
    worst_overlap = 0.0
    worst_overlap_pair = None
    worst_oob = 0.0
    worst_oob_box = None
    n_overlap_pairs = 0
    n_oob = 0
    for st in result.pallets:
        pls = st.placements
        # bounds (no overhang assumed — fractional bounds vs float pallet dims)
        for p in pls:
            x2 = p.x + p.dx
            y2 = p.y + p.dy
            z2 = p.z + p.dz
            excess = max(
                -min(p.x, p.y, p.z),
                x2 - pallet.length,
                y2 - pallet.width,
                z2 - pallet.height,
            )
            if excess > 1e-9:
                n_oob += 1
                if excess > worst_oob:
                    worst_oob = excess
                    worst_oob_box = (st.pallet_id, p.box.id, p.x, p.y, p.z,
                                     p.dx, p.dy, p.dz)
        # pairwise overlap depth (true float, min penetration across 3 axes)
        for i in range(len(pls)):
            a = pls[i]
            ax2, ay2, az2 = a.x + a.dx, a.y + a.dy, a.z + a.dz
            for j in range(i + 1, len(pls)):
                b = pls[j]
                bx2, by2, bz2 = b.x + b.dx, b.y + b.dy, b.z + b.dz
                ox = min(ax2, bx2) - max(a.x, b.x)
                oy = min(ay2, by2) - max(a.y, b.y)
                oz = min(az2, bz2) - max(a.z, b.z)
                if ox > 1e-9 and oy > 1e-9 and oz > 1e-9:
                    pen = min(ox, oy, oz)  # min penetration = overlap depth
                    n_overlap_pairs += 1
                    if pen > worst_overlap:
                        worst_overlap = pen
                        worst_overlap_pair = (st.pallet_id, a.box.id, b.box.id,
                                              ox, oy, oz)
    return dict(
        worst_overlap=worst_overlap, worst_overlap_pair=worst_overlap_pair,
        n_overlap_pairs=n_overlap_pairs,
        worst_oob=worst_oob, worst_oob_box=worst_oob_box, n_oob=n_oob,
    )


def geom_cfg(max_pallets=1) -> PackerConfig:
    """Pure-geometric config (no constraints -> has_constraints=False)."""
    return PackerConfig(
        support_ratio=0.0,            # don't reject for support; isolate the dim issue
        require_centroid_supported=False,
        enforce_load_bearing=False,
        allow_pallet_overhang=False,
        cog_envelope_fraction=1.0,    # CoG inactive
        max_pallets=max_pallets,
        optimize="max_util",
    )


def solve(boxes, pallet, cfg, seed=42, t=4.0, max_pallets=1):
    return brkga_pack_v35(
        boxes, pallet, cfg,
        time_limit_s=t, max_pallets=max_pallets,
        population_size=200, n_populations=3, patience=120,
        local_search_budget_s=1.0, seed=seed, verbose=False, n_modes=6,
    )


def round_box(b: Box) -> Box:
    return Box(id=b.id, length=round(b.length), width=round(b.width),
               height=round(b.height), weight=b.weight,
               max_load_on_top=b.max_load_on_top,
               allowed_rotations=list(b.allowed_rotations),
               group=b.group, requires_full_support=b.requires_full_support)


def round_pallet(p: Pallet) -> Pallet:
    return Pallet(length=round(p.length), width=round(p.width),
                  height=round(p.height), max_weight=p.max_weight,
                  max_overhang=p.max_overhang)


# ---------------------------------------------------------------------------
# Sub-test runner: solve, validate, float-audit, print, accumulate.
# ---------------------------------------------------------------------------
class Acc:
    def __init__(self):
        self.float_runs = 0
        self.float_runs_with_errs = 0
        self.int_runs = 0
        self.int_runs_with_errs = 0
        self.worst = None  # (worst_overlap_or_oob, label, details)

    def note_worst(self, val, label, details):
        if self.worst is None or val > self.worst[0]:
            self.worst = (val, label, details)


def run_case(acc: Acc, label, boxes, pallet, cfg, is_float, seed=42,
             t=4.0, max_pallets=1):
    r = solve(boxes, pallet, cfg, seed=seed, t=t, max_pallets=max_pallets)
    errs = validate(r, pallet, cfg)
    audit = float_geometry_audit(r, pallet)
    n_placed = sum(len(st.placements) for st in r.pallets)
    if is_float:
        acc.float_runs += 1
        if errs:
            acc.float_runs_with_errs += 1
    else:
        acc.int_runs += 1
        if errs:
            acc.int_runs_with_errs += 1
    tag = "FLOAT" if is_float else "INT  "
    status = "FAIL" if errs else "ok"
    print(f"  [{tag}] {label:<34} placed={n_placed:>3} "
          f"val_errs={len(errs):>2} worst_overlap={audit['worst_overlap']:.4f} "
          f"worst_oob={audit['worst_oob']:.4f}  [{status}]")
    if errs:
        ex = errs[0]
        print(f"           e.g. {ex}")
    if audit['worst_overlap'] > 1e-6:
        acc.note_worst(audit['worst_overlap'], f"{label} (overlap)",
                       audit['worst_overlap_pair'])
    if audit['worst_oob'] > 1e-6:
        acc.note_worst(audit['worst_oob'], f"{label} (oob)",
                       audit['worst_oob_box'])
    return r, errs, audit


def main() -> int:
    warmup_jit()
    acc = Acc()
    cfg = geom_cfg(max_pallets=1)
    rng = random.Random(12345)

    print("=" * 78)
    print("FLOAT-INT DIMS PROBE: int-decode vs float-validate")
    print("=" * 78)

    # ---------------------------------------------------------------
    # TEST 1: hand-crafted ADVERSARIAL pack guaranteed to butt boxes that
    # round DOWN against each other. A row of boxes with dim ending in .49
    # rounds down; the decoder packs them at integer pitch (rounded-down),
    # so consecutive boxes overlap by ~0.49 each in float.
    # ---------------------------------------------------------------
    print("\n=== TEST 1: round-DOWN row (forces float overlap) ===")
    # 20 identical boxes 100.49 x 100.49 x 100, in a pallet that fits exactly
    # 10x10 if decoded as 100. Pallet 1004 x 1004 x 100.
    fb = [Box(id=f"d{i}", length=100.49, width=100.49, height=100.0,
              weight=0.0, allowed_rotations=list(ALL_ROTATIONS))
          for i in range(64)]
    fp = Pallet(length=804.0, width=804.0, height=100.0)  # decodes to 8x8=64
    run_case(acc, "100.49^2 x100 (rounds down)", fb, fp, cfg, is_float=True)
    run_case(acc, "100.49^2 x100 -> rounded", [round_box(b) for b in fb],
             round_pallet(fp), cfg, is_float=False)

    # ---------------------------------------------------------------
    # TEST 2: round-UP row (forces out-of-bounds / lost density). dim .51
    # rounds up; decoder thinks each box is 101 wide; if pallet decodes to a
    # tight multiple, boxes can be pushed so the LAST one's float extent
    # exceeds the float pallet bound — or density is silently lost.
    # ---------------------------------------------------------------
    print("\n=== TEST 2: round-UP row (gaps / OOB) ===")
    fb2 = [Box(id=f"u{i}", length=99.51, width=99.51, height=100.0,
               weight=0.0, allowed_rotations=list(ALL_ROTATIONS))
           for i in range(100)]
    fp2 = Pallet(length=1000.0, width=1000.0, height=100.0)
    run_case(acc, "99.51^2 x100 (rounds up)", fb2, fp2, cfg, is_float=True)
    run_case(acc, "99.51^2 x100 -> rounded", [round_box(b) for b in fb2],
             round_pallet(fp2), cfg, is_float=False)

    # ---------------------------------------------------------------
    # TEST 3: pallet HEIGHT round-down → tall stack pokes through ceiling.
    # box height 33.49 -> decodes as 33. Pallet height 100.4 -> decodes 100.
    # decoder stacks 3 (99<100); float stack = 3*33.49 = 100.47 > 100.4.
    # ---------------------------------------------------------------
    print("\n=== TEST 3: height round-down (ceiling breach) ===")
    fb3 = [Box(id=f"h{i}", length=100.0, width=100.0, height=33.49,
               weight=0.0, allowed_rotations=[ALL_ROTATIONS[0]])
           for i in range(30)]
    fp3 = Pallet(length=300.0, width=300.0, height=100.0)
    run_case(acc, "h=33.49 stack, ceil=100", fb3, fp3, cfg, is_float=True)
    run_case(acc, "h=33.49 stack -> rounded", [round_box(b) for b in fb3],
             round_pallet(fp3), cfg, is_float=False)

    # ---------------------------------------------------------------
    # TEST 4: realistic fractional dims (mandate examples) — random mixes.
    # 333.33 x 250.7 x 124.9 etc. Many SKUs, many runs to estimate a RATE.
    # ---------------------------------------------------------------
    print("\n=== TEST 4: realistic fractional SKUs (rate over many runs) ===")
    realistic_dims = [
        (333.33, 250.7, 124.9), (199.95, 150.45, 99.99),
        (412.6, 318.2, 205.1), (88.8, 88.8, 88.8),
        (1000.5, 600.25, 400.75), (55.55, 44.44, 33.33),
    ]
    for di, (L, W, H) in enumerate(realistic_dims):
        n = rng.randint(20, 50)
        fb4 = [Box(id=f"r{di}_{i}", length=L, width=W, height=H,
                   weight=0.0, allowed_rotations=list(ALL_ROTATIONS))
               for i in range(n)]
        fp4 = Pallet(length=1200.0, width=1000.0, height=1500.0)
        run_case(acc, f"{L}x{W}x{H} n={n}", fb4, fp4, cfg, is_float=True,
                 seed=100 + di, t=3.0)
        run_case(acc, f"{L}x{W}x{H} n={n} -> rnd",
                 [round_box(b) for b in fb4], fp4, cfg, is_float=False,
                 seed=100 + di, t=3.0)

    # ---------------------------------------------------------------
    # TEST 5: MANY decimals + heterogeneous random fractional mix.
    # Stress the rate estimate with fully random fractional everything.
    # ---------------------------------------------------------------
    print("\n=== TEST 5: random heterogeneous fractional mixes ===")
    for trial in range(8):
        n = rng.randint(25, 45)
        fb5 = []
        for i in range(n):
            L = rng.uniform(60, 400) + rng.random()  # arbitrary decimals
            W = rng.uniform(60, 400) + rng.random()
            H = rng.uniform(60, 300) + rng.random()
            fb5.append(Box(id=f"x{trial}_{i}", length=round(L, 4),
                           width=round(W, 4), height=round(H, 4),
                           weight=0.0, allowed_rotations=list(ALL_ROTATIONS)))
        fp5 = Pallet(length=1200.0 + rng.random(),
                     width=1000.0 + rng.random(),
                     height=1500.0 + rng.random())
        run_case(acc, f"rand mix #{trial} n={n}", fb5, fp5, cfg,
                 is_float=True, seed=200 + trial, t=3.0)

    # ---------------------------------------------------------------
    # TEST 6: SUB-MILLIMETRE boxes — dims < 1 round to 0 (degenerate!).
    # A box 0.4 x 0.4 x 0.4 decodes to dim 0. Catastrophic: zero-volume in
    # the decoder, real volume in float. Also dims 0.6 -> 1.
    # ---------------------------------------------------------------
    print("\n=== TEST 6: sub-millimetre boxes (round-to-zero) ===")
    fb6 = [Box(id=f"s{i}", length=0.4, width=0.4, height=0.4, weight=0.0,
               allowed_rotations=list(ALL_ROTATIONS)) for i in range(50)]
    fp6 = Pallet(length=5.0, width=5.0, height=5.0)
    run_case(acc, "0.4^3 sub-mm (rounds to 0)", fb6, fp6, cfg, is_float=True,
             t=3.0)
    fb6b = [Box(id=f"s{i}", length=0.6, width=0.6, height=0.6, weight=0.0,
                allowed_rotations=list(ALL_ROTATIONS)) for i in range(50)]
    run_case(acc, "0.6^3 sub-mm (rounds to 1)", fb6b, fp6, cfg, is_float=True,
             t=3.0)

    # ---------------------------------------------------------------
    # TEST 7: VERY LARGE dims (1e6+) — float precision of int(round) and
    # int64 capacity. Also check the decoder doesn't lose precision.
    # ---------------------------------------------------------------
    print("\n=== TEST 7: very large dims (1e6+) ===")
    fb7 = [Box(id=f"big{i}", length=333333.7, width=250000.3, height=124999.9,
               weight=0.0, allowed_rotations=list(ALL_ROTATIONS))
           for i in range(20)]
    fp7 = Pallet(length=1200000.5, width=1000000.5, height=1500000.5)
    run_case(acc, "1e6-scale fractional", fb7, fp7, cfg, is_float=True, t=3.0)
    run_case(acc, "1e6-scale -> rounded", [round_box(b) for b in fb7],
             round_pallet(fp7), cfg, is_float=False, t=3.0)

    # ---------------------------------------------------------------
    # TEST 8: tight exact-fit fractional — pallet whose float dims are an
    # EXACT multiple of a fractional box, but rounding breaks the tiling.
    # box 120.5 wide, pallet 1205.0 wide = exactly 10. But 120.5 -> 120 (round
    # half to even) or 121; pallet 1205 -> 1205. Tiling at int pitch 120/121
    # vs 1205 -> either gap or overlap accumulation.
    # ---------------------------------------------------------------
    print("\n=== TEST 8: exact-multiple fractional tiling ===")
    fb8 = [Box(id=f"t{i}", length=120.5, width=120.5, height=100.0,
               weight=0.0, allowed_rotations=list(ALL_ROTATIONS))
           for i in range(100)]
    fp8 = Pallet(length=1205.0, width=1205.0, height=100.0)
    run_case(acc, "120.5 tile in 1205", fb8, fp8, cfg, is_float=True, t=3.0)

    # ---------------------------------------------------------------
    # Summary + rates
    # ---------------------------------------------------------------
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    fr = acc.float_runs
    ir = acc.int_runs
    fre = acc.float_runs_with_errs
    ire = acc.int_runs_with_errs
    f_rate = (100.0 * fre / fr) if fr else 0.0
    i_rate = (100.0 * ire / ir) if ir else 0.0
    print(f"FLOAT-input runs:   {fr:>3}  with validator errors: {fre:>3}  "
          f"rate = {f_rate:.1f}%")
    print(f"INTEGER-input runs: {ir:>3}  with validator errors: {ire:>3}  "
          f"rate = {i_rate:.1f}%")
    if acc.worst is not None:
        val, label, details = acc.worst
        print(f"\nWORST float-geometry defect: {val:.4f} units "
              f"({label})")
        print(f"  details: {details}")
    else:
        print("\nNo float-geometry defects detected.")

    # Verdict: PASS only if FLOAT inputs never produce validator errors AND
    # never produce a float-overlap/oob beyond a benign rounding tolerance.
    benign = 1e-3  # anything above this is a genuine geometric defect
    worst_val = acc.worst[0] if acc.worst else 0.0
    if fre == 0 and worst_val <= benign:
        print("\n=== RESULT: PASS (no float defects found) ===")
        return 0
    print(f"\n=== RESULT: FAIL "
          f"(float err-rate {f_rate:.1f}% vs int {i_rate:.1f}%, "
          f"worst defect {worst_val:.4f}) ===")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
