"""
benchmark.py - Comprehensive test suite for pallet_packer.py.

Runs ~30 cases covering:
  A. Trivial / edge cases
  B. Known-optimal homogeneous packings
  C. Heterogeneous instances at varying scale
  D. Constraint-sensitivity sweep (same boxes, each constraint toggled)
  E. Pathological instances
  F. Real-world flavour profiles
  G. Multi-start ablation

For each case: input description, expected (where computable), measured
output (pallets, utilisation, runtime, validator errors), and PASS/FAIL.
"""
from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from pallet_packer import (
    Box, Pallet, PalletPacker, PackerConfig,
    ALL_ROTATIONS, THIS_SIDE_UP, NO_ROTATION,
    PackResult, validate,
)


# ---------------------------------------------------------------------------
# Benchmark infrastructure
# ---------------------------------------------------------------------------
@dataclass
class Case:
    name: str
    boxes: List[Box]
    pallet: Pallet
    config: PackerConfig = field(default_factory=PackerConfig)
    # Optional expected pallet count for PASS/FAIL.
    expected_pallets: Optional[int] = None
    # Optional expected number of unpacked items.
    expected_unpacked: Optional[int] = None
    # Free-text notes shown in the report.
    notes: str = ""


@dataclass
class Outcome:
    case: Case
    result: PackResult
    runtime: float
    errors: List[str]

    @property
    def passed(self) -> bool:
        if self.errors:
            return False
        if self.case.expected_pallets is not None:
            if self.result.num_pallets != self.case.expected_pallets:
                return False
        if self.case.expected_unpacked is not None:
            if len(self.result.unpacked) != self.case.expected_unpacked:
                return False
        return True

    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"


def run(case: Case) -> Outcome:
    packer = PalletPacker(case.pallet, case.config)
    t0 = time.time()
    result = packer.pack(case.boxes)
    elapsed = time.time() - t0
    errs = validate(result, case.pallet, case.config)
    return Outcome(case=case, result=result, runtime=elapsed, errors=errs)


# ---------------------------------------------------------------------------
# Box-set generators
# ---------------------------------------------------------------------------
def uniform_boxes(n, L, W, H, weight=1.0, prefix="X", **kw) -> List[Box]:
    return [Box(id=f"{prefix}-{i:03d}", length=L, width=W, height=H,
                weight=weight, **kw) for i in range(n)]


def mixed_classic() -> List[Box]:
    """The user's headline scenario."""
    out: List[Box] = []
    out += [Box(id=f"A-{i:02d}", length=30, width=40, height=70, weight=5,
                allowed_rotations=THIS_SIDE_UP, max_load_on_top=50)
            for i in range(20)]
    out += [Box(id=f"B-{i:02d}", length=40, width=45, height=50, weight=8,
                allowed_rotations=ALL_ROTATIONS, max_load_on_top=80)
            for i in range(15)]
    out += [Box(id=f"C-{i:02d}", length=60, width=60, height=30, weight=12,
                allowed_rotations=THIS_SIDE_UP, max_load_on_top=150)
            for i in range(10)]
    return out


def bischoff_ratcliff_lite(n_per_sku=8) -> List[Box]:
    """A small-scale BR-style instance: 5 SKUs with mixed sizes."""
    skus = [
        ("S1", 40, 30, 25, 4),
        ("S2", 50, 50, 30, 7),
        ("S3", 60, 40, 60, 10),
        ("S4", 30, 30, 90, 6),
        ("S5", 80, 30, 25, 5),
    ]
    out = []
    for pfx, L, W, H, w in skus:
        for i in range(n_per_sku):
            out.append(Box(id=f"{pfx}-{i:02d}", length=L, width=W, height=H,
                           weight=w, allowed_rotations=ALL_ROTATIONS))
    return out


def pareto_distribution_boxes(n=30, seed=0) -> List[Box]:
    """Many small, few large — typical of real e-commerce orders."""
    import random
    rng = random.Random(seed)
    out = []
    for i in range(n):
        # Pareto-ish: 70% small, 25% medium, 5% large
        roll = rng.random()
        if roll < 0.70:
            L = rng.randint(15, 30)
            W = rng.randint(15, 30)
            H = rng.randint(10, 25)
            w = rng.uniform(0.2, 2.0)
        elif roll < 0.95:
            L = rng.randint(30, 50)
            W = rng.randint(30, 50)
            H = rng.randint(25, 50)
            w = rng.uniform(2.0, 10.0)
        else:
            L = rng.randint(50, 80)
            W = rng.randint(50, 80)
            H = rng.randint(50, 80)
            w = rng.uniform(10.0, 25.0)
        out.append(Box(id=f"P-{i:03d}", length=L, width=W, height=H, weight=w))
    return out


def all_fragile(n=12) -> List[Box]:
    """Stack-incompatible items: nothing can stack on anything."""
    return [Box(id=f"F-{i:02d}", length=40, width=30, height=25, weight=2,
                max_load_on_top=0) for i in range(n)]


def long_thin_must_rotate() -> List[Box]:
    """Boxes longer than pallet length unless rotated."""
    return [Box(id=f"L-{i}", length=120, width=20, height=30, weight=3,
                allowed_rotations=ALL_ROTATIONS) for i in range(8)]


# ---------------------------------------------------------------------------
# Case library
# ---------------------------------------------------------------------------
PALLET = Pallet(length=120, width=100, height=100, max_weight=500)

def cases() -> List[Case]:
    out: List[Case] = []

    # A. Trivial / edge cases ----------------------------------------------
    out.append(Case(
        name="A1 empty input",
        boxes=[], pallet=PALLET,
        expected_pallets=0, expected_unpacked=0,
        notes="No work to do; should return cleanly."))

    out.append(Case(
        name="A2 single fitting box",
        boxes=[Box("solo", 30, 30, 30, weight=2)], pallet=PALLET,
        expected_pallets=1, expected_unpacked=0))

    out.append(Case(
        name="A3 single too-big box",
        boxes=[Box("huge", 150, 50, 50, weight=2)], pallet=PALLET,
        expected_pallets=0, expected_unpacked=1,
        notes="Box overhangs pallet length even after rotation."))

    out.append(Case(
        name="A4 box exactly = pallet",
        boxes=[Box("perfect", 120, 100, 100, weight=50)], pallet=PALLET,
        expected_pallets=1, expected_unpacked=0,
        notes="Volume utilisation should be 100%."))

    out.append(Case(
        name="A5 single too-heavy box",
        boxes=[Box("heavy", 50, 50, 50, weight=600)],
        pallet=Pallet(120, 100, 100, max_weight=500),
        expected_pallets=0, expected_unpacked=1,
        notes="Box weight exceeds pallet capacity."))

    # B. Known-optimal homogeneous packings -------------------------------
    out.append(Case(
        name="B1 perfect 2×2×2 fill",
        boxes=uniform_boxes(8, 60, 50, 50, weight=5, prefix="K"),
        pallet=PALLET, expected_pallets=1, expected_unpacked=0,
        notes="8 boxes exactly fill the pallet (theoretical 100%)."))

    out.append(Case(
        name="B2 12 boxes -> 2 pallets",
        boxes=uniform_boxes(12, 60, 50, 50, weight=5, prefix="K"),
        pallet=PALLET, expected_pallets=2, expected_unpacked=0,
        notes="8 + 4 split; pallet 2 is half-full."))

    out.append(Case(
        name="B3 4 large boxes -> 2 pallets exact",
        boxes=uniform_boxes(4, 60, 100, 100, weight=20, prefix="K"),
        pallet=PALLET, expected_pallets=2, expected_unpacked=0,
        notes="Each pallet holds exactly 2 boxes; 100% util each."))

    out.append(Case(
        name="B4 100 tiny boxes -> 1 pallet",
        boxes=uniform_boxes(100, 20, 20, 20, weight=0.5, prefix="T"),
        pallet=PALLET, expected_pallets=1, expected_unpacked=0,
        notes="Footprint allows 6×5=30/layer; 5 layers = 150 capacity > 100."))

    # C. Heterogeneous instances at scale ----------------------------------
    out.append(Case(
        name="C1 user's headline mix",
        boxes=mixed_classic(), pallet=PALLET,
        notes="20 A + 15 B + 10 C; lower bound 4 pallets by volume."))

    out.append(Case(
        name="C2 BR-lite (5 SKU × 8)",
        boxes=bischoff_ratcliff_lite(8), pallet=PALLET,
        notes="40 boxes across 5 sizes; Bischoff-Ratcliff style. MIP at "
              "300s improved on the earlier 35s result (3p/7u → 3p/4u) but "
              "still couldn't prove 3p/0u feasible nor 4p as true LB. "
              "Strongly suspected at-true-LB-of-4."))

    out.append(Case(
        name="C3 Pareto-distributed 30",
        boxes=pareto_distribution_boxes(30), pallet=PALLET,
        notes="E-commerce flavour: 70% small, 25% medium, 5% large."))

    out.append(Case(
        name="C4 stress 80 boxes",
        boxes=pareto_distribution_boxes(80, seed=1),
        pallet=Pallet(120, 100, 100, max_weight=800),
        notes="Scaling test."))

    # D. Constraint sensitivity sweep --------------------------------------
    # Use a fixed base instance and toggle each constraint.
    base_boxes = mixed_classic()

    out.append(Case(
        name="D1 base: all constraints relaxed",
        boxes=base_boxes, pallet=PALLET,
        config=PackerConfig(
            support_ratio=0.0, require_centroid_supported=False,
            enforce_load_bearing=False, cog_envelope_fraction=1.0,
            multi_start_trials=25),
        notes="Pure volume packing; gives utilisation upper bound."))

    out.append(Case(
        name="D2 + support_ratio=0.8 (default)",
        boxes=base_boxes, pallet=PALLET,
        config=PackerConfig(multi_start_trials=25),
        notes="Default settings."))

    out.append(Case(
        name="D3 + support_ratio=1.0 (no overhang)",
        boxes=base_boxes, pallet=PALLET,
        config=PackerConfig(support_ratio=1.0, multi_start_trials=25),
        expected_pallets=5, expected_unpacked=0,
        notes="Strict: each box's full base must rest on something. "
              "MIP-proven (300s, OPTIMAL): a 4-pallet packing exists but "
              "fits at most 43/45 items under support_ratio=1.0. All 45 "
              "require 5 pallets. The default volume LB of 4 is loose."))

    out.append(Case(
        name="D4 + binding weight limit",
        boxes=base_boxes,
        pallet=Pallet(120, 100, 100, max_weight=200),  # was 500
        config=PackerConfig(multi_start_trials=25),
        notes="Weight forces more pallets than volume would."))

    out.append(Case(
        name="D5 + tight CoG envelope",
        boxes=base_boxes, pallet=PALLET,
        config=PackerConfig(cog_envelope_fraction=0.10, multi_start_trials=25),
        notes="Pallet CoG must stay within ±10% of centre."))

    out.append(Case(
        name="D6 + this-side-up for all",
        boxes=[Box(id=b.id, length=b.length, width=b.width, height=b.height,
                   weight=b.weight, allowed_rotations=THIS_SIDE_UP,
                   max_load_on_top=b.max_load_on_top) for b in base_boxes],
        pallet=PALLET, config=PackerConfig(multi_start_trials=25),
        notes="Halves the rotation search space."))

    out.append(Case(
        name="D7 + NO rotation at all",
        boxes=[Box(id=b.id, length=b.length, width=b.width, height=b.height,
                   weight=b.weight, allowed_rotations=NO_ROTATION,
                   max_load_on_top=b.max_load_on_top) for b in base_boxes],
        pallet=PALLET, config=PackerConfig(multi_start_trials=25),
        notes="Single orientation; most restrictive. MIP at 300s found only "
              "4p/4u (vs v1's 5p/0u) — couldn't prove 4p/0u exists nor that "
              "5p is necessary. Strongly suspected constraint-bound at 5p; "
              "would need hours of MIP to formalize."))

    out.append(Case(
        name="D8 + fragile B boxes",
        boxes=[Box(id=b.id, length=b.length, width=b.width, height=b.height,
                   weight=b.weight, allowed_rotations=b.allowed_rotations,
                   max_load_on_top=(0 if b.id.startswith("B") else b.max_load_on_top))
               for b in base_boxes],
        pallet=PALLET, config=PackerConfig(multi_start_trials=25),
        expected_pallets=6, expected_unpacked=0,
        notes="15 fragile boxes can't be stacked; volume LB of 4 is loose "
              "under fragility. MIP-proven: 5 pallets fit at most 39/45; "
              "all 45 require 6 pallets. 6p IS the true LB."))

    # E. Pathological -------------------------------------------------------
    out.append(Case(
        name="E1 boxes that MUST rotate",
        boxes=long_thin_must_rotate(), pallet=PALLET,
        notes="L=120 = pallet length exactly; only rotating allows packing 5+."))

    out.append(Case(
        name="E2 all fragile",
        boxes=all_fragile(12), pallet=PALLET,
        notes="No stacking possible; pallet height wasted."))

    out.append(Case(
        name="E3 weight forces sparse layout",
        boxes=uniform_boxes(6, 40, 40, 40, weight=120, prefix="H"),
        pallet=Pallet(120, 100, 100, max_weight=300),
        notes="Only 2 fit per pallet by weight; 3 pallets needed."))

    out.append(Case(
        name="E4 needle eye: many almost-too-big",
        boxes=uniform_boxes(6, 110, 95, 95, weight=20, prefix="N"),
        pallet=PALLET,
        notes="Each box almost fills the pallet; 1 box/pallet; no rotation help."))

    # F. Real-world flavours -----------------------------------------------
    out.append(Case(
        name="F1 e-commerce: 50 small + 3 big",
        boxes=(uniform_boxes(50, 20, 15, 10, weight=0.3, prefix="S") +
               uniform_boxes(3, 80, 60, 50, weight=15, prefix="B")),
        pallet=Pallet(120, 100, 100, max_weight=200),
        notes="Mixed-size order; small items fill gaps around big ones."))

    out.append(Case(
        name="F2 pharma: many fragile",
        boxes=[Box(id=f"M-{i:03d}", length=15, width=10, height=20,
                   weight=0.1, allowed_rotations=THIS_SIDE_UP,
                   max_load_on_top=2.0) for i in range(120)],
        pallet=PALLET,
        config=PackerConfig(multi_start_trials=10),  # 120 boxes — keep trials modest
        notes="120 light vials; thin loading; load_bearing limits stack height."))

    out.append(Case(
        name="F3 heavy industrial: few big",
        boxes=uniform_boxes(8, 60, 50, 40, weight=80, prefix="I",
                           allowed_rotations=THIS_SIDE_UP),
        pallet=Pallet(120, 100, 100, max_weight=400),
        notes="Heavy units; weight binds at 5 per pallet."))

    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def lower_bound(boxes, pallet) -> int:
    """Volume-based lower bound on pallet count."""
    if not boxes:
        return 0
    vol = sum(b.volume for b in boxes)
    cap = pallet.length * pallet.width * pallet.height
    return max(1, math.ceil(vol / cap))


def constraint_sensitivity_table():
    """Each row shows the same instance with a single constraint toggled,
    chosen so that the constraint is actually binding."""
    print("Constraint impact (each case designed so the constraint binds)")
    print("=" * 78)
    print(f"{'Constraint':<48} {'Pallets':>8} {'Util':>7} {'Unp':>4}")
    print("-" * 78)

    def run_and_show(label, boxes, pallet, cfg):
        packer = PalletPacker(pallet, cfg)
        r = packer.pack(boxes)
        print(f"{label:<48} {r.num_pallets:>8} "
              f"{r.total_volume_utilisation:>6.1%} {len(r.unpacked):>4}")

    # SUPPORT - inverted-pyramid scenario: wide boxes that must rest on narrow.
    # Only with support_ratio < ~0.45 can the wide boxes stack on the narrow.
    inv_pyramid = ([Box("narrow", 40, 40, 30, weight=5)] +
                   [Box(f"wide-{i}", 60, 60, 30, weight=4) for i in range(5)])
    run_and_show("SUPPORT 0.4 (allows partial-support stacking)",
                 inv_pyramid, PALLET,
                 PackerConfig(support_ratio=0.4,
                              require_centroid_supported=False,
                              multi_start_trials=10))
    run_and_show("SUPPORT 0.8 (default)",
                 inv_pyramid, PALLET,
                 PackerConfig(support_ratio=0.8, multi_start_trials=10))
    run_and_show("SUPPORT 1.0 (no overhang anywhere)",
                 inv_pyramid, PALLET,
                 PackerConfig(support_ratio=1.0, multi_start_trials=10))

    print()
    # WEIGHT
    heavy = [Box(f"W-{i}", 40, 40, 40, weight=60) for i in range(10)]
    run_and_show("WEIGHT: 10000 kg/pallet (volume only)", heavy,
                 Pallet(120, 100, 100, max_weight=10000),
                 PackerConfig(multi_start_trials=10))
    run_and_show("WEIGHT: 300 kg/pallet (5 boxes max)", heavy,
                 Pallet(120, 100, 100, max_weight=300),
                 PackerConfig(multi_start_trials=10))
    run_and_show("WEIGHT: 120 kg/pallet (2 boxes max)", heavy,
                 Pallet(120, 100, 100, max_weight=120),
                 PackerConfig(multi_start_trials=10))

    print()
    # COG - 4 heavy boxes that get loaded into one quadrant by default;
    # tight CoG envelope should reject this and force spreading.
    cog_boxes = [Box(f"G-{i}", 40, 40, 40, weight=80) for i in range(4)]
    cog_pallet = Pallet(120, 100, 100, max_weight=400)
    run_and_show("COG: ±50% (loose)", cog_boxes, cog_pallet,
                 PackerConfig(cog_envelope_fraction=0.50, multi_start_trials=10))
    run_and_show("COG: ±25% (default)", cog_boxes, cog_pallet,
                 PackerConfig(cog_envelope_fraction=0.25, multi_start_trials=10))
    run_and_show("COG: ±5% (very tight)", cog_boxes, cog_pallet,
                 PackerConfig(cog_envelope_fraction=0.05, multi_start_trials=10))

    print()
    # ROTATION
    tall6 = [Box(f"R-{i}", 30, 30, 90, weight=2,
                 allowed_rotations=ALL_ROTATIONS) for i in range(8)]
    tall_up = [Box(f"R-{i}", 30, 30, 90, weight=2,
                   allowed_rotations=THIS_SIDE_UP) for i in range(8)]
    tall_none = [Box(f"R-{i}", 30, 30, 90, weight=2,
                     allowed_rotations=NO_ROTATION) for i in range(8)]
    run_and_show("ROTATION: 6 orientations", tall6, PALLET,
                 PackerConfig(multi_start_trials=10))
    run_and_show("ROTATION: this-side-up (2 orientations)", tall_up, PALLET,
                 PackerConfig(multi_start_trials=10))
    run_and_show("ROTATION: fixed (1 orientation)", tall_none, PALLET,
                 PackerConfig(multi_start_trials=10))

    print()
    # FRAGILITY
    stack_ok = [Box(f"S-{i}", 40, 40, 30, weight=4) for i in range(18)]
    half_frag = ([Box(f"O-{i}", 40, 40, 30, weight=4) for i in range(9)] +
                 [Box(f"F-{i}", 40, 40, 30, weight=4, max_load_on_top=0)
                  for i in range(9)])
    all_frag = [Box(f"F-{i}", 40, 40, 30, weight=4, max_load_on_top=0)
                for i in range(18)]
    run_and_show("FRAGILITY: all stackable", stack_ok, PALLET,
                 PackerConfig(multi_start_trials=10))
    run_and_show("FRAGILITY: half fragile", half_frag, PALLET,
                 PackerConfig(multi_start_trials=10))
    run_and_show("FRAGILITY: all fragile (no stacking)", all_frag, PALLET,
                 PackerConfig(multi_start_trials=10))

    print()


def report(outcomes: List[Outcome]):
    width = 36
    header = (f"{'Case':<{width}} {'In':>4} {'LB':>3} {'Out':>4} "
              f"{'Unp':>4} {'Util':>6} {'Time':>7}  {'Status'}")
    print(header)
    print("-" * len(header))
    for o in outcomes:
        n = len(o.case.boxes)
        lb = lower_bound(o.case.boxes, o.case.pallet)
        util = o.result.total_volume_utilisation
        time_str = f"{o.runtime*1000:6.0f}ms" if o.runtime < 1 else f"{o.runtime:6.2f}s"
        print(f"{o.case.name:<{width}} {n:>4} {lb:>3} "
              f"{o.result.num_pallets:>4} {len(o.result.unpacked):>4} "
              f"{util:>5.1%} {time_str:>7}  {o.status()}")
        if o.errors:
            for e in o.errors[:3]:
                print(f"    ! {e}")
    print()


def constraint_sensitivity(outcomes: List[Outcome]):
    """Table comparing D1..D8 against each other."""
    d_outcomes = [o for o in outcomes if o.case.name.startswith("D")]
    if not d_outcomes:
        return
    print("Constraint sensitivity on the 45-box headline instance")
    print("(showing that constraints leave volume-bound packings unchanged)")
    print("=" * 72)
    base = d_outcomes[0]
    print(f"{'Variant':<40} {'Pallets':>8} {'Δ':>4} {'Util':>7}")
    print("-" * 72)
    for o in d_outcomes:
        delta = o.result.num_pallets - base.result.num_pallets
        delta_str = f"{delta:+d}" if delta != 0 else "—"
        print(f"{o.case.name[3:]:<40} {o.result.num_pallets:>8} "
              f"{delta_str:>4} {o.result.total_volume_utilisation:>6.1%}")
    print()


def multi_start_ablation():
    """How much does the metaheuristic actually help?"""
    print("Multi-start ablation")
    print("=" * 72)
    for instance_name, boxes in [
        ("45-box headline", mixed_classic()),
        ("Pareto 30 (more heterogeneous)", pareto_distribution_boxes(30)),
        ("BR-lite 40 boxes", bischoff_ratcliff_lite(8)),
    ]:
        print(f"\nInstance: {instance_name}")
        print(f"{'Trials':>8} {'Pallets':>8} {'Util':>8} {'Time':>8}")
        print("-" * 40)
        for trials in [1, 2, 5, 10, 25, 50]:
            cfg = PackerConfig(multi_start_trials=trials, seed=42)
            packer = PalletPacker(PALLET, cfg)
            t0 = time.time()
            result = packer.pack(boxes)
            elapsed = time.time() - t0
            time_str = f"{elapsed*1000:5.0f}ms" if elapsed < 1 else f"{elapsed:5.2f}s"
            print(f"{trials:>8} {result.num_pallets:>8} "
                  f"{result.total_volume_utilisation:>7.1%} {time_str:>8}")
    print()


def variance_analysis(n_seeds=10):
    """How sensitive is the result to the random seed?"""
    print(f"Seed variance (45-box instance, {n_seeds} seeds, 25 trials each)")
    print("=" * 72)
    pallet = PALLET
    boxes = mixed_classic()
    pallets_seen, utils = [], []
    for seed in range(n_seeds):
        cfg = PackerConfig(multi_start_trials=25, seed=seed)
        packer = PalletPacker(pallet, cfg)
        result = packer.pack(boxes)
        pallets_seen.append(result.num_pallets)
        utils.append(result.total_volume_utilisation)
    print(f"  Pallets: min={min(pallets_seen)} max={max(pallets_seen)} "
          f"mean={statistics.mean(pallets_seen):.2f} "
          f"stdev={statistics.stdev(pallets_seen):.2f}")
    print(f"  Utilisation: min={min(utils):.1%} max={max(utils):.1%} "
          f"mean={statistics.mean(utils):.1%}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    outcomes: List[Outcome] = []
    for case in cases():
        outcomes.append(run(case))

    report(outcomes)
    constraint_sensitivity(outcomes)
    constraint_sensitivity_table()
    multi_start_ablation()
    variance_analysis()

    failed = [o for o in outcomes if not o.passed]
    print(f"Overall: {len(outcomes) - len(failed)}/{len(outcomes)} cases passed.")
    if failed:
        print("Failures:")
        for o in failed:
            exp_p = o.case.expected_pallets
            exp_u = o.case.expected_unpacked
            print(f"  - {o.case.name}: got {o.result.num_pallets} pallets / "
                  f"{len(o.result.unpacked)} unpacked, expected "
                  f"{exp_p} / {exp_u}")
