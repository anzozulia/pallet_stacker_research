"""
failure_cases.py - Cases designed to expose where the heuristic fails.

For each case:
  - Construct an instance with a knowable better solution
  - Measure what our algorithm produces
  - Quantify the gap
  - Map to a literature fix (Eley 2002, Bortfeldt 2000, etc.)
"""
from __future__ import annotations
import math
import time
from dataclasses import dataclass
from typing import List, Optional, Callable

from pallet_packer import (
    Box, Pallet, PalletPacker, PackerConfig,
    ALL_ROTATIONS, THIS_SIDE_UP, NO_ROTATION,
    PackResult, validate,
)

PALLET = Pallet(length=120, width=100, height=100, max_weight=1000)


@dataclass
class FailureCase:
    name: str
    description: str                # What the case is testing
    boxes_factory: Callable[[], List[Box]]
    pallet: Pallet
    config: PackerConfig
    known_better: str               # What we know an oracle / smarter algo could do
    fix_reference: str              # Which literature method addresses this


def run_case(case: FailureCase):
    boxes = case.boxes_factory()
    packer = PalletPacker(case.pallet, case.config)
    t0 = time.time()
    result = packer.pack(boxes)
    elapsed = time.time() - t0
    errors = validate(result, case.pallet, case.config)
    return result, elapsed, errors, boxes


def lb_volume(boxes, pallet) -> int:
    if not boxes:
        return 0
    vol = sum(b.volume for b in boxes)
    cap = pallet.length * pallet.width * pallet.height
    return max(1, math.ceil(vol / cap))


# ---------------------------------------------------------------------------
# Failure case constructors
# ---------------------------------------------------------------------------
def case_F1_pareto_continuum() -> FailureCase:
    """Highly heterogeneous: classic 'long tail' e-commerce profile."""
    def boxes():
        import random
        rng = random.Random(123)
        out = []
        # 8 sizes drawn from a wide range — each appears 5-8 times.
        for sku_idx in range(8):
            L = rng.randint(20, 90)
            W = rng.randint(20, 80)
            H = rng.randint(20, 70)
            n = rng.randint(5, 8)
            wt = (L * W * H) / 15000.0
            for j in range(n):
                out.append(Box(id=f"S{sku_idx}-{j:02d}",
                               length=L, width=W, height=H, weight=wt))
        return out
    return FailureCase(
        name="F1 Pareto continuum",
        description=(
            "8 SKUs with semi-random sizes from 20–90 mm in each dimension. "
            "Common in e-commerce where every SKU is different."),
        boxes_factory=boxes,
        pallet=PALLET,
        config=PackerConfig(multi_start_trials=15),
        known_better=(
            "An oracle that runs an exact MIP solver (do Nascimento et al. 2021) "
            "would close this gap; for heuristics, GRASP + maximal-space "
            "(Parreño et al. 2008) typically gains 5–10 pp utilisation."),
        fix_reference="Parreño et al. (2008); do Nascimento et al. (2021)")


def case_F2_block_of_identicals() -> FailureCase:
    """Many identical small + a few large. Block-building wins big."""
    def boxes():
        # 24 small identical boxes (would block-build into a single 120×80×30 layer)
        out = [Box(id=f"K-{i:02d}", length=20, width=20, height=30, weight=1.0)
               for i in range(24)]
        # Plus 6 medium boxes
        out += [Box(id=f"M-{i:02d}", length=60, width=50, height=70, weight=8.0)
                for i in range(6)]
        return out
    return FailureCase(
        name="F2 large block of identicals",
        description=(
            "24 identical small boxes (20×20×30) + 6 medium (60×50×70). "
            "The 24 identicals SHOULD be packed as a single block; our "
            "one-at-a-time placer disperses them and fragments the pallet."),
        boxes_factory=boxes,
        pallet=PALLET,
        config=PackerConfig(multi_start_trials=15),
        known_better=(
            "Eley (2002) block-building: identify identical SKUs, pack as "
            "rectangular blocks of (n_x × n_y × n_z) units, then arrange the "
            "blocks. A 6×4×1 block of the small boxes occupies 120×80×30 — "
            "leaving a clean 120×100×70 space for the medium boxes."),
        fix_reference="Eley (2002); Bortfeldt (2000)")


def case_F3_interlock_pattern() -> FailureCase:
    """Boxes that tile perfectly only if interlocked across a row."""
    def boxes():
        # 6 of 80×50×30 + 6 of 40×50×30.
        # In a 120×100 footprint, one 80+40 row fits along X = 120, taking 50mm of Y.
        # Two rows = 100mm of Y. So 2 rows × 1 (each row is 1 box of 80 + 1 of 40)
        #   = 2 boxes 80 + 2 boxes 40 per layer.
        # Height 100 / 30 = 3 layers → 6 of 80, 6 of 40 = 12 boxes total = exact fit.
        # Volume = 12 × (80*50*30 + 40*50*30) /2 = 12 × 90K = ... let me redo
        # 80*50*30 = 120K, 40*50*30 = 60K. 6 of each = 6*120K + 6*60K = 1080K.
        # Pallet volume 1.2M. So 90% util if perfectly packed.
        out = [Box(id=f"L-{i:02d}", length=80, width=50, height=30, weight=4)
               for i in range(6)]
        out += [Box(id=f"S-{i:02d}", length=40, width=50, height=30, weight=2)
                for i in range(6)]
        return out
    return FailureCase(
        name="F3 interlock big+small",
        description=(
            "12 boxes that perfectly tile 1 pallet at 90% util — but only "
            "if a big (80) and small (40) alternate in each row. Our "
            "greedy heaviest-first sort places all 6 big boxes before "
            "any small, creating fragmented gaps."),
        boxes_factory=boxes,
        pallet=PALLET,
        config=PackerConfig(multi_start_trials=15),
        known_better=(
            "Layer-/wall-building heuristics (Bischoff & Ratcliff 1995; "
            "George & Robinson 1980) construct each layer with a mix of "
            "compatible SKUs, exactly the structure this case demands."),
        fix_reference="George & Robinson (1980); Bischoff & Ratcliff (1995)")


def case_F4_height_mismatch() -> FailureCase:
    """Three different heights that should be paired to total = pallet H."""
    def boxes():
        # Pallet H = 100. Pairs: 60+40=100, 70+30=100, 50+50=100.
        # 4 of each height, all same footprint 60×50.
        # Footprint 60×50 fits 2×2=4 per layer → 100% util if paired optimally.
        out = []
        out += [Box(id=f"T-{i}", length=60, width=50, height=60, weight=3)
                for i in range(2)]
        out += [Box(id=f"M-{i}", length=60, width=50, height=40, weight=2)
                for i in range(2)]
        out += [Box(id=f"X-{i}", length=60, width=50, height=70, weight=4)
                for i in range(2)]
        out += [Box(id=f"Y-{i}", length=60, width=50, height=30, weight=1)
                for i in range(2)]
        return out
    return FailureCase(
        name="F4 height pairing",
        description=(
            "8 boxes that pair into 4 columns of pallet-height 100mm each "
            "(60+40 and 70+30). Optimal: 1 pallet, 100% util. The heuristic "
            "picks heaviest-first (X70 weighs 4kg, then T60 at 3kg) and "
            "stacks short-on-tall, wasting headroom."),
        boxes_factory=boxes,
        pallet=PALLET,
        config=PackerConfig(multi_start_trials=20),
        known_better=(
            "Pre-process: pair items whose heights sum to pallet H (column "
            "construction, Bortfeldt 2000). Or post-process: identify two "
            "short columns that would consolidate into one full column."),
        fix_reference="Bortfeldt (2000) column construction")


def case_F5_must_rotate_early() -> FailureCase:
    """Picking the wrong rotation for early boxes fragments the pallet."""
    def boxes():
        # 6 boxes of 100×40×30. They fit "long way" along X (100<120) → 1×2 = 2/layer.
        # OR they fit "long way" along Y (100=100) → only 1/layer because 100=pallet Y.
        # Optimal: all aligned along Y → 1 wide × 3 deep × 3 high = wait,
        # 100 along Y means 100/40 if rotated? Let me think.
        # Box 100×40×30. Rotations: (100,40,30), (40,100,30), (100,30,40),
        #   (30,40,100), (40,30,100), (30,100,40).
        # Putting H=30 vertical: footprints (100,40) or (40,100).
        # (100,40): 1 along X (120/100=1), 2 along Y (100/40=2) → 2/layer.
        # (40,100): 3 along X (120/40=3), 1 along Y (100/100=1) → 3/layer ← better!
        # 6 boxes → 2 layers of 3 → height 60mm. 1 pallet, 60% util.
        # Volume: 6 × 120K = 720K. 720K/1.2M = 60% if perfectly packed.
        # If algorithm picks (100,40) for some and (40,100) for others, fragmentation.
        return [Box(id=f"R-{i}", length=100, width=40, height=30, weight=2,
                    allowed_rotations=ALL_ROTATIONS) for i in range(6)]
    return FailureCase(
        name="F5 wrong-rotation cascade",
        description=(
            "6 boxes of 100×40×30. The optimal orientation packs 3 per layer; "
            "any other packs ≤2 per layer. The heuristic tries each rotation "
            "for each box independently and may inconsistently use both."),
        boxes_factory=boxes,
        pallet=PALLET,
        config=PackerConfig(multi_start_trials=20),
        known_better=(
            "Rotation-grouped heuristics: choose a global rotation per SKU "
            "(based on a layer-density score) and reuse it across all units of "
            "that SKU. Implemented in Bortfeldt & Gehring (2001) GA."),
        fix_reference="Bortfeldt & Gehring (2001); SKU-consistent rotation")


def case_F6_constraint_cascade() -> FailureCase:
    """Multiple constraints that individually allow but jointly forbid the best layout."""
    def boxes():
        # 8 fragile boxes (40×40×50, max_load_on_top=0) + 8 robust same size.
        # Fragile must be on TOP. With 4 floor positions per layer (120/40=3, 100/40=2),
        # max 6 per floor. Need 8 robust on floor → won't fit in 1 layer.
        # Best: floor layer (6 robust) → top layer (6 robust + fragile alternating?
        # No, fragile can carry nothing on top so they must be uppermost).
        # Actually 16 boxes × 40×40×50 = 16×80K = 1280K > 1.2M → needs 2 pallets.
        # But fragility could push it to 3 if the heuristic stacks badly.
        return ([Box(id=f"F-{i:02d}", length=40, width=40, height=50, weight=2,
                     max_load_on_top=0) for i in range(8)] +
                [Box(id=f"R-{i:02d}", length=40, width=40, height=50, weight=2)
                 for i in range(8)])
    return FailureCase(
        name="F6 fragility + dense fit interact",
        description=(
            "8 fragile + 8 robust, all 40×40×50. Volume bound = 2 pallets. "
            "The fragility constraint can push count to 3 if the heuristic "
            "places fragile items before allocating capacity to robust ones."),
        boxes_factory=boxes,
        pallet=PALLET,
        config=PackerConfig(multi_start_trials=20),
        known_better=(
            "Constraint-aware item ordering: sort fragile items LAST so they "
            "naturally end up at the top of stacks. Or use a hierarchical "
            "scheme (Junqueira et al. 2012) that explicitly models the "
            "stackability DAG."),
        fix_reference="Junqueira, Morabito & Yamashita (2012)")


def case_F7_layered_pyramid() -> FailureCase:
    """Wide base + narrow tower — needs pyramid construction, not greedy stacking."""
    def boxes():
        # 1 wide flat box (120×100×20) — full pallet footprint, low.
        # Then 8 medium boxes (40×50×40) on top — 3×2=6 per layer × 2 layers = could fit 12.
        # Then 16 small boxes (20×25×20) on the very top — 6×4=24 per layer.
        # Total volume: 240K + 8×80K + 16×10K = 240K + 640K + 160K = 1040K ≈ 87% util.
        # Optimal: 1 pallet.
        # The heuristic sorts heaviest first; the wide flat is the lightest by
        # weight-per-volume but biggest volume, so depending on tie-breaking
        # it might go first (correctly) or get displaced.
        out = [Box(id="WIDE-1", length=120, width=100, height=20, weight=10)]
        out += [Box(id=f"MED-{i:02d}", length=40, width=50, height=40, weight=4)
                for i in range(8)]
        out += [Box(id=f"SML-{i:02d}", length=20, width=25, height=20, weight=0.5)
                for i in range(16)]
        return out
    return FailureCase(
        name="F7 layered pyramid",
        description=(
            "1 wide flat base + 8 medium + 16 small. Optimal stacks in 3 "
            "tiers totaling 80 mm of height (20+40+20), 87 % util, 1 pallet. "
            "The heuristic's heaviest-first order might place medium boxes "
            "before the flat base, leaving no room for the base."),
        boxes_factory=boxes,
        pallet=PALLET,
        config=PackerConfig(multi_start_trials=20),
        known_better=(
            "Bischoff-Ratcliff 1995 layer-building constructs from the floor "
            "up, naturally putting flat large boxes first regardless of weight. "
            "Equivalent to a different seed-ordering heuristic."),
        fix_reference="Bischoff & Ratcliff (1995) layer-building")


def case_F8_strongly_heterogeneous() -> FailureCase:
    """Many distinct SKUs, none in quantity — worst case for any heuristic."""
    def boxes():
        out = []
        # 30 SKUs, 1-2 boxes each.
        import random
        rng = random.Random(7)
        sizes = [(40, 30, 30), (60, 40, 40), (80, 30, 50), (50, 50, 40),
                 (30, 30, 70), (90, 40, 30), (20, 60, 50), (70, 50, 30),
                 (40, 40, 60), (60, 30, 80), (50, 40, 50), (30, 80, 40),
                 (20, 50, 30), (60, 60, 30), (40, 50, 80), (90, 20, 40)]
        for i, (L, W, H) in enumerate(sizes):
            for j in range(rng.randint(1, 2)):
                out.append(Box(id=f"H{i:02d}-{j}", length=L, width=W,
                               height=H, weight=L*W*H/20000))
        return out
    return FailureCase(
        name="F8 strongly heterogeneous",
        description=(
            "16 distinct SKUs, 1-2 boxes each. Volume bound is ~2 pallets but "
            "no two boxes have the same shape — block construction can't help. "
            "This is Wäscher's SCLP-strongly-heterogeneous, the hardest "
            "variant."),
        boxes_factory=boxes,
        pallet=PALLET,
        config=PackerConfig(multi_start_trials=30),
        known_better=(
            "GRASP + restart with diverse seed orderings (Parreño et al. 2008) "
            "or BRKGA with full crossover (Gonçalves & Resende 2013). "
            "For small N (<30 here), exact MIP (do Nascimento et al. 2021) "
            "is feasible."),
        fix_reference="Parreño et al. (2008); Gonçalves & Resende (2013)")


def case_F9_long_unrotatable() -> FailureCase:
    """Long items + this-side-up: tests rotation pruning under constraint."""
    def boxes():
        # 110×30×40 with this-side-up. The two allowed orientations are
        # (110,30,40) and (30,110,40). The latter needs Y=110 > pallet 100,
        # so only (110,30,40) is geometrically feasible — 3 per layer,
        # 2 layers, 6 per pallet → 3 pallets for 18 boxes.
        # Volume LB says 2, but the TRUE LB (considering geometry) is 3.
        return [Box(id=f"L-{i:02d}", length=110, width=30, height=40, weight=5,
                    allowed_rotations=THIS_SIDE_UP) for i in range(18)]
    return FailureCase(
        name="F9 long this-side-up",
        description=(
            "18 long boxes (110×30×40), this-side-up. Volume LB says 2, "
            "but the (30,110,40) rotation exceeds pallet Y; only (110,30,40) "
            "fits and yields 6/pallet → 3 pallets is OPTIMAL. Demonstrates "
            "that volume LB is loose: the true LB needs a feasibility check."),
        boxes_factory=boxes,
        pallet=PALLET,
        config=PackerConfig(multi_start_trials=20),
        known_better=(
            "Not a failure — the algorithm matches the true geometric LB. "
            "Lesson: the volume-based LB used in the report is a lower bound "
            "on the lower bound, not the actual optimum."),
        fix_reference="(none — volume LB is loose for tight pallet dimensions)")


def case_F10_overhang_required() -> FailureCase:
    """Hypothesis: overhang would be needed but algorithm refuses. RESULT: passes."""
    def boxes():
        return [Box(id=f"O-{i}", length=70, width=60, height=100, weight=4,
                    allowed_rotations=THIS_SIDE_UP) for i in range(4)]
    return FailureCase(
        name="F10 overhang test (PASSES: overhang IS used)",
        description=(
            "4 boxes 70×60×100 (matching pallet height, this-side-up). Without "
            "overhang only 1 fits per pallet; with 20 mm overhang permitted, "
            "4 fit in 1 pallet (2×2 grid, each box overhangs by 20 mm in one "
            "or both directions). The algorithm finds this correctly via the "
            "rotation-aware corner anchors. Note: utilisation reports >100% "
            "because the cargo footprint exceeds pallet footprint — that's the "
            "right answer, just an artefact of measuring against pallet volume."),
        boxes_factory=boxes,
        pallet=Pallet(120, 100, 100, max_weight=200, max_overhang=20),
        config=PackerConfig(allow_pallet_overhang=True, multi_start_trials=10),
        known_better=(
            "Algorithm already handles this. The fix was added during sanity "
            "testing: rotation-aware floor-corner anchors at "
            "(max(0, L-dx), max(0, W-dy), 0) per rotation."),
        fix_reference="(addressed in v1; was a bug, not a limitation)")


def case_F11_spurious_unpacked() -> FailureCase:
    """A small box left unpacked due to greedy first-fit when a swap fits everything."""
    def boxes():
        # Pallet 120×100×100. Strategy: place a "wide" obstacle, then small
        # boxes try to slot around it. The greedy algorithm might place a
        # medium box in a position that geometrically forbids the last box.
        #
        # Construct: 1 large box that fills central area; small boxes around;
        # plus a "tight-fit" box that ONLY has one feasible position.
        # If the tight-fit is placed first, it works. If placed last, the
        # heuristic may have consumed its only slot.
        out = []
        # 6 medium boxes that the heuristic places first (heaviest)
        for i in range(6):
            out.append(Box(id=f"M-{i:02d}", length=40, width=40, height=30,
                           weight=5))
        # 1 awkward narrow tall box that needs a specific corner free
        out.append(Box(id="TALL", length=20, width=20, height=100, weight=1))
        return out
    return FailureCase(
        name="F11 greedy leaves an item unpacked",
        description=(
            "6 medium boxes + 1 tall narrow box. The narrow box only fits if "
            "a corner is left free; the heuristic's heaviest-first sort fills "
            "those corners with medium boxes. Tests whether the algorithm "
            "wrongly leaves the narrow box unpacked, or finds a configuration."),
        boxes_factory=boxes,
        pallet=PALLET,
        config=PackerConfig(multi_start_trials=15),
        known_better=(
            "GRASP (Greedy Randomized Adaptive Search Procedure) restarts "
            "from many random orderings; with enough restarts a feasible "
            "configuration emerges. Repair heuristics (Tabu Search ejection "
            "chains) can also rescue an unpacked item by displacing items."),
        fix_reference="Parreño et al. (2010) GRASP; Crainic et al. (2009) TS²PACK")


def case_F12_strongly_heterogeneous_at_scale() -> FailureCase:
    """Many distinct SKUs — Bortfeldt & Wäscher's hardest sub-class."""
    def boxes():
        import random
        rng = random.Random(99)
        out = []
        # 40 SKUs each with 1-3 units, mostly small but with a few big.
        for i in range(40):
            L = rng.choice([15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
            W = rng.choice([15, 20, 25, 30, 35, 40, 45, 50])
            H = rng.choice([10, 15, 20, 25, 30, 40, 50])
            n = rng.choice([1, 1, 2, 2, 3])
            for j in range(n):
                out.append(Box(id=f"X{i:02d}-{j}", length=L, width=W,
                               height=H, weight=L*W*H/50000))
        return out
    return FailureCase(
        name="F12 strongly heterogeneous at scale",
        description=(
            "40 distinct SKUs × 1–3 units each (~80 boxes). Wäscher's "
            "'strongly heterogeneous' class. No block construction can "
            "help — each SKU has only a few units. Heuristic must rely "
            "purely on placement intelligence."),
        boxes_factory=boxes,
        pallet=PALLET,
        config=PackerConfig(multi_start_trials=10),  # 80 boxes — keep modest
        known_better=(
            "True BRKGA with population recombination (Gonçalves & Resende "
            "2013) typically beats single-decoder heuristics by 2-3 pp "
            "utilisation on these instances. For under 30 boxes per pallet, "
            "exact MIP polish (do Nascimento et al. 2021) can close further."),
        fix_reference="Gonçalves & Resende (2013); do Nascimento et al. (2021)")


def case_F13_layer_advantage() -> FailureCase:
    """Pure layered cargo: all boxes have one of two heights = layer-building heaven."""
    def boxes():
        # All H=50 mm. Mix of footprints. Pallet H=100 = exactly 2 layers.
        # Optimal: build 2 dense layers.
        return ([Box(id=f"A-{i:02d}", length=60, width=40, height=50, weight=4)
                 for i in range(8)] +
                [Box(id=f"B-{i:02d}", length=40, width=40, height=50, weight=3)
                 for i in range(6)] +
                [Box(id=f"C-{i:02d}", length=30, width=20, height=50, weight=1)
                 for i in range(8)])
    return FailureCase(
        name="F13 pure layered cargo",
        description=(
            "22 boxes, all H=50, pallet H=100 = exactly 2 layers. Mixed "
            "footprints. A layer-building heuristic would tile each layer "
            "independently. Our column-oriented placer treats it as a 3D "
            "problem and may waste capacity."),
        boxes_factory=boxes,
        pallet=PALLET,
        config=PackerConfig(multi_start_trials=20),
        known_better=(
            "When all (or most) boxes share a height, the problem reduces "
            "to 2D bin packing per layer. George & Robinson (1980) "
            "wall-building and Bischoff-Ratcliff (1995) layer-building "
            "exploit this directly; running a 2D-MaxRects (`rectpack`) "
            "per layer can be far better than 3D EP."),
        fix_reference="George & Robinson (1980); Burke, Kendall & Whitwell (2004)")


# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------
def main():
    cases = [
        case_F1_pareto_continuum(),
        case_F2_block_of_identicals(),
        case_F3_interlock_pattern(),
        case_F4_height_mismatch(),
        case_F5_must_rotate_early(),
        case_F6_constraint_cascade(),
        case_F7_layered_pyramid(),
        case_F8_strongly_heterogeneous(),
        case_F9_long_unrotatable(),
        case_F10_overhang_required(),
        case_F11_spurious_unpacked(),
        case_F12_strongly_heterogeneous_at_scale(),
        case_F13_layer_advantage(),
    ]

    print(f"{'Case':<38} {'N':>4} {'LB':>3} {'Got':>4} {'Util':>6} "
          f"{'Unp':>4} {'Time':>7}")
    print("-" * 78)
    results = []
    for c in cases:
        result, elapsed, errors, boxes = run_case(c)
        lb = lb_volume(boxes, c.pallet)
        gap = result.num_pallets - lb
        gap_str = f"+{gap}" if gap > 0 else "=" if gap == 0 else f"{gap}"
        time_str = f"{elapsed*1000:5.0f}ms" if elapsed < 1 else f"{elapsed:5.2f}s"
        print(f"{c.name:<38} {len(boxes):>4} {lb:>3} "
              f"{result.num_pallets:>4} {result.total_volume_utilisation:>5.1%} "
              f"{len(result.unpacked):>4} {time_str:>7}  gap {gap_str}")
        if errors:
            for e in errors[:2]:
                print(f"    ! {e}")
        results.append((c, result, elapsed, errors, boxes, lb))
    print()

    # Detailed analysis per case
    for c, result, elapsed, errors, boxes, lb in results:
        gap = result.num_pallets - lb
        print(f"### {c.name}")
        print(f"  Description: {c.description}")
        print(f"  Got: {result.num_pallets} pallets, "
              f"{result.total_volume_utilisation:.1%} util, "
              f"{len(result.unpacked)} unpacked")
        print(f"  Theoretical LB: {lb} pallets (gap: +{gap})")
        if gap == 0 and not result.unpacked:
            verdict = "PASS — algorithm matches lower bound."
        elif gap > 0:
            verdict = f"FAILURE — uses {gap} more pallet(s) than necessary."
        else:
            verdict = "?"
        print(f"  Verdict: {verdict}")
        print(f"  Better approach: {c.known_better}")
        print(f"  Literature reference: {c.fix_reference}")
        print()


if __name__ == "__main__":
    main()
