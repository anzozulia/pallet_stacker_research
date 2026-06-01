"""reverify_realworld_combos — adversarial COMBINED-constraint real-world cases.

Per-feature probes test one constraint at a time. This probe forces 5+ hard
constraints to interact on the SAME instance and proves they all hold together:

  per-SKU weights  +  fragility (max_load_on_top=0 for some SKUs)
  + Box.group co-location  +  multi-pallet (max_pallets 3..8)
  + pallet weight cap  +  default support_ratio=0.8 (centroid + load-bearing on)

All spatial dims are POSITIVE INTEGERS and brkga_pack_v35 is called with
validate_input=True so the boundary GATE is exercised on every case.

For each case we assert, independently of the engine:
  V  validate(result, pallet, config) == []   (bounds/overlap/weight/support/
                                                rotation/load-bearing oracle)
  C  conservation: placed + unpacked == n_input  (no box invented/lost)
  P  max_pallets honored: len(result.pallets) <= max_pallets
  G  no group split: every non-None group lives on exactly one pallet
  W  weight cap respected: per-pallet sum(weight) <= pallet.max_weight  (belt-
     and-suspenders on top of validate's #3)
  F  fragile uncrushed: every box with max_load_on_top==0 carries ZERO load
     (recomputed here independently of validate's #6)

Case 7 is the headline adversarial mix: groups are made HEAVY so the pallet
weight cap binds and FORCES groups onto different pallets, while half the SKUs
are fragile. Co-location must still hold (a group never splits), the cap must
never be exceeded, and no fragile box may be crushed — all at once.
"""
from __future__ import annotations
import sys
sys.path.insert(0, '.')
from collections import defaultdict

from pallet_packer import (
    Box, Pallet, PackerConfig, THIS_SIDE_UP, ALL_ROTATIONS, NO_ROTATION,
    validate, validate_packing_input, check_packing_input, PackingInputError,
)
from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit

EPS = 1e-6
PASS = 0
FAIL = 0
FAILURES = []


def check(label, cond, detail=""):
    global PASS, FAIL
    ok = bool(cond)
    PASS += ok
    FAIL += (not ok)
    tag = "PASS" if ok else "FAIL"
    line = f"    [{tag}] {label}" + ((": " + detail) if detail else "")
    print(line)
    if not ok:
        FAILURES.append(label + ((" :: " + detail) if detail else ""))


def all_integer_dims(boxes, pallet):
    """True iff every spatial dim is an integer value (gate precondition)."""
    def isint(v):
        return float(v) == int(v)
    for b in boxes:
        if not (isint(b.length) and isint(b.width) and isint(b.height)):
            return False
    return isint(pallet.length) and isint(pallet.width) and isint(pallet.height)


def fragile_carried_load(st):
    """Independently recompute load resting on each fragile (mlot==0) box.

    Mirrors validate's per-supporter, contact-area-weighted recursion but
    recomputed here so a bug shared between engine+validate can't hide.
    Returns dict box_id -> carried_kg for boxes whose max_load_on_top==0.
    """
    load_on = {id(p): 0.0 for p in st.placements}
    for placed in st.placements:
        sups = []
        for q in st.placements:
            if q is placed:
                continue
            if abs(placed.z - q.z2) > EPS:
                continue
            ox = max(0.0, min(placed.x2, q.x2) - max(placed.x, q.x))
            oy = max(0.0, min(placed.y2, q.y2) - max(placed.y, q.y))
            if ox * oy > EPS:
                sups.append((q, ox * oy))
        sup_area = sum(a for _, a in sups)
        if sup_area <= 0:
            continue
        for q, a in sups:
            load_on[id(q)] += placed.box.weight * (a / sup_area)
    out = {}
    for p in st.placements:
        if p.box.max_load_on_top == 0.0:
            out[p.box.id] = load_on[id(p)]
    return out


def assess(name, boxes, pallet, config, max_pallets, time_limit_s=14):
    print(f"\n=== {name} (n={len(boxes)}, max_pallets={max_pallets}, "
          f"cap={pallet.max_weight}, support_ratio={config.support_ratio}) ===")

    # --- exercise the input gate explicitly (should be CLEAN; dims integer) ---
    gate_problems = validate_packing_input(boxes, pallet)
    check("input gate clean (well-formed)", gate_problems == [],
          str(gate_problems[:3]))
    check("dims are all integers", all_integer_dims(boxes, pallet))

    # --- solve with validate_input=True so the GATE runs inside the solver ---
    try:
        r = brkga_pack_v35(
            boxes, pallet, config,
            time_limit_s=time_limit_s, max_pallets=max_pallets, seed=42,
            population_size=300, n_populations=3, patience=150, n_modes=6,
            validate_input=True, verbose=False,
        )
    except PackingInputError as e:
        check("solver did not raise on valid input", False, str(e.problems[:3]))
        return
    except Exception as e:  # noqa
        check("solver did not crash", False, f"{type(e).__name__}: {e}")
        return

    placed = sum(len(st.placements) for st in r.pallets)
    n_unp = len(r.unpacked)

    # group -> set of pallet indices it appears on
    group_pallets = defaultdict(set)
    for pi, st in enumerate(r.pallets):
        for p in st.placements:
            g = getattr(p.box, "group", None)
            if g is not None:
                group_pallets[g].add(pi)
    splits = {g: sorted(ps) for g, ps in group_pallets.items() if len(ps) > 1}

    # weight per pallet
    overweight = []
    for st in r.pallets:
        w = sum(p.box.weight for p in st.placements)
        if w > pallet.max_weight + 1e-3:
            overweight.append((st.pallet_id, w))

    # fragile load
    crushed = []
    for st in r.pallets:
        for bid, kg in fragile_carried_load(st).items():
            if kg > EPS:
                crushed.append((st.pallet_id, bid, round(kg, 3)))

    errs = validate(r, pallet, config)

    print(f"    -> pallets={len(r.pallets)} placed={placed} unpacked={n_unp} "
          f"validator_errs={len(errs)} group_splits={len(splits)} "
          f"overweight={len(overweight)} crushed_fragile={len(crushed)}")
    if errs:
        print(f"       first errs: {errs[:4]}")

    check("V validator clean", errs == [], str(errs[:3]))
    check("C conservation (placed+unpacked==input)",
          placed + n_unp == len(boxes),
          f"{placed}+{n_unp}!={len(boxes)}")
    check("P max_pallets honored",
          len(r.pallets) <= max_pallets,
          f"{len(r.pallets)}>{max_pallets}")
    check("G no group split", len(splits) == 0, str(splits))
    check("W weight cap respected (per pallet)",
          len(overweight) == 0, str(overweight))
    check("F fragile uncrushed (mlot=0 carries 0)",
          len(crushed) == 0, str(crushed[:4]))

    # --- E: efficiency sanity vs the SAME boxes WITHOUT groups -------------
    # The group path must not pack DRAMATICALLY worse than the group-free
    # solve under the same pallet/cap budget. A weight-cap-blind bundle
    # assigner crams heavy low-volume groups onto one volume-feasible pallet,
    # then drops everything over the cap to `unpacked` while allowed pallets
    # sit EMPTY. Flag when grouped places much less AND pallets are unused.
    if any(getattr(b, "group", None) is not None for b in boxes):
        ung = [Box(id=b.id, length=b.length, width=b.width, height=b.height,
                   weight=b.weight, allowed_rotations=b.allowed_rotations,
                   max_load_on_top=b.max_load_on_top, group=None)
               for b in boxes]
        r2 = brkga_pack_v35(ung, pallet, config, time_limit_s=time_limit_s,
                            max_pallets=max_pallets, seed=42,
                            population_size=300, n_populations=3, patience=150,
                            n_modes=6, validate_input=True, verbose=False)
        placed_ung = sum(len(st.placements) for st in r2.pallets)
        lost_extra = placed_ung - placed
        pallets_unused = max_pallets - len(r.pallets)
        print(f"    -> [E] ungrouped placed={placed_ung} (grouped={placed}); "
              f"grouped dropped {lost_extra} MORE while {pallets_unused} "
              f"pallets unused")
        # Defect signature: group path leaves >=2 pallets empty AND drops
        # >=10% more boxes than the group-free solve under the same budget.
        defect = (pallets_unused >= 2 and lost_extra >= max(8, int(0.1 * len(boxes))))
        check("E group path not weight-cap-blind (no big drop vs ungrouped)",
              not defect,
              f"grouped lost {lost_extra} extra boxes, {pallets_unused} "
              f"pallets left empty")
    return r


# ---------------------------------------------------------------------------
# Cargo builders. All integer spatial dims; fractional WEIGHTS are allowed.
# ---------------------------------------------------------------------------
def case1_mixed_3pl():
    """3PL mixed order: 4 SKUs, 2 fragile, 3 customer groups, light cap."""
    boxes = []
    # group GA: sturdy mediums
    for i in range(20):
        boxes.append(Box(id=f"GA-M{i:02d}", length=300, width=200, height=150,
                         weight=4.0, allowed_rotations=ALL_ROTATIONS,
                         max_load_on_top=30.0, group="GA"))
    # group GA: fragile flats (mlot=0)
    for i in range(10):
        boxes.append(Box(id=f"GA-F{i:02d}", length=250, width=200, height=80,
                         weight=2.0, allowed_rotations=THIS_SIDE_UP,
                         max_load_on_top=0.0, group="GA"))
    # group GB: heavy smalls
    for i in range(25):
        boxes.append(Box(id=f"GB-H{i:02d}", length=200, width=150, height=150,
                         weight=8.0, allowed_rotations=ALL_ROTATIONS,
                         max_load_on_top=50.0, group="GB"))
    # group GC: fragile electronics (mlot=0)
    for i in range(15):
        boxes.append(Box(id=f"GC-E{i:02d}", length=180, width=160, height=120,
                         weight=1.5, allowed_rotations=THIS_SIDE_UP,
                         max_load_on_top=0.0, group="GC"))
    # ungrouped filler
    for i in range(20):
        boxes.append(Box(id=f"U{i:02d}", length=220, width=180, height=140,
                         weight=3.0, allowed_rotations=ALL_ROTATIONS,
                         max_load_on_top=25.0))
    pal = Pallet(length=1200, width=1000, height=1500, max_weight=350)
    return "C1 3PL mixed (4 SKU/2 fragile/3 grp)", boxes, pal, PackerConfig(), 6


def case2_pharma_grouped():
    """Pharma: many fragile SKUs split into shipment groups, tight cap."""
    boxes = []
    for gi, g in enumerate(["RX-A", "RX-B", "RX-C", "RX-D"]):
        for i in range(30):
            frag = (i % 3 == 0)  # 1/3 fragile
            boxes.append(Box(id=f"{g}-{i:02d}", length=180, width=120,
                             height=100, weight=2.5,
                             allowed_rotations=THIS_SIDE_UP,
                             max_load_on_top=0.0 if frag else 12.0, group=g))
    pal = Pallet(length=1200, width=1000, height=1500, max_weight=300)
    return "C2 pharma grouped (120, 4 grp, 1/3 fragile)", boxes, pal, PackerConfig(), 8


def case3_beverage_heavy():
    """Heavy beverage cases, weight cap binds hard, grouped by customer."""
    boxes = []
    for g in ["CUST1", "CUST2", "CUST3", "CUST4", "CUST5"]:
        for i in range(24):
            # heavy uniform cases; sturdy (can stack) but heavy
            boxes.append(Box(id=f"{g}-B{i:02d}", length=400, width=300,
                             height=300, weight=15.0,
                             allowed_rotations=THIS_SIDE_UP,
                             max_load_on_top=45.0, group=g))
    pal = Pallet(length=1200, width=1000, height=1500, max_weight=500)
    return "C3 beverage heavy (120, 5 grp, cap binds)", boxes, pal, PackerConfig(), 8


def case4_furniture_fragile():
    """Mixed appliance order: rotation-locked, tall fragile flats, groups."""
    boxes = []
    for g in ["ORD-A", "ORD-B", "ORD-C"]:
        # fragile TV flats (mlot=0)
        for i in range(8):
            boxes.append(Box(id=f"{g}-TV{i:02d}", length=900, width=500,
                             height=120, weight=12.0,
                             allowed_rotations=THIS_SIDE_UP,
                             max_load_on_top=0.0, group=g))
        # sturdy bases
        for i in range(12):
            boxes.append(Box(id=f"{g}-BASE{i:02d}", length=600, width=400,
                             height=400, weight=10.0,
                             allowed_rotations=THIS_SIDE_UP,
                             max_load_on_top=60.0, group=g))
    # ungrouped accessories
    for i in range(30):
        boxes.append(Box(id=f"ACC{i:02d}", length=300, width=250, height=200,
                         weight=2.0, allowed_rotations=ALL_ROTATIONS,
                         max_load_on_top=15.0))
    pal = Pallet(length=1200, width=1000, height=1500, max_weight=400)
    return "C4 furniture+fragile (90, 3 grp, rot-lock)", boxes, pal, PackerConfig(), 5


def case5_large_scale():
    """300-box stress: 6 SKUs, alternating fragile, 6 groups, mid cap."""
    boxes = []
    skus = [
        ("A", 250, 200, 150, 3.0, 20.0),
        ("B", 200, 180, 120, 2.0, 0.0),    # fragile
        ("C", 300, 250, 200, 5.0, 40.0),
        ("D", 160, 140, 100, 1.2, 0.0),    # fragile
        ("E", 350, 300, 250, 7.0, 50.0),
        ("F", 220, 200, 160, 2.8, 18.0),
    ]
    for k in range(300):
        s = skus[k % len(skus)]
        name, l, w, h, wt, mlot = s
        g = f"G{k % 6}"
        rot = THIS_SIDE_UP if mlot == 0.0 else ALL_ROTATIONS
        boxes.append(Box(id=f"{name}-{k:03d}", length=l, width=w, height=h,
                         weight=wt, allowed_rotations=rot,
                         max_load_on_top=mlot, group=g))
    pal = Pallet(length=1200, width=1000, height=1500, max_weight=450)
    return "C5 large-scale (300, 6 SKU/grp, alt-fragile)", boxes, pal, PackerConfig(), 8


def case6_overhang_off_small_pallet():
    """Small pallet, low cap, overhang OFF: support_ratio=0.8 must bind too."""
    boxes = []
    for g in ["X", "Y", "Z"]:
        for i in range(28):
            frag = (i % 4 == 0)
            boxes.append(Box(id=f"{g}-{i:02d}", length=240, width=160,
                             height=140, weight=3.5,
                             allowed_rotations=ALL_ROTATIONS if not frag
                             else THIS_SIDE_UP,
                             max_load_on_top=0.0 if frag else 22.0, group=g))
    pal = Pallet(length=1000, width=800, height=1200, max_weight=200)
    return "C6 small-pallet low-cap (84, 3 grp)", boxes, pal, PackerConfig(), 6


def case7_cap_forces_groups_apart():
    """HEADLINE adversarial: heavy groups + binding weight cap + fragile.

    Each group is ~heavy enough that two groups CANNOT coexist on one pallet
    under the weight cap, so the cap FORCES groups onto separate pallets while
    co-location must still hold (no group ever spans 2 pallets). Half the SKUs
    are fragile. We deliberately give enough pallets that everything CAN pack,
    so the only way to fail is a real constraint interaction bug.
    """
    boxes = []
    # 5 groups, each ~ 18 boxes. Per-group weight chosen so 1 group ≈ 180 kg,
    # two groups (360) would blow a 250 kg cap -> cap forces 1 group/pallet.
    for gi, g in enumerate(["H1", "H2", "H3", "H4", "H5"]):
        for i in range(18):
            fragile = (i % 2 == 0)          # half fragile within each group
            boxes.append(Box(
                id=f"{g}-{i:02d}", length=300, width=250, height=200,
                weight=10.0,
                allowed_rotations=THIS_SIDE_UP if fragile else ALL_ROTATIONS,
                max_load_on_top=0.0 if fragile else 35.0, group=g))
    # group weight = 18*10 = 180; two groups = 360 > 250 cap -> must separate.
    pal = Pallet(length=1200, width=1000, height=1500, max_weight=250)
    return ("C7 cap-forces-groups-apart (90, 5 heavy grp, 50% fragile)",
            boxes, pal, PackerConfig(), 8)


def main():
    warmup_jit()
    builders = [case1_mixed_3pl, case2_pharma_grouped, case3_beverage_heavy,
                case4_furniture_fragile, case5_large_scale,
                case6_overhang_off_small_pallet, case7_cap_forces_groups_apart]
    for b in builders:
        name, boxes, pal, cfg, mp = b()
        tl = 18 if len(boxes) >= 250 else 12
        assess(name, boxes, pal, cfg, mp, time_limit_s=tl)

    print(f"\n========== {PASS} passed, {FAIL} failed ==========")
    if FAILURES:
        print("FAILURES:")
        for f in FAILURES:
            print("  -", f)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
