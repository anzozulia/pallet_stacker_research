"""Test Box.group co-location. Run in Docker (pure Python, no rebuild).

Guarantee: boxes sharing a non-None group never span >1 pallet. Verifies:
  1. IND6 (LTL groupage) — no group split, validator-clean, conservation.
  2. Constructed multi-pallet case (3 groups too big for one pallet) — groups
     forced apart by capacity must each still be wholly on one pallet.
  3. Ungrouped workload unaffected (no behaviour change).
"""
from __future__ import annotations
import sys
sys.path.insert(0, '.')
from collections import defaultdict
from pallet_packer import Box, Pallet, PackerConfig, THIS_SIDE_UP, ALL_ROTATIONS, validate
from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit
from benchmarks.industry import cases as industry_cases


def group_report(result, n_input):
    """Return (max_pallets_any_group_spans, splits, placed, conservation_ok)."""
    group_pallets = defaultdict(set)
    placed = 0
    for pi, st in enumerate(result.pallets):
        for p in st.placements:
            placed += 1
            g = getattr(p.box, 'group', None)
            if g is not None:
                group_pallets[g].add(pi)
    splits = {g: sorted(ps) for g, ps in group_pallets.items() if len(ps) > 1}
    conservation = (placed + len(result.unpacked) == n_input)
    return splits, placed, conservation


PASS = FAIL = 0


def check(label, cond, detail=""):
    global PASS, FAIL
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}{(': ' + detail) if detail else ''}")
    PASS += cond
    FAIL += (not cond)


def main():
    warmup_jit()

    print("=== 1. IND6 LTL groupage (3 customer groups) ===")
    ind6 = [c for c in industry_cases() if c.name.startswith("IND6")][0]
    r = brkga_pack_v35(ind6.boxes, ind6.pallet, ind6.config, time_limit_s=12,
                       max_pallets=10, population_size=300, n_populations=3,
                       patience=150, seed=42, n_modes=6, verbose=False)
    splits, placed, cons = group_report(r, len(ind6.boxes))
    errs = validate(r, ind6.pallet, ind6.config)
    print(f"    pallets={len(r.pallets)} placed={placed} unpkd={len(r.unpacked)} "
          f"errs={len(errs)} group_splits={splits}")
    check("IND6 no group split", len(splits) == 0)
    check("IND6 validator clean", len(errs) == 0)
    check("IND6 conservation", cons)

    print("\n=== 2. Constructed: 3 groups of 18 (need multiple pallets) ===")
    boxes = []
    for gi, gname in enumerate(["GA", "GB", "GC"]):
        for k in range(18):
            boxes.append(Box(id=f"{gname}-{k:02d}", length=400, width=350, height=300,
                             weight=5.0, allowed_rotations=THIS_SIDE_UP,
                             max_load_on_top=40.0, group=gname))
    pal = Pallet(length=1200, width=1000, height=1500, max_weight=2000)
    for cap in [2, 3, 5]:
        r = brkga_pack_v35(boxes, pal, PackerConfig(), time_limit_s=12, max_pallets=cap,
                           population_size=300, n_populations=3, patience=150,
                           seed=42, n_modes=6, verbose=False)
        splits, placed, cons = group_report(r, len(boxes))
        errs = validate(r, pal, PackerConfig())
        print(f"    cap={cap}: pallets={len(r.pallets)} placed={placed} "
              f"unpkd={len(r.unpacked)} errs={len(errs)} splits={splits}")
        check(f"cap={cap}: no group split", len(splits) == 0,
              "" if not splits else str(splits))
        check(f"cap={cap}: validator clean", len(errs) == 0)
        check(f"cap={cap}: conservation", cons)

    print("\n=== 3. Ungrouped workload still works (no group dispatch) ===")
    ung = [Box(id=f"U{i}", length=200, width=150, height=120, weight=1.0,
               allowed_rotations=ALL_ROTATIONS) for i in range(60)]
    r = brkga_pack_v35(ung, pal, PackerConfig(support_ratio=0.0), time_limit_s=8,
                       max_pallets=5, population_size=300, n_populations=3,
                       patience=150, seed=42, n_modes=6, verbose=False)
    _, placed, cons = group_report(r, len(ung))
    errs = validate(r, pal, PackerConfig(support_ratio=0.0))
    print(f"    pallets={len(r.pallets)} placed={placed} unpkd={len(r.unpacked)} errs={len(errs)}")
    check("ungrouped conservation", cons)
    check("ungrouped validator clean", len(errs) == 0)

    print(f"\n=== {PASS} passed, {FAIL} failed ===")
    return 0 if FAIL == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
