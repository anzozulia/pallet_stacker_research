"""reverify_fixes-hold: adversarial cross-checks supplementing the 5 canned probes.

A. Geometric SKU partition is BIT-IDENTICAL to the baseline key (no mlot/rfs).
   - For weightless/geometric boxes (mlot=inf, rfs=False), the new 7-tuple key
     must induce the EXACT same partition as the old 5-tuple key. Verified by
     reconstructing both keys and comparing the partitions on real BR data +
     synthetic weightless workloads.
B. Part A actually SEPARATES fragile/sturdy same-size boxes (the bug it fixes):
   - identical dims+weight+rot, mlot 0 vs 40 must be DIFFERENT skus now.
   - geometric default (mlot=inf for all) must be ONE sku (no over-fragmentation).
C. precompute has_constraints flag for weightless == False (D2 / v2-seed guard).
"""
from __future__ import annotations
import sys, math
sys.path.insert(0, '.')
import numpy as np
from pallet_packer import Box, Pallet, PackerConfig, THIS_SIDE_UP, ALL_ROTATIONS
from pallet_packer._brkga_core.precompute import (
    precompute_box_dims_and_sku, precompute_constraint_arrays)
from benchmarks.br import parse_thpack

PASS = FAIL = 0
def check(label, cond, detail=""):
    global PASS, FAIL
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}{(': ' + detail) if detail else ''}")
    PASS += bool(cond); FAIL += (not cond)


def old_key(b):
    """Baseline (cead45d) 5-tuple SKU key — NO mlot/rfs."""
    rot_key = tuple(sorted(r.name for r in b.allowed_rotations))
    return (round(b.length, 6), round(b.width, 6), round(b.height, 6),
            round(b.weight, 6), rot_key)


def partition_from_keys(keyfn, boxes):
    seen, ids = {}, []
    for b in boxes:
        k = keyfn(b)
        if k not in seen:
            seen[k] = len(seen)
        ids.append(seen[k])
    return ids


def main():
    print("=== A. Geometric partition bit-identical (old 5-tuple == new key) ===")
    # New partition is what precompute_box_dims_and_sku returns (sku_id_per_box).
    insts = parse_thpack('benchmarks/data/thpack1.txt', set_id='BR1')[:5]
    all_ident = True
    for inst in insts:
        _, _, sku_new = precompute_box_dims_and_sku(inst.boxes)
        sku_old = partition_from_keys(old_key, inst.boxes)
        # Compare as PARTITIONS (label-invariant): same grouping structure.
        same = _same_partition(list(sku_new), sku_old)
        all_ident &= same
        if not same:
            print(f"    BR1#{inst.instance_id}: MISMATCH n_new={len(set(sku_new))} "
                  f"n_old={len(set(sku_old))}")
    check("BR1(5) geometric partition identical to baseline key", all_ident)

    # synthetic weightless workloads (mlot=inf, rfs=False defaults)
    rng = np.random.default_rng(7)
    wok = True
    for _ in range(20):
        N = int(rng.integers(30, 200))
        skus = [(int(rng.integers(100, 500)), int(rng.integers(100, 400)),
                 int(rng.integers(80, 350))) for _ in range(int(rng.integers(2, 10)))]
        boxes = [Box(id=f'B{i}', length=skus[i % len(skus)][0],
                     width=skus[i % len(skus)][1], height=skus[i % len(skus)][2],
                     weight=0.0, allowed_rotations=ALL_ROTATIONS) for i in range(N)]
        _, _, sku_new = precompute_box_dims_and_sku(boxes)
        sku_old = partition_from_keys(old_key, boxes)
        wok &= _same_partition(list(sku_new), sku_old)
    check("20 synthetic weightless workloads partition identical", wok)

    print("\n=== B. Part A: fragile/sturdy same-size boxes now SEPARATE ===")
    # Identical dims+weight+rot, mlot differs -> must be 2 skus.
    frag = Box(id='f', length=300, width=200, height=150, weight=5.0,
               allowed_rotations=THIS_SIDE_UP, max_load_on_top=0.0)
    sturdy = Box(id='s', length=300, width=200, height=150, weight=5.0,
                 allowed_rotations=THIS_SIDE_UP, max_load_on_top=40.0)
    _, _, sku2 = precompute_box_dims_and_sku([frag, sturdy])
    check("fragile(mlot=0) vs sturdy(mlot=40) -> 2 distinct skus",
          sku2[0] != sku2[1], f"sku_ids={list(sku2)}")
    # old key would have MERGED them (the bug):
    check("baseline key WOULD merge them (confirms this is the fixed bug)",
          old_key(frag) == old_key(sturdy))

    # rfs differs -> separate
    a = Box(id='a', length=300, width=200, height=150, weight=5.0,
            allowed_rotations=THIS_SIDE_UP, requires_full_support=True)
    b = Box(id='b', length=300, width=200, height=150, weight=5.0,
            allowed_rotations=THIS_SIDE_UP, requires_full_support=False)
    _, _, sku_rfs = precompute_box_dims_and_sku([a, b])
    check("rfs True vs False same-size -> 2 distinct skus", sku_rfs[0] != sku_rfs[1])

    # No over-fragmentation: all-default (mlot=inf, rfs=False) identical dims -> 1 sku
    same = [Box(id=f'x{i}', length=300, width=200, height=150, weight=5.0,
                allowed_rotations=THIS_SIDE_UP) for i in range(10)]
    _, _, sku_same = precompute_box_dims_and_sku(same)
    check("10 identical default boxes -> exactly 1 sku (no over-fragment)",
          len(set(sku_same)) == 1, f"n_sku={len(set(sku_same))}")

    # adversarial: two huge-but-finite different mlot -> separate; both inf -> merge
    m1 = Box(id='m1', length=300, width=200, height=150, weight=5.0,
             allowed_rotations=THIS_SIDE_UP, max_load_on_top=1e9)
    m2 = Box(id='m2', length=300, width=200, height=150, weight=5.0,
             allowed_rotations=THIS_SIDE_UP, max_load_on_top=2e9)
    _, _, sm = precompute_box_dims_and_sku([m1, m2])
    check("distinct large finite mlot (1e9 vs 2e9) -> 2 skus", sm[0] != sm[1])

    print("\n=== C. Weightless has_constraints flag == False (v2-seed guard) ===")
    boxes = [Box(id=f'W{i}', length=300, width=200, height=150, weight=0.0,
                 allowed_rotations=ALL_ROTATIONS) for i in range(40)]
    pal = Pallet(length=1200, width=1000, height=1500)
    *_, hc = precompute_constraint_arrays(boxes, pal)
    check("weightless workload has_constraints == False", hc is False, f"hc={hc}")

    print(f"\n=== {PASS} passed, {FAIL} failed ===")
    return 0 if FAIL == 0 else 1


def _same_partition(a, b):
    """True iff a and b induce the same grouping (label-invariant)."""
    if len(a) != len(b):
        return False
    from collections import defaultdict
    ga, gb = defaultdict(set), defaultdict(set)
    for i, (x, y) in enumerate(zip(a, b)):
        ga[x].add(i); gb[y].add(i)
    return sorted(map(frozenset, ga.values())) == sorted(map(frozenset, gb.values()))


if __name__ == '__main__':
    raise SystemExit(main())
