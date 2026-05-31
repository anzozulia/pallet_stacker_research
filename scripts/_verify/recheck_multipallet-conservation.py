"""Counter-probe for the 'Box.group co-location silently never enforced' claim.

Re-derives independently. Four sub-tests:

  T1  SOURCE FACT: confirm precompute_box_dims_and_sku / precompute_constraint_arrays
      do not encode group. Build two box-lists identical except for group labels;
      assert the precompute outputs (dims, sku_id, constraint arrays,
      has_constraints) are byte-identical. If group were encoded anywhere it would
      change them. (Inert == evidence FOR the defect.)

  T2  RUNTIME SPLIT: K1-style case where a single group spans 2 pallets even when
      max_pallets is UNLIMITED (capacity NOT binding). A split NOT forced by cap.

  T3  GROUP-IS-INERT: solve the SAME geometry twice — once with every box grouped,
      once with group=None — at a fixed seed. Bit-identical pallet assignment
      PROVES group has zero effect on the solver.

  T4  validate() blind: confirm validate() returns [] for the group split.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from pallet_packer import Box, Pallet, PackerConfig, THIS_SIDE_UP, validate
from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit
from pallet_packer._brkga_core.precompute import (
    precompute_box_dims_and_sku, precompute_constraint_arrays,
)


def make_boxes(group_fn):
    boxes = []
    for g, n in [("GA", 18), ("GB", 18), ("GC", 18)]:
        for i in range(n):
            boxes.append(Box(id=f"{g}-{i:02d}", length=400, width=350, height=300,
                             weight=3.0, allowed_rotations=THIS_SIDE_UP,
                             max_load_on_top=20.0, group=group_fn(g, i)))
    return boxes


def pallet_index_of_groups(result, input_boxes):
    box_pallet = {}
    for pi, st in enumerate(result.pallets):
        for p in st.placements:
            box_pallet[id(p.box)] = pi
    unpacked = {id(b) for b in result.unpacked}
    groups = {}
    for b in input_boxes:
        if b.group is not None:
            groups.setdefault(b.group, []).append(b)
    out = {}
    for label, members in groups.items():
        seen, n_unp = set(), 0
        for b in members:
            if id(b) in box_pallet:
                seen.add(box_pallet[id(b)])
            elif id(b) in unpacked:
                n_unp += 1
        out[label] = (sorted(seen), n_unp, len(members))
    return out


def solve(boxes, pallet, config, max_pallets, seed=42, t=4.0):
    mp = max_pallets if max_pallets is not None else 1000
    return brkga_pack_v35(
        boxes, pallet, config, time_limit_s=t, max_pallets=mp,
        population_size=120, n_populations=2, patience=120,
        local_search_budget_s=1.0, seed=seed, verbose=False, n_modes=6,
    )


def pallet_assignment_signature(result):
    rows = []
    for pi, st in enumerate(result.pallets):
        for p in st.placements:
            rows.append((p.box.id, pi, round(p.x, 3), round(p.y, 3), round(p.z, 3)))
    for b in result.unpacked:
        rows.append((b.id, -1, None, None, None))
    rows.sort(key=lambda r: (r[0], r[1]))
    return tuple(rows)


def main():
    warmup_jit()
    pallet = Pallet(length=1200, width=1000, height=1500, max_weight=1000)
    cfg = PackerConfig()
    print("=" * 78)
    print("RECHECK: Box.group co-location enforcement")
    print("=" * 78)

    # ---- T1: precompute is group-blind -------------------------------------
    print("\n[T1] precompute encodes group?  (group-labelled vs group=None)")
    b_grouped = make_boxes(lambda g, i: g)
    b_none = make_boxes(lambda g, i: None)
    d1 = precompute_box_dims_and_sku(b_grouped)
    d2 = precompute_box_dims_and_sku(b_none)
    c1 = precompute_constraint_arrays(b_grouped, pallet)
    c2 = precompute_constraint_arrays(b_none, pallet)
    dims_same = np.array_equal(d1[1], d2[1])
    sku_same = np.array_equal(d1[2], d2[2])
    nrots_same = np.array_equal(d1[0], d2[0])
    cstr_same = (np.array_equal(c1[0], c2[0]) and np.array_equal(c1[1], c2[1])
                 and np.array_equal(c1[2], c2[2]) and c1[3] == c2[3]
                 and c1[4] == c2[4])
    t1_inert = dims_same and sku_same and nrots_same and cstr_same
    print(f"     dims identical          : {dims_same}")
    print(f"     sku_id_per_box identical: {sku_same}  (group NOT in SKU key)")
    print(f"     n_rots identical        : {nrots_same}")
    print(f"     constraint arrays same  : {cstr_same}  has_constraints={c1[4]}")
    print(f"     T1 {'PASS-INERT (group not encoded)' if t1_inert else 'group DOES change arrays'}")

    # ---- T2: group split with UNLIMITED pallets (cap not binding) ----------
    print("\n[T2] group split with max_pallets=UNLIMITED (cap not binding)")
    boxes = make_boxes(lambda g, i: g)
    r = solve(boxes, pallet, cfg, max_pallets=None)
    gi = pallet_index_of_groups(r, boxes)
    n_pallets = len(r.pallets)
    splits = {lab: v for lab, v in gi.items() if len(v[0]) > 1}
    print(f"     pallets opened = {n_pallets}")
    for lab, (pidx, n_unp, n_mem) in sorted(gi.items()):
        flag = "  <<< SPLIT" if len(pidx) > 1 else ""
        print(f"       group {lab}: pallets {pidx} unpacked {n_unp}/{n_mem}{flag}")
    t2_split = len(splits) > 0
    print(f"     T2 {'SPLIT OBSERVED (unlimited cap, not forced)' if t2_split else 'no split'}")

    # ---- T3: identical solve with group vs no-group => bit-identical? ------
    print("\n[T3] solve grouped vs ungrouped at same seed (bit-identity)")
    b_g = make_boxes(lambda g, i: g)
    b_n = make_boxes(lambda g, i: None)
    r_g = solve(b_g, pallet, cfg, max_pallets=None, seed=7)
    r_n = solve(b_n, pallet, cfg, max_pallets=None, seed=7)
    sig_g = pallet_assignment_signature(r_g)
    sig_n = pallet_assignment_signature(r_n)
    t3_identical = (sig_g == sig_n)
    print(f"     grouped pallets={len(r_g.pallets)} placed={sum(len(s.placements) for s in r_g.pallets)}")
    print(f"     ungroup pallets={len(r_n.pallets)} placed={sum(len(s.placements) for s in r_n.pallets)}")
    print(f"     assignment signatures identical: {t3_identical}")
    print(f"     T3 {'GROUP IS INERT (bit-identical => zero effect)' if t3_identical else 'group changed solve'}")

    # ---- T4: validate() blind to the split --------------------------------
    print("\n[T4] validate() sees the T2 split?")
    verrs = validate(r, pallet, cfg)
    t4_blind = (len(verrs) == 0)
    print(f"     validate() errors = {len(verrs)}  (sample: {verrs[:2]})")
    print(f"     T4 {'validate() BLIND to group split' if t4_blind else 'validate() flagged it'}")

    # ---- Verdict synthesis -------------------------------------------------
    print("\n" + "=" * 78)
    defect = t1_inert and t2_split and t3_identical and t4_blind
    print("SYNTHESIS:")
    print(f"  group not encoded in precompute (T1)      : {t1_inert}")
    print(f"  real split w/ unlimited pallets (T2)      : {t2_split}")
    print(f"  group bit-identical => inert in solve (T3): {t3_identical}")
    print(f"  validate() blind to split (T4)            : {t4_blind}")
    print(f"\n  => Box.group co-location {'NOT ENFORCED (defect reproduced)' if defect else 'inconclusive/enforced'}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
