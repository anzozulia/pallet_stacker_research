"""Verify multi-pallet correctness + BOX CONSERVATION (adversarial).

Mandate: for many instances (industry + constructed), across max_pallets in
{1,2,3,4,5,unlimited}, assert:

  (C) CONSERVATION: every input Box appears EXACTLY ONCE in
      (placements across all pallets) UNION (unpacked).
      Checked two ways:
        - by python identity id() : no duplicate id, no missing id
        - by id-string count       : multiset of box.id matches input multiset
  (P) MAX_PALLETS: len(result.pallets) <= max_pallets (when capped).
  (G) GROUP: boxes with the same Box.group must all land on the same pallet
      (or all be unpacked). IND6 (LTL groupage) is the canonical group case.
  (V) VALIDITY: validate() returns zero errors per result.

This probe is adversarial: it specifically constructs group-laden instances
and pushes max_pallets down to force the solver to split / drop, then checks
whether conservation and group co-location actually hold.

Run:
    docker run --rm -e OMP_NUM_THREADS=1 -e PYTHONHASHSEED=0 \
        -v "$(pwd)":/app -w /app pallet-packer:dev \
        python scripts/_verify/verify_multipallet_conservation.py
"""
from __future__ import annotations

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pallet_packer import Box, Pallet, PackerConfig, THIS_SIDE_UP, ALL_ROTATIONS, validate
from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit
from benchmarks.industry import cases as industry_cases


# ---------------------------------------------------------------------------
# Core invariant checks. Each returns a list of violation strings (empty=ok).
# ---------------------------------------------------------------------------
def check_conservation(input_boxes, result):
    """Every input box appears exactly once across placed+unpacked.

    Returns (errors, stats) where stats has placed/unpacked/total counts.
    """
    errors = []

    placed_ids = []  # python id()
    placed_keys = []  # box.id string
    for st in result.pallets:
        for p in st.placements:
            placed_ids.append(id(p.box))
            placed_keys.append(p.box.id)
    unpacked_ids = [id(b) for b in result.unpacked]
    unpacked_keys = [b.id for b in result.unpacked]

    in_ids = [id(b) for b in input_boxes]
    in_keys = [b.id for b in input_boxes]

    n_placed = len(placed_ids)
    n_unpacked = len(unpacked_ids)
    n_total = n_placed + n_unpacked
    n_in = len(in_ids)

    # --- count parity ---
    if n_total != n_in:
        errors.append(
            f"COUNT: placed({n_placed})+unpacked({n_unpacked})={n_total} "
            f"!= input({n_in})  delta={n_total - n_in}"
        )

    # --- by python identity: each input id present exactly once ---
    all_out_ids = Counter(placed_ids) + Counter(unpacked_ids)
    in_id_set = Counter(in_ids)
    dup_ids = {i: c for i, c in all_out_ids.items() if c > 1}
    if dup_ids:
        # map a few back to box.id for readability
        id2key = {id(b): b.id for b in input_boxes}
        sample = {id2key.get(i, "<alien>"): c for i, c in list(dup_ids.items())[:6]}
        errors.append(f"DUP(id): {len(dup_ids)} box object(s) appear >1x  e.g. {sample}")
    missing = [i for i in in_id_set if all_out_ids.get(i, 0) == 0]
    if missing:
        id2key = {id(b): b.id for b in input_boxes}
        errors.append(
            f"LOST(id): {len(missing)} input box object(s) absent from output "
            f"e.g. {[id2key[i] for i in missing[:6]]}"
        )
    alien = [i for i in all_out_ids if i not in in_id_set]
    if alien:
        errors.append(f"ALIEN(id): {len(alien)} output box object(s) not in input")

    # --- by box.id string multiset (catches clones / re-ids) ---
    out_key_mset = Counter(placed_keys) + Counter(unpacked_keys)
    in_key_mset = Counter(in_keys)
    if out_key_mset != in_key_mset:
        # report the specific diffs
        extra = out_key_mset - in_key_mset
        gone = in_key_mset - out_key_mset
        msg = []
        if extra:
            msg.append(f"extra={dict(list(extra.items())[:6])}")
        if gone:
            msg.append(f"missing={dict(list(gone.items())[:6])}")
        errors.append(f"MULTISET(id-string) mismatch: {' '.join(msg)}")

    # --- placed box appearing in unpacked too (cross dup) ---
    cross = set(placed_ids) & set(unpacked_ids)
    if cross:
        errors.append(f"CROSS-DUP: {len(cross)} box(es) both placed AND unpacked")

    stats = dict(placed=n_placed, unpacked=n_unpacked, total=n_total, input=n_in,
                 pallets=len(result.pallets))
    return errors, stats


def check_max_pallets(result, max_pallets):
    if max_pallets is None:
        return []
    if len(result.pallets) > max_pallets:
        return [f"MAX_PALLETS: opened {len(result.pallets)} > cap {max_pallets}"]
    return []


def check_groups(input_boxes, result):
    """Boxes sharing a non-None Box.group must all sit on ONE pallet.

    Determine each placed box's pallet index, then ensure every group maps to
    at most one pallet index. (Unpacked group members are tolerated only if the
    WHOLE group is unpacked; a group split between a pallet and unpacked is a
    violation — the group did not stay together on its destination pallet.)
    """
    errors = []
    # pallet index per box id-object
    box_pallet = {}  # id(box) -> pallet index
    for pi, st in enumerate(result.pallets):
        for p in st.placements:
            box_pallet[id(p.box)] = pi
    unpacked_ids = {id(b) for b in result.unpacked}

    # collect groups
    groups = {}  # group label -> list of boxes
    for b in input_boxes:
        if b.group is not None:
            groups.setdefault(b.group, []).append(b)

    n_groups = len(groups)
    for label, members in groups.items():
        pallets_seen = set()
        n_unpacked = 0
        for b in members:
            if id(b) in box_pallet:
                pallets_seen.add(box_pallet[id(b)])
            elif id(b) in unpacked_ids:
                n_unpacked += 1
        # Violation cases:
        #  - members on >1 distinct pallet  => split across pallets
        #  - members partly placed and partly unpacked => split placed/unpacked
        if len(pallets_seen) > 1:
            errors.append(
                f"GROUP-SPLIT '{label}': spans pallets {sorted(pallets_seen)} "
                f"({len(members)} members)"
            )
        if pallets_seen and n_unpacked:
            errors.append(
                f"GROUP-PARTIAL '{label}': {n_unpacked}/{len(members)} unpacked, "
                f"rest on pallet(s) {sorted(pallets_seen)}"
            )
    return errors, n_groups


def check_validity(result, pallet, config):
    return validate(result, pallet, config)


# ---------------------------------------------------------------------------
# Constructed adversarial instances
# ---------------------------------------------------------------------------
def constructed_cases():
    out = []

    # K1 — heavy group pressure: 3 groups, each must co-locate, but volume
    # forces ~2-3 pallets. Pushes the solver to either split a group or
    # leave a whole group unpacked.
    boxes = []
    for g, n, (l, w, h) in [("GA", 18, (400, 350, 300)),
                            ("GB", 18, (400, 350, 300)),
                            ("GC", 18, (400, 350, 300))]:
        for i in range(n):
            boxes.append(Box(id=f"{g}-{i:02d}", length=l, width=w, height=h,
                             weight=3.0, allowed_rotations=THIS_SIDE_UP,
                             max_load_on_top=20.0, group=g))
    out.append(("K1 group-pressure 3x18", boxes,
                Pallet(length=1200, width=1000, height=1500, max_weight=1000),
                PackerConfig()))

    # K2 — one oversized group + many singletons. Group GX is huge (won't fit
    # one pallet), so if groups were enforced the whole group would be
    # unpacked. Singletons fill remaining space.
    boxes = []
    for i in range(40):
        boxes.append(Box(id=f"GX-{i:02d}", length=400, width=300, height=300,
                         weight=2.0, allowed_rotations=THIS_SIDE_UP,
                         max_load_on_top=20.0, group="GX"))
    for i in range(10):
        boxes.append(Box(id=f"S-{i:02d}", length=200, width=150, height=100,
                         weight=0.5, allowed_rotations=THIS_SIDE_UP,
                         max_load_on_top=10.0))
    out.append(("K2 oversized-group + singletons", boxes,
                Pallet(length=1200, width=1000, height=1500, max_weight=1000),
                PackerConfig()))

    # K3 — duplicate box.id collision stress: two distinct objects share an id
    # string. Conservation by-id-string must still net out (multiset).
    boxes = []
    for i in range(20):
        boxes.append(Box(id=f"DUP-{i % 5}", length=250, width=200, height=150,
                         weight=1.0, allowed_rotations=ALL_ROTATIONS,
                         max_load_on_top=15.0))
    out.append(("K3 colliding id-strings (4x each of 5 ids)", boxes,
                Pallet(length=1200, width=1000, height=1500, max_weight=1000),
                PackerConfig()))

    # K4 — single huge box that fits NO pallet (oversize) mixed with normals.
    # Oversize must always be unpacked; never lost or duplicated.
    boxes = [Box(id="HUGE", length=5000, width=5000, height=5000, weight=1.0,
                 allowed_rotations=ALL_ROTATIONS)]
    for i in range(25):
        boxes.append(Box(id=f"N-{i:02d}", length=300, width=250, height=200,
                         weight=1.0, allowed_rotations=ALL_ROTATIONS,
                         max_load_on_top=10.0))
    out.append(("K4 oversize + normals", boxes,
                Pallet(length=1200, width=1000, height=1500, max_weight=1000),
                PackerConfig()))

    # K5 — pure-geometric (no constraints) many small boxes, forces several
    # pallets when capped. Conservation under multi-pallet geometric path.
    boxes = []
    for i in range(120):
        boxes.append(Box(id=f"G-{i:03d}", length=300, width=250, height=200,
                         weight=0.0, allowed_rotations=ALL_ROTATIONS))
    out.append(("K5 geometric 120 small", boxes,
                Pallet(length=1200, width=1000, height=1500),
                PackerConfig()))

    # K6 — singleton groups (each box its own group). Should never violate.
    boxes = []
    for i in range(30):
        boxes.append(Box(id=f"U-{i:02d}", length=300, width=250, height=200,
                         weight=1.0, allowed_rotations=THIS_SIDE_UP,
                         max_load_on_top=10.0, group=f"u{i}"))
    out.append(("K6 singleton-groups 30", boxes,
                Pallet(length=1200, width=1000, height=1500, max_weight=1000),
                PackerConfig()))

    return out


# ---------------------------------------------------------------------------
def run_one(name, boxes, pallet, config, max_pallets, seed=42, t=4.0):
    mp_arg = max_pallets if max_pallets is not None else 1000
    r = brkga_pack_v35(
        boxes, pallet, config,
        time_limit_s=t, max_pallets=mp_arg,
        population_size=120, n_populations=2, patience=120,
        local_search_budget_s=1.0, seed=seed,
        verbose=False, n_modes=6,
    )
    return r


def main():
    warmup_jit()

    overall_ok = True
    # aggregate trackers
    n_runs = 0
    cons_viol = 0
    pallet_viol = 0
    group_viol = 0
    valid_viol = 0
    worst = []  # (label, errors...)

    MP_GRID = [1, 2, 3, 4, 5, None]  # None = unlimited

    inds = industry_cases()
    # focus group-relevant + a representative spread; include IND1 (no groups),
    # IND6 (groups), and a couple multi-pallet-forcing ones.
    ind_subset = [inds[0], inds[3], inds[5], inds[1], inds[8]]  # IND1,IND4,IND6,IND2,IND9

    suite = [(c.name, c.boxes, c.pallet, c.config) for c in ind_subset]
    suite += constructed_cases()

    print("=" * 78)
    print("MULTI-PALLET CONSERVATION / GROUP / MAX_PALLETS / VALIDITY PROBE")
    print("=" * 78)

    for (name, boxes, pallet, config) in suite:
        n_groups_total = len({b.group for b in boxes if b.group is not None})
        print(f"\n### {name}  (N={len(boxes)}, groups={n_groups_total})")
        for mp in MP_GRID:
            n_runs += 1
            label = f"{name} | max_pallets={mp}"
            try:
                r = run_one(name, boxes, pallet, config, mp)
            except Exception as e:  # noqa
                overall_ok = False
                print(f"  [mp={str(mp):>4}] EXCEPTION: {type(e).__name__}: {e}")
                worst.append((label, [f"EXCEPTION {type(e).__name__}: {e}"]))
                continue

            ce, stats = check_conservation(boxes, r)
            pe = check_max_pallets(r, mp)
            ge, ng = check_groups(boxes, r)
            ve = check_validity(r, pallet, config)

            errs = []
            if ce:
                cons_viol += 1
                errs += ce
            if pe:
                pallet_viol += 1
                errs += pe
            if ge:
                group_viol += 1
                errs += ge
            if ve:
                valid_viol += 1
                errs += [f"VALIDATE: {v}" for v in ve[:3]]

            tag = "PASS" if not errs else "FAIL"
            if errs:
                overall_ok = False
                worst.append((label, errs))
            print(f"  [mp={str(mp):>4}] {tag}  "
                  f"plt={stats['pallets']} placed={stats['placed']} "
                  f"unp={stats['unpacked']} tot={stats['total']}/{stats['input']}"
                  + ("" if not errs else "   <<< " + " ; ".join(errs[:2])))

    print("\n" + "=" * 78)
    print("AGGREGATE")
    print(f"  runs                 : {n_runs}")
    print(f"  conservation viol    : {cons_viol}")
    print(f"  max_pallets viol     : {pallet_viol}")
    print(f"  group-split viol     : {group_viol}")
    print(f"  validate() viol      : {valid_viol}")
    if worst:
        print("\n  WORST OFFENDERS (up to 12):")
        for label, errs in worst[:12]:
            print(f"   - {label}")
            for e in errs[:3]:
                print(f"       {e}")

    print("\n" + "=" * 78)
    print(f"RESULT: {'PASS' if overall_ok else 'FAIL'}")
    print("=" * 78)
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
