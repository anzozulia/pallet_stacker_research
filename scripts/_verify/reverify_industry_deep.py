"""DEEP re-verification of all 10 industry cases (industry-deep dimension).

For EACH case in benchmarks.industry.cases() run brkga_pack_v35 with the
mandated config (time_limit_s=12, max_pallets=10, pop=300, n_pop=3,
patience=150, seed=42, n_modes=6) and rigorously check:

  (V)  validate() returns [] (weight, support, load-bearing, bounds, rotation)
  (C)  BOX CONSERVATION — every input box id appears exactly once across
       placed+unpacked, by python-identity AND by id-string multiset.
  (P)  max_pallets never exceeded (cap=10).
  (G)  group co-location — no Box.group spans >1 pallet (IND6 canonical).
       (validator does NOT check this — checked independently here.)
  (W)  per-pallet weight <= pallet.max_weight.
  (F)  fragile boxes (max_load_on_top==0) carry ~0 load (independent recompute).
  (B)  bounds: every placement inside pallet (height/length/width).
  (S)  sanity vs 30_verification.md table (informational, not pass/fail):
       IND2 ~4 pallets, IND4 1, IND9 2, IND10 2; all 0 unpacked where expected.

Run in Docker:
  docker run --rm -e OMP_NUM_THREADS=1 -e PYTHONHASHSEED=0 \
    -v "$(pwd)":/app -w /app pallet-packer:dev \
    python scripts/_verify/reverify_industry_deep.py
"""
from __future__ import annotations

import os
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pallet_packer import validate
from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit
from benchmarks.industry import cases as industry_cases

EPS = 1e-6

# Reference from docs/reports/30_verification.md industry table (PRE-fix audit).
# (pallets, unpacked) — informational sanity bounds, NOT hard pass/fail.
REF = {
    "IND1": (9, 0), "IND2": (4, 0), "IND3": (8, 0), "IND4": (1, 0),
    "IND5": (2, 0), "IND6": (1, 0), "IND7": (8, 0), "IND8": (4, 0),
    "IND9": (2, 0), "IND10": (2, 0),
}


def short(name):
    return name.split()[0]  # "IND6"


def check_conservation(input_boxes, result):
    errs = []
    placed_ids, placed_keys = [], []
    for st in result.pallets:
        for p in st.placements:
            placed_ids.append(id(p.box))
            placed_keys.append(p.box.id)
    unpacked_ids = [id(b) for b in result.unpacked]
    unpacked_keys = [b.id for b in result.unpacked]

    in_ids = [id(b) for b in input_boxes]
    in_keys = [b.id for b in input_boxes]

    n_placed, n_unpacked = len(placed_ids), len(unpacked_ids)
    n_total, n_in = n_placed + n_unpacked, len(in_ids)

    if n_total != n_in:
        errs.append(f"COUNT placed({n_placed})+unp({n_unpacked})={n_total}!=in({n_in})")

    all_out = Counter(placed_ids) + Counter(unpacked_ids)
    in_set = Counter(in_ids)
    dup = {i: c for i, c in all_out.items() if c > 1}
    if dup:
        id2k = {id(b): b.id for b in input_boxes}
        errs.append(f"DUP(id): {len(dup)} e.g. {[id2k.get(i,'?') for i in list(dup)[:5]]}")
    missing = [i for i in in_set if all_out.get(i, 0) == 0]
    if missing:
        id2k = {id(b): b.id for b in input_boxes}
        errs.append(f"LOST(id): {len(missing)} e.g. {[id2k[i] for i in missing[:5]]}")
    alien = [i for i in all_out if i not in in_set]
    if alien:
        errs.append(f"ALIEN(id): {len(alien)}")

    out_ms = Counter(placed_keys) + Counter(unpacked_keys)
    in_ms = Counter(in_keys)
    if out_ms != in_ms:
        extra = out_ms - in_ms
        gone = in_ms - out_ms
        errs.append(f"MULTISET mismatch extra={dict(list(extra.items())[:5])} "
                    f"gone={dict(list(gone.items())[:5])}")

    cross = set(placed_ids) & set(unpacked_ids)
    if cross:
        errs.append(f"CROSS-DUP: {len(cross)} both placed AND unpacked")

    return errs, dict(placed=n_placed, unpacked=n_unpacked, pallets=len(result.pallets))


def check_groups(input_boxes, result):
    """Boxes sharing a non-None group must all be on ONE pallet (or all unpacked)."""
    errs = []
    box_pallet = {}
    for pi, st in enumerate(result.pallets):
        for p in st.placements:
            box_pallet[id(p.box)] = pi
    unpacked = {id(b) for b in result.unpacked}
    groups = defaultdict(list)
    for b in input_boxes:
        if b.group is not None:
            groups[b.group].append(b)
    for label, members in groups.items():
        seen = set()
        nunp = 0
        for b in members:
            if id(b) in box_pallet:
                seen.add(box_pallet[id(b)])
            elif id(b) in unpacked:
                nunp += 1
        if len(seen) > 1:
            errs.append(f"GROUP-SPLIT '{label}' spans pallets {sorted(seen)} "
                        f"({len(members)} members)")
        if seen and nunp:
            errs.append(f"GROUP-PARTIAL '{label}' {nunp}/{len(members)} unpacked, "
                        f"rest on {sorted(seen)}")
    return errs, len(groups)


def check_per_pallet_weight(result, pallet):
    errs = []
    for pi, st in enumerate(result.pallets):
        w = sum(p.box.weight for p in st.placements)
        # cross-check engine's cached total_weight too
        cached = getattr(st, "total_weight", None)
        if w > pallet.max_weight + 1e-6:
            errs.append(f"WEIGHT pallet#{pi} {w:.2f} > cap {pallet.max_weight}")
        if cached is not None and abs(cached - w) > 1e-3:
            errs.append(f"WEIGHT-CACHE pallet#{pi} cached={cached:.3f} recomputed={w:.3f}")
    return errs


def check_bounds(result, pallet):
    """Independent in-bounds recheck (no overhang for these cases)."""
    errs = []
    for pi, st in enumerate(result.pallets):
        for p in st.placements:
            if (p.x < -EPS or p.y < -EPS or p.z < -EPS or
                    p.x2 > pallet.length + EPS or
                    p.y2 > pallet.width + EPS or
                    p.z2 > pallet.height + EPS):
                errs.append(f"BOUNDS pallet#{pi} {p.box.id} "
                            f"({p.x},{p.y},{p.z})+({p.dx},{p.dy},{p.dz})")
    return errs


def check_fragile_zero_load(result):
    """Independently recompute load resting on each fragile box (mlot==0).

    A fragile box (max_load_on_top==0) must carry ~0 kg. We recompute the
    direct-supporter load distribution exactly as the validator does and
    flag any fragile box bearing > EPS.
    """
    errs = []
    n_fragile = 0
    max_frag_load = 0.0
    for pi, st in enumerate(result.pallets):
        plc = st.placements
        load_on = {id(p): 0.0 for p in plc}
        for placed in plc:
            sups = []
            for q in plc:
                if q is placed:
                    continue
                if abs(placed.z - q.z2) > EPS:
                    continue
                ox = max(0.0, min(placed.x2, q.x2) - max(placed.x, q.x))
                oy = max(0.0, min(placed.y2, q.y2) - max(placed.y, q.y))
                if ox * oy > EPS:
                    sups.append((q, ox * oy))
            sa = sum(a for _, a in sups)
            if sa <= 0:
                continue
            for q, a in sups:
                load_on[id(q)] += placed.box.weight * (a / sa)
        for q in plc:
            if q.box.max_load_on_top == 0.0:
                n_fragile += 1
                lod = load_on[id(q)]
                max_frag_load = max(max_frag_load, lod)
                if lod > 1e-3:
                    errs.append(f"FRAGILE pallet#{pi} {q.box.id} carries {lod:.3f}kg "
                                f"(mlot=0)")
    return errs, n_fragile, max_frag_load


def run():
    warmup_jit()
    cases = industry_cases()

    print("=" * 100)
    print("INDUSTRY-DEEP RE-VERIFICATION — 10 cases, seed=42, t=12s, mp=10, "
          "pop=300x3, patience=150, n_modes=6")
    print("=" * 100)

    rows = []
    overall_ok = True
    agg = dict(V=0, C=0, P=0, G=0, W=0, F=0, B=0)

    for c in cases:
        key = short(c.name)
        n_in = len(c.boxes)
        n_groups = len({b.group for b in c.boxes if b.group is not None})
        n_fragile_in = sum(1 for b in c.boxes if b.max_load_on_top == 0.0)

        t0 = time.time()
        try:
            r = brkga_pack_v35(
                c.boxes, c.pallet, c.config,
                time_limit_s=12, max_pallets=10,
                population_size=300, n_populations=3, patience=150,
                seed=42, n_modes=6, verbose=False,
            )
        except Exception as e:
            overall_ok = False
            print(f"\n### {key}  EXCEPTION: {type(e).__name__}: {e}")
            rows.append((key, "EXCEPTION", str(e)))
            continue
        dt = time.time() - t0

        ve = validate(r, c.pallet, c.config)
        ce, stats = check_conservation(c.boxes, r)
        pe = [] if len(r.pallets) <= 10 else [f"MAX_PALLETS {len(r.pallets)}>10"]
        ge, ng = check_groups(c.boxes, r)
        we = check_per_pallet_weight(r, c.pallet)
        be = check_bounds(r, c.pallet)
        fe, n_frag, max_frag = check_fragile_zero_load(r)

        if ve: agg['V'] += 1
        if ce: agg['C'] += 1
        if pe: agg['P'] += 1
        if ge: agg['G'] += 1
        if we: agg['W'] += 1
        if fe: agg['F'] += 1
        if be: agg['B'] += 1

        case_errs = []
        if ve: case_errs += [f"VALIDATE({len(ve)}): " + ve[0]]
        case_errs += ce + pe + ge + we + be + fe
        ok = not case_errs
        if not ok:
            overall_ok = False

        ref_p, ref_u = REF.get(key, (None, None))
        sane = ""
        if ref_p is not None:
            # informational: flag large deviation in pallets, and any unexpected unpacked
            if stats['unpacked'] != ref_u:
                sane += f" UNP={stats['unpacked']}(ref {ref_u})"
            if ref_p and abs(stats['pallets'] - ref_p) > max(2, ref_p):
                sane += f" PLT={stats['pallets']}(ref {ref_p})"

        print(f"\n### {key}  N={n_in} groups={n_groups} frag_in={n_fragile_in}  ({dt:.1f}s)")
        print(f"    pallets={stats['pallets']} placed={stats['placed']} "
              f"unpacked={stats['unpacked']}  frag_placed={n_frag} "
              f"max_frag_load={max_frag:.4f}kg")
        print(f"    [{'OK ' if not ve else 'ERR'}] validate ({len(ve)} errs)   "
              f"[{'OK ' if not ce else 'ERR'}] conservation   "
              f"[{'OK ' if not pe else 'ERR'}] max_pallets   "
              f"[{'OK ' if not ge else 'ERR'}] groups({ng})")
        print(f"    [{'OK ' if not we else 'ERR'}] weight   "
              f"[{'OK ' if not be else 'ERR'}] bounds   "
              f"[{'OK ' if not fe else 'ERR'}] fragile-zero-load")
        if sane:
            print(f"    SANITY-NOTE:{sane}")
        if case_errs:
            for e in case_errs[:6]:
                print(f"      <<< {e}")

        rows.append((key, "PASS" if ok else "FAIL",
                     f"plt={stats['pallets']} plc={stats['placed']} "
                     f"unp={stats['unpacked']}{sane}"))

    print("\n" + "=" * 100)
    print("PER-CASE SUMMARY")
    print("=" * 100)
    for key, st, detail in rows:
        print(f"  {key:6s} {st:9s} {detail}")

    print("\nVIOLATION COUNTS (cases with >=1 violation in that check):")
    for k, label in [('V', 'validate'), ('C', 'conservation'), ('P', 'max_pallets'),
                     ('G', 'group-split'), ('W', 'weight'), ('F', 'fragile-load'),
                     ('B', 'bounds')]:
        print(f"  {label:14s}: {agg[k]}")

    print("\n" + "=" * 100)
    print(f"RESULT: {'PASS' if overall_ok else 'FAIL'}")
    print("=" * 100)
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(run())
