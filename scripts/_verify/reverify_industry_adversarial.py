"""Adversarial follow-ups for industry-deep.

A) DETERMINISM: rerun IND1/IND6/IND9 twice with seed=42; placements must be
   bit-identical (count + sorted (id,x,y,z,rot)). Proves the fixes are stable.
B) MAX_PALLETS PRESSURE: drive IND6 (groups) and IND2 (fragile-heavy) down to
   caps {1,2,3} — group co-location must still hold (no split), conservation
   must hold (placed+unpacked==N), cap honored, validator clean.
C) FRAGILE NON-VACUITY: prove the engine actually stacks (some box has another
   box resting on it) so the fragile-zero-load check on IND1/IND2/IND7 is not
   trivially satisfied by a flat single-layer packing. Report stacking depth.
D) FRAGILE-AS-SUPPORTER count: how many fragile boxes actually have something
   above their z-level footprint region (should be 0 carried-load but we report
   how many fragile boxes are NOT top-of-stack).

Run in Docker (OMP_NUM_THREADS=1).
"""
from __future__ import annotations
import os, sys, time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pallet_packer import validate
from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit
from benchmarks.industry import cases as industry_cases

EPS = 1e-6
PASS = FAIL = 0


def chk(label, cond, detail=""):
    global PASS, FAIL
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}{(': ' + detail) if detail else ''}")
    PASS += cond
    FAIL += (not cond)


def fingerprint(r):
    """Order-independent fingerprint of a packing."""
    fp = []
    for pi, st in enumerate(r.pallets):
        for p in st.placements:
            fp.append((pi, p.box.id, round(p.x, 4), round(p.y, 4), round(p.z, 4),
                       p.rotation.name, round(p.dx, 4), round(p.dy, 4), round(p.dz, 4)))
    fp.sort()
    unp = sorted(b.id for b in r.unpacked)
    return (tuple(fp), tuple(unp))


def run_case(c, mp=10, t=12, seed=42):
    return brkga_pack_v35(c.boxes, c.pallet, c.config, time_limit_s=t,
                          max_pallets=mp, population_size=300, n_populations=3,
                          patience=150, seed=seed, n_modes=6, verbose=False)


def group_split(r):
    gp = defaultdict(set)
    for pi, st in enumerate(r.pallets):
        for p in st.placements:
            if p.box.group is not None:
                gp[p.box.group].add(pi)
    return {g: sorted(s) for g, s in gp.items() if len(s) > 1}


def stacking_stats(r):
    """Return (max_layers, n_boxes_with_something_on_top, n_fragile_not_top)."""
    n_supported_above = 0
    n_frag_not_top = 0
    max_z_levels = 0
    for st in r.pallets:
        plc = st.placements
        zlevels = set(round(p.z, 3) for p in plc)
        max_z_levels = max(max_z_levels, len(zlevels))
        for q in plc:
            has_above = False
            for p in plc:
                if p is q:
                    continue
                if abs(p.z - q.z2) > EPS:
                    continue
                ox = max(0.0, min(p.x2, q.x2) - max(p.x, q.x))
                oy = max(0.0, min(p.y2, q.y2) - max(p.y, q.y))
                if ox * oy > EPS:
                    has_above = True
                    break
            if has_above:
                n_supported_above += 1
                if q.box.max_load_on_top == 0.0:
                    n_frag_not_top += 1
    return max_z_levels, n_supported_above, n_frag_not_top


def main():
    warmup_jit()
    cases = {c.name.split()[0]: c for c in industry_cases()}

    print("=" * 90)
    print("A) DETERMINISM (seed=42, two runs each must be bit-identical)")
    print("=" * 90)
    for key in ["IND1", "IND6", "IND9"]:
        c = cases[key]
        r1 = run_case(c)
        r2 = run_case(c)
        fp1, fp2 = fingerprint(r1), fingerprint(r2)
        chk(f"{key} deterministic (pallets {len(r1.pallets)}/{len(r2.pallets)})",
            fp1 == fp2,
            "" if fp1 == fp2 else "FINGERPRINT DIFF")

    print("\n" + "=" * 90)
    print("B) MAX_PALLETS PRESSURE (force splits; group + conservation must hold)")
    print("=" * 90)
    for key in ["IND6", "IND2"]:
        c = cases[key]
        n_in = len(c.boxes)
        for cap in [1, 2, 3]:
            r = run_case(c, mp=cap)
            placed = sum(len(st.placements) for st in r.pallets)
            unp = len(r.unpacked)
            errs = validate(r, c.pallet, c.config)
            splits = group_split(r)
            cons = (placed + unp == n_in)
            cap_ok = len(r.pallets) <= cap
            print(f"  {key} cap={cap}: pallets={len(r.pallets)} placed={placed} "
                  f"unp={unp} errs={len(errs)} splits={splits}")
            chk(f"{key} cap={cap} no group split", len(splits) == 0, str(splits))
            chk(f"{key} cap={cap} conservation", cons, f"{placed}+{unp}!={n_in}")
            chk(f"{key} cap={cap} cap honored", cap_ok, f"{len(r.pallets)}>{cap}")
            chk(f"{key} cap={cap} validator clean", len(errs) == 0,
                "" if not errs else errs[0])

    print("\n" + "=" * 90)
    print("C) FRAGILE NON-VACUITY (prove the engine actually stacks)")
    print("=" * 90)
    for key in ["IND1", "IND2", "IND7", "IND9", "IND10"]:
        c = cases[key]
        r = run_case(c)
        max_layers, n_above, n_frag_not_top = stacking_stats(r)
        nfrag = sum(1 for st in r.pallets for p in st.placements
                    if p.box.max_load_on_top == 0.0)
        print(f"  {key}: max_z_levels={max_layers} boxes_with_something_on_top={n_above} "
              f"fragile_placed={nfrag} fragile_NOT_top={n_frag_not_top}")
        # Non-vacuity: at least one case must genuinely stack (>1 z-level and
        # boxes carrying load). And no fragile box may be a non-top supporter.
        chk(f"{key} no fragile carries load (fragile_not_top=={n_frag_not_top})",
            n_frag_not_top == 0)

    # Global non-vacuity assertion: engine DOES stack somewhere.
    any_stack = False
    for key in ["IND1", "IND2", "IND7", "IND9", "IND10"]:
        r = run_case(cases[key])
        ml, na, _ = stacking_stats(r)
        if ml > 1 and na > 0:
            any_stack = True
            break
    chk("engine genuinely stacks (>=1 case with >1 layer and load-bearing)", any_stack)

    print(f"\n=== {PASS} passed, {FAIL} failed ===")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
