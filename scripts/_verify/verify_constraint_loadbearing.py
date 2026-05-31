"""verify_constraint_loadbearing.py — adversarial load-bearing / fragility probe.

Mandate: break LOAD-BEARING (max_load_on_top) + requires_full_support (rfs)
+ fragility (mlot=0). Construct stacking-tempting instances, run them through
brkga_pack_v35 with enforce_load_bearing, and assert validate() reports ZERO
load-bearing violations. Also independently re-derive the per-supporter load
distribution (the validator's contact-area model) to confirm the validator
itself is sound and agrees with the solver.

Run (from repo root):
    docker run --rm -e OMP_NUM_THREADS=1 -e PYTHONHASHSEED=0 \
        -v "$(pwd)":/app -w /app pallet-packer:dev \
        python scripts/_verify/verify_constraint_loadbearing.py

Sub-tests (>=20 generated instances across families):
  A. Pure-fragile (mlot=0) — every fragile box must end up on top or alone.
  B. Small finite mlot, heavy tempting items — density temptation.
  C. Mixed fragile/sturdy at scale.
  D. requires_full_support boxes.
  E. Tall homogeneous stacks where DIRECT vs CUMULATIVE load diverge
     (probes whether validator/decoder model recursive load — documented as
     "recursively" in PackerConfig but implemented as direct-only).
  F. Float / non-integer dims — int-decode vs float-validate mismatch in the
     contact-area load split.
  G. enforce_load_bearing=False asymmetry (decoder still enforces via
     has_constraints; validator skips — confirm no surprise).
  H. Boundary mlot (load == mlot exactly).
  I. The 10 industry cases (real configs with mlot set).

For every result we run validate() (ground truth) AND an independent
recompute of per-box top-load. We print the worst offender per family.
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import math
from dataclasses import replace
from typing import List

from pallet_packer import (
    Box, Pallet, PackerConfig, validate,
    ALL_ROTATIONS, THIS_SIDE_UP, NO_ROTATION,
)
from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit
from benchmarks.industry import cases as industry_cases

EPS = 1e-6


# ---------------------------------------------------------------------------
# Independent re-derivation of the validator's load model (DIRECT supporters,
# contact-area-weighted). This is a *second* implementation so we cross-check
# the validator's own arithmetic rather than trust it blindly.
# ---------------------------------------------------------------------------
def recompute_top_loads(st):
    """Return list of (box_id, weight_carried, mlot, slack) per placement.

    Mirrors validate.py step 6: each placed box distributes its own weight to
    its DIRECT supporters (those whose top z == placed.z) in proportion to
    contact area. slack < 0 means the box carries more than its mlot.
    """
    pls = st.placements
    carried = {id(p): 0.0 for p in pls}
    for placed in pls:
        sups = []
        for q in pls:
            if q is placed:
                continue
            if abs(placed.z - q.z2) > EPS:
                continue
            ox = max(0.0, min(placed.x2, q.x2) - max(placed.x, q.x))
            oy = max(0.0, min(placed.y2, q.y2) - max(placed.y, q.y))
            if ox * oy > EPS:
                sups.append((q, ox * oy))
        tot = sum(a for _, a in sups)
        if tot <= 0:
            continue
        for q, a in sups:
            carried[id(q)] += placed.box.weight * (a / tot)
    rows = []
    for p in pls:
        w = carried[id(p)]
        mlot = p.box.max_load_on_top
        rows.append((p.box.id, w, mlot, mlot - w))
    return rows


def recompute_cumulative_loads(st):
    """CUMULATIVE (recursive) load: total weight of everything resting in the
    column above each box, propagated down through the stack by contact area.

    This is the *physical* load a box bears — NOT what validate.py checks
    (validate.py only counts the directly-resting box's own weight). Used to
    expose whether the "recursively" claim in PackerConfig holds.
    """
    pls = st.placements
    # topo order: process highest boxes first so their accumulated load flows
    # down. carried[id] = total weight pressing on this box's top face.
    carried = {id(p): 0.0 for p in pls}
    order = sorted(pls, key=lambda p: p.z2, reverse=True)
    for placed in order:
        # weight this box transmits downward = its own weight + everything on it
        transmit = placed.box.weight + carried[id(placed)]
        sups = []
        for q in pls:
            if q is placed:
                continue
            if abs(placed.z - q.z2) > EPS:
                continue
            ox = max(0.0, min(placed.x2, q.x2) - max(placed.x, q.x))
            oy = max(0.0, min(placed.y2, q.y2) - max(placed.y, q.y))
            if ox * oy > EPS:
                sups.append((q, ox * oy))
        tot = sum(a for _, a in sups)
        if tot <= 0:
            continue
        for q, a in sups:
            carried[id(q)] += transmit * (a / tot)
    rows = []
    for p in pls:
        w = carried[id(p)]
        mlot = p.box.max_load_on_top
        rows.append((p.box.id, w, mlot, mlot - w))
    return rows


def solve(boxes, pallet, config, *, max_pallets, seed=42, t=6.0, n=None):
    return brkga_pack_v35(
        boxes, pallet, config,
        time_limit_s=t, max_pallets=max_pallets, seed=seed,
        population_size=200, n_populations=3, patience=150,
        local_search_budget_s=1.5, verbose=False, n_modes=6,
    )


def n_placed(r):
    return sum(len(st.placements) for st in r.pallets)


def lb_errors(errs):
    return [e for e in errs if "max_load_on_top" in e]


# ---------------------------------------------------------------------------
# Instance families
# ---------------------------------------------------------------------------
def make_pure_fragile(n_fragile, n_sturdy):
    """All-fragile (mlot=0) light boxes + a few sturdy heavy bases.

    Tempting: stack the light fragile ones on the big base for density.
    Correct: nothing may rest on a fragile box; fragile boxes must be on top
    or on the floor.
    """
    boxes = []
    for i in range(n_sturdy):
        boxes.append(Box(id=f"BASE-{i:02d}", length=600, width=400, height=200,
                         weight=40.0, allowed_rotations=THIS_SIDE_UP,
                         max_load_on_top=500.0))
    for i in range(n_fragile):
        boxes.append(Box(id=f"FRAG-{i:02d}", length=300, width=200, height=100,
                         weight=5.0, allowed_rotations=THIS_SIDE_UP,
                         max_load_on_top=0.0))
    return boxes


def make_small_mlot_heavy(n):
    """Boxes with a small finite mlot, plus heavy boxes that would love to
    stack on them. mlot = 10kg but heavy boxes weigh 50kg.
    """
    boxes = []
    for i in range(n):
        boxes.append(Box(id=f"WEAK-{i:02d}", length=400, width=300, height=200,
                         weight=8.0, allowed_rotations=THIS_SIDE_UP,
                         max_load_on_top=10.0))
    for i in range(n // 2):
        boxes.append(Box(id=f"HEAVY-{i:02d}", length=400, width=300, height=200,
                         weight=50.0, allowed_rotations=THIS_SIDE_UP,
                         max_load_on_top=200.0))
    return boxes


def make_mixed(n):
    boxes = []
    for i in range(n):
        if i % 3 == 0:
            boxes.append(Box(id=f"FR-{i:02d}", length=250, width=250, height=150,
                             weight=3.0, allowed_rotations=THIS_SIDE_UP,
                             max_load_on_top=0.0))
        elif i % 3 == 1:
            boxes.append(Box(id=f"MD-{i:02d}", length=300, width=300, height=150,
                             weight=10.0, allowed_rotations=THIS_SIDE_UP,
                             max_load_on_top=15.0))
        else:
            boxes.append(Box(id=f"ST-{i:02d}", length=350, width=300, height=200,
                             weight=20.0, allowed_rotations=THIS_SIDE_UP,
                             max_load_on_top=120.0))
    return boxes


def make_rfs(n):
    """requires_full_support boxes that also have a tight mlot."""
    boxes = []
    for i in range(n):
        boxes.append(Box(id=f"RFS-{i:02d}", length=300, width=300, height=200,
                         weight=12.0, allowed_rotations=THIS_SIDE_UP,
                         max_load_on_top=12.0, requires_full_support=True))
    return boxes


def make_tall_stack():
    """Homogeneous column-tempting: identical boxes, mlot just over ONE box's
    weight. A 5-high stack puts 4*w on the bottom box (cumulative) but each
    box's DIRECT supporter only sees 1*w. Probes direct-vs-cumulative model.
    weight=20, mlot=25 → direct OK (20<25) but cumulative bottom sees 80.
    """
    boxes = []
    for i in range(30):
        boxes.append(Box(id=f"COL-{i:02d}", length=300, width=300, height=200,
                         weight=20.0, allowed_rotations=THIS_SIDE_UP,
                         max_load_on_top=25.0))
    return boxes


def make_float_dims():
    """Non-integer dims + weights. int-decode rounds dims; validate uses float
    dims. The contact-area split can then differ between decoder and validator.
    """
    boxes = []
    for i in range(20):
        boxes.append(Box(id=f"FL-{i:02d}", length=333.7, width=247.3,
                         height=151.9, weight=7.4, allowed_rotations=THIS_SIDE_UP,
                         max_load_on_top=9.5))
    for i in range(6):
        boxes.append(Box(id=f"FH-{i:02d}", length=333.7, width=247.3,
                         height=151.9, weight=44.2, allowed_rotations=THIS_SIDE_UP,
                         max_load_on_top=300.0))
    return boxes


def make_boundary():
    """mlot exactly equal to one box weight; partial-area overlap means the
    split lands right on the boundary. Tempts an EPS-edge violation.
    """
    boxes = []
    # large bases with mlot exactly = weight of one small box
    for i in range(6):
        boxes.append(Box(id=f"B-{i:02d}", length=600, width=400, height=200,
                         weight=30.0, allowed_rotations=THIS_SIDE_UP,
                         max_load_on_top=10.0))
    for i in range(18):
        boxes.append(Box(id=f"T-{i:02d}", length=300, width=200, height=150,
                         weight=10.0, allowed_rotations=THIS_SIDE_UP,
                         max_load_on_top=100.0))
    return boxes


# ---------------------------------------------------------------------------
def check_instance(label, boxes, pallet, config, *, max_pallets, seeds=(42, 7, 123)):
    """Run an instance across several seeds; aggregate validator + independent
    load-model checks. Returns (ok, summary_dict).
    """
    worst_validator = None      # (slack, box, carried, mlot)  most-negative slack
    worst_independent = None
    worst_cumulative = None
    total_lb_violations = 0
    total_validator_errs = 0
    runs = 0
    fragile_on_floor_ok = True
    fragile_violation_example = None

    for seed in seeds:
        r = solve(boxes, pallet, config, max_pallets=max_pallets, seed=seed)
        runs += 1
        errs = validate(r, pallet, config)
        total_validator_errs += len(errs)
        total_lb_violations += len(lb_errors(errs))

        for st in r.pallets:
            # independent direct-load recompute (mirror validator math)
            for bid, w, mlot, slack in recompute_top_loads(st):
                if worst_independent is None or slack < worst_independent[0]:
                    worst_independent = (slack, bid, w, mlot)
            # cumulative (physical) recompute — informational
            for bid, w, mlot, slack in recompute_cumulative_loads(st):
                if worst_cumulative is None or slack < worst_cumulative[0]:
                    worst_cumulative = (slack, bid, w, mlot)
            # explicit fragile check: any box with mlot==0 carrying anything?
            for bid, w, mlot, slack in recompute_top_loads(st):
                if mlot <= EPS and w > EPS:
                    fragile_on_floor_ok = False
                    if (fragile_violation_example is None
                            or w > fragile_violation_example[1]):
                        fragile_violation_example = (bid, w)

        # parse validator's own reported numbers for worst slack
        for e in lb_errors(errs):
            # format: "...: <id> carries <carried> kg > max_load_on_top <mlot>"
            try:
                carried = float(e.split("carries")[1].split("kg")[0])
                mlot = float(e.split("max_load_on_top")[1])
                slack = mlot - carried
                if worst_validator is None or slack < worst_validator[0]:
                    worst_validator = (slack, e.split(":")[1].strip().split()[0],
                                       carried, mlot)
            except Exception:
                worst_validator = (-1.0, "parse-fail", 0.0, 0.0)

    ok = (total_lb_violations == 0 and fragile_on_floor_ok
          and (worst_independent is None or worst_independent[0] >= -1e-3))
    summary = dict(
        label=label, runs=runs, n_boxes=len(boxes),
        validator_errs=total_validator_errs,
        lb_violations=total_lb_violations,
        worst_validator=worst_validator,
        worst_independent=worst_independent,
        worst_cumulative=worst_cumulative,
        fragile_on_floor_ok=fragile_on_floor_ok,
        fragile_violation_example=fragile_violation_example,
        n_placed=n_placed(r),
    )
    return ok, summary


def print_summary(ok, s):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {s['label']:<40} N={s['n_boxes']:>3} "
          f"placed={s['n_placed']:>3} validator_errs={s['validator_errs']} "
          f"LB_viol={s['lb_violations']}")
    wi = s['worst_independent']
    if wi is not None:
        flag = "  <-- VIOLATION" if wi[0] < -1e-3 else ""
        print(f"        worst DIRECT load: {wi[1]} carries {wi[2]:.3f}kg "
              f"vs mlot {wi[3]:.3f} (slack {wi[0]:+.3f}){flag}")
    wc = s['worst_cumulative']
    if wc is not None and wc[0] < -1e-3:
        print(f"        [info] worst CUMULATIVE (physical) load: {wc[1]} bears "
              f"{wc[2]:.3f}kg vs mlot {wc[3]:.3f} (slack {wc[0]:+.3f}) "
              f"-- NOT checked by validator (direct-only model)")
    if not s['fragile_on_floor_ok']:
        ex = s['fragile_violation_example']
        print(f"        FRAGILE VIOLATION: {ex[0]} (mlot=0) carries {ex[1]:.3f}kg")
    if s['worst_validator'] is not None:
        wv = s['worst_validator']
        print(f"        validator-reported worst: {wv[1]} carries {wv[2]:.3f} "
              f"vs {wv[3]} (slack {wv[0]:+.3f})")


def main():
    warmup_jit()
    print("=== Load-bearing / fragility adversarial probe ===\n")

    SMALL = Pallet(length=1200, width=1000, height=1500, max_weight=2000)
    TALL = Pallet(length=1200, width=1000, height=2000, max_weight=5000)

    all_ok = True
    results = []

    print("--- Family A: pure fragile (mlot=0), must be on top/floor ---")
    for nf, ns in [(10, 2), (20, 3), (40, 4)]:
        ok, s = check_instance(f"A pure-fragile {nf}frag/{ns}base",
                               make_pure_fragile(nf, ns), SMALL,
                               PackerConfig(), max_pallets=4)
        print_summary(ok, s); results.append((ok, s)); all_ok &= ok

    print("\n--- Family B: small finite mlot + heavy tempters ---")
    for nn in [8, 16, 24]:
        ok, s = check_instance(f"B small-mlot+heavy n={nn}",
                               make_small_mlot_heavy(nn), SMALL,
                               PackerConfig(), max_pallets=5)
        print_summary(ok, s); results.append((ok, s)); all_ok &= ok

    print("\n--- Family C: mixed fragile/sturdy ---")
    for nn in [15, 30, 60]:
        ok, s = check_instance(f"C mixed n={nn}", make_mixed(nn), SMALL,
                               PackerConfig(), max_pallets=4)
        print_summary(ok, s); results.append((ok, s)); all_ok &= ok

    print("\n--- Family D: requires_full_support + tight mlot ---")
    for nn in [10, 20]:
        ok, s = check_instance(f"D rfs n={nn}", make_rfs(nn), SMALL,
                               PackerConfig(), max_pallets=4)
        print_summary(ok, s); results.append((ok, s)); all_ok &= ok

    print("\n--- Family E: tall homogeneous stack (direct vs cumulative) ---")
    ok, s = check_instance("E tall-stack mlot=25 w=20", make_tall_stack(), TALL,
                           PackerConfig(), max_pallets=2)
    print_summary(ok, s); results.append((ok, s)); all_ok &= ok

    print("\n--- Family F: float / non-integer dims (int-decode vs float-validate) ---")
    ok, s = check_instance("F float-dims mlot=9.5 w=7.4/44.2", make_float_dims(),
                           SMALL, PackerConfig(), max_pallets=4)
    print_summary(ok, s); results.append((ok, s)); all_ok &= ok

    print("\n--- Family G: enforce_load_bearing=False asymmetry ---")
    # decoder enforces because has_constraints=True (finite mlot); validator
    # skips the LB check. We assert: (a) no LB error reported (validator off),
    # AND (b) our independent DIRECT recompute also shows no violation (decoder
    # actually kept it safe even though validator wasn't watching).
    cfg_off = PackerConfig(enforce_load_bearing=False)
    ok, s = check_instance("G enforce=False small-mlot",
                           make_small_mlot_heavy(16), SMALL, cfg_off,
                           max_pallets=5)
    # For this family the meaningful gate is the independent recompute.
    indep_ok = (s['worst_independent'] is None
                or s['worst_independent'][0] >= -1e-3)
    print_summary(indep_ok, s); results.append((indep_ok, s)); all_ok &= indep_ok
    if not indep_ok:
        print("        NOTE: validator silent (enforce=False) but DECODER let a "
              "box exceed mlot -> real defect")

    print("\n--- Family H: boundary mlot (load == mlot exactly) ---")
    ok, s = check_instance("H boundary mlot=10 w=10", make_boundary(), SMALL,
                           PackerConfig(), max_pallets=4)
    print_summary(ok, s); results.append((ok, s)); all_ok &= ok

    print("\n--- Family I: the 10 industry cases (real mlot configs) ---")
    for c in industry_cases():
        ok, s = check_instance(c.name[:40], c.boxes, c.pallet, c.config,
                               max_pallets=10, seeds=(42,))
        print_summary(ok, s); results.append((ok, s)); all_ok &= ok

    # ----------------------------------------------------------------------
    n_fam = len(results)
    n_pass = sum(1 for ok, _ in results if ok)
    tot_lb = sum(s['lb_violations'] for _, s in results)
    tot_val = sum(s['validator_errs'] for _, s in results)
    print("\n=== AGGREGATE ===")
    print(f"  instances checked: {n_fam}  passed: {n_pass}  failed: {n_fam - n_pass}")
    print(f"  total validator load-bearing violations: {tot_lb}")
    print(f"  total validator errors (all kinds):       {tot_val}")
    # Report any family where cumulative (physical) load exceeds mlot even
    # though direct model passed — documentation/model gap, not a validator
    # disagreement.
    cum_gaps = [(s['label'], s['worst_cumulative']) for _, s in results
                if s['worst_cumulative'] is not None
                and s['worst_cumulative'][0] < -1e-3]
    if cum_gaps:
        print(f"  [model gap] {len(cum_gaps)} families have CUMULATIVE load > mlot "
              f"(validator uses DIRECT-only; PackerConfig doc says 'recursively'):")
        for lbl, wc in cum_gaps[:5]:
            print(f"      {lbl}: {wc[1]} bears {wc[2]:.1f}kg cumulative vs mlot {wc[3]:.1f}")

    print(f"\n=== RESULT: {'PASS' if all_ok else 'FAIL'} ===")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
