"""verify_constraint_weight.py — adversarially verify PALLET WEIGHT CAP enforcement.

Mandate: weight is the BINDING constraint. Construct instances where overfilling a
single pallet would raise utilization (so the algorithm is tempted to violate the
cap), then check the GROUND-TRUTH validator never reports a pallet whose total box
weight exceeds pallet.max_weight. Also confirm overflow correctly goes to additional
pallets / unpacked, and that max_weight=inf makes weight irrelevant.

We DO NOT trust the solver's self-report; we recompute each pallet's total weight
independently from the PackResult placements and compare against the cap, AND we run
the independent validate() (which has its own weight check, validate.py rule #3).

Run (inside Docker, from repo root):
    docker run --rm -e OMP_NUM_THREADS=1 -e PYTHONHASHSEED=0 \
        -v "$(pwd)":/app -w /app pallet-packer:dev \
        python scripts/_verify/verify_constraint_weight.py
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

from pallet_packer import (
    Box, Pallet, PackerConfig, validate,
    ALL_ROTATIONS, THIS_SIDE_UP,
)
from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit

EPS = 1e-6

# ---------------------------------------------------------------------------
# Independent weight recomputation from a PackResult — DOES NOT trust the engine.
# ---------------------------------------------------------------------------
def pallet_weights(result) -> List[float]:
    """Recompute each pallet's total box weight directly from placements."""
    return [sum(p.box.weight for p in st.placements) for st in result.pallets]


def weight_overruns(result, cap: float) -> List[Tuple[int, float]]:
    """Return [(pallet_index, total_weight)] for pallets that exceed cap+EPS."""
    out = []
    for i, w in enumerate(pallet_weights(result)):
        if cap < float("inf") and w > cap + EPS:
            out.append((i, w))
    return out


def all_box_ids(boxes) -> set:
    return {b.id for b in boxes}


def placed_box_ids(result) -> list:
    ids = []
    for st in result.pallets:
        for p in st.placements:
            ids.append(p.box.id)
    return ids


def accounting_ok(result, boxes) -> Tuple[bool, str]:
    """Every input box must appear exactly once across placed ∪ unpacked."""
    placed = placed_box_ids(result)
    unp = [b.id for b in result.unpacked]
    seen = placed + unp
    inp = [b.id for b in boxes]
    # duplicates?
    if len(set(placed)) != len(placed):
        dups = [x for x in set(placed) if placed.count(x) > 1]
        return False, f"DUPLICATE placed ids: {dups[:5]}"
    missing = set(inp) - set(seen)
    extra = set(seen) - set(inp)
    if missing:
        return False, f"MISSING ids (lost boxes): {sorted(missing)[:5]}"
    if extra:
        return False, f"EXTRA ids (phantom boxes): {sorted(extra)[:5]}"
    if len(seen) != len(inp):
        return False, f"count mismatch: input={len(inp)} placed+unp={len(seen)}"
    return True, "ok"


def solve(boxes, pallet, config, *, max_pallets, seed=42, t=6.0,
          use_v2_seed=None):
    return brkga_pack_v35(
        boxes, pallet, config,
        time_limit_s=t, max_pallets=max_pallets, seed=seed,
        population_size=150, n_populations=2, patience=120,
        local_search_budget_s=1.5, use_v2_seed=use_v2_seed,
        n_modes=6, verbose=False,
    )


@dataclass
class CaseResult:
    name: str
    cap: float
    n_pallets: int
    n_unpacked: int
    pallet_ws: List[float]
    overruns: List[Tuple[int, float]]
    validator_errs: int
    validator_weight_errs: int
    acct_ok: bool
    acct_msg: str
    extra: str = ""

    @property
    def passed(self) -> bool:
        return (not self.overruns and self.validator_weight_errs == 0
                and self.acct_ok)


def run_case(name, boxes, pallet, config, *, max_pallets, seed=42, t=6.0,
             use_v2_seed=None, expect_extra_check=None) -> CaseResult:
    r = solve(boxes, pallet, config, max_pallets=max_pallets, seed=seed, t=t,
              use_v2_seed=use_v2_seed)
    cap = pallet.max_weight
    pw = pallet_weights(r)
    ovr = weight_overruns(r, cap)
    errs = validate(r, pallet, config)
    werrs = [e for e in errs if "total weight" in e]
    acct_ok, acct_msg = accounting_ok(r, boxes)
    extra = ""
    if expect_extra_check is not None:
        extra = expect_extra_check(r, pallet, config)
    cr = CaseResult(
        name=name, cap=cap, n_pallets=len(r.pallets),
        n_unpacked=len(r.unpacked), pallet_ws=pw, overruns=ovr,
        validator_errs=len(errs), validator_weight_errs=len(werrs),
        acct_ok=acct_ok, acct_msg=acct_msg, extra=extra,
    )
    return cr


def report(cr: CaseResult):
    status = "PASS" if cr.passed else "FAIL"
    capstr = "inf" if cr.cap == float("inf") else f"{cr.cap:g}"
    maxw = max(cr.pallet_ws) if cr.pallet_ws else 0.0
    print(f"[{status}] {cr.name}")
    print(f"    cap={capstr}  pallets={cr.n_pallets} unpacked={cr.n_unpacked}  "
          f"max_pallet_weight={maxw:.2f}")
    ws = "  ".join(f"P{i}={w:.1f}" for i, w in enumerate(cr.pallet_ws))
    print(f"    per-pallet weights: [{ws}]")
    if cr.overruns:
        print(f"    !!! WEIGHT-CAP OVERRUN (independent recompute): "
              f"{[(i, round(w,2)) for i,w in cr.overruns]} > cap {capstr}")
    if cr.validator_weight_errs:
        print(f"    !!! validator weight errors: {cr.validator_weight_errs} "
              f"(total validator errs: {cr.validator_errs})")
    if not cr.acct_ok:
        print(f"    !!! box accounting: {cr.acct_msg}")
    if cr.extra:
        print(f"    note: {cr.extra}")
    return cr.passed


def main() -> int:
    warmup_jit()
    all_pass = True
    results: List[CaseResult] = []

    print("=" * 78)
    print("CONSTRAINT-WEIGHT ADVERSARIAL VERIFICATION")
    print("=" * 78)

    # ----------------------------------------------------------------------
    # GROUP A: uniform heavy boxes, total weight >> single-pallet cap.
    # Geometry is loose (boxes are small relative to pallet) so the solver
    # is tempted to overfill one pallet to maximize density. Vary tightness.
    # ----------------------------------------------------------------------
    print("\n--- GROUP A: uniform heavy boxes, weight-binding, vary cap ---")
    # Box: 300x300x300, 40kg. 60 boxes => 2400kg total. Pallet 1200x1000x1500
    # volumetric capacity ~ huge (60 such boxes fit by volume on ~2 pallets).
    def make_heavy(n, wt, l=300, w=300, h=300):
        return [Box(id=f"H{i:03d}", length=l, width=w, height=h, weight=wt,
                    allowed_rotations=ALL_ROTATIONS, max_load_on_top=float("inf"))
                for i in range(n)]

    # Note: max_load_on_top=inf on purpose => the ONLY active constraint is the
    # pallet weight cap. If has_constraints logic or the cap check is buggy,
    # nothing else masks it.
    A_specs = [
        # (cap, n_boxes, box_weight) — vary tightness
        (200.0, 30, 40.0),   # cap=5 boxes/pallet
        (500.0, 30, 40.0),   # cap=12 boxes/pallet
        (1000.0, 30, 40.0),  # cap=25 boxes/pallet
        (43.0, 20, 40.0),    # cap just above one box (1 box/pallet)
        (40.0, 20, 40.0),    # cap == exactly one box
        (39.5, 12, 40.0),    # cap BELOW one box => ALL must be unpacked
        (80.0, 24, 40.0),    # cap == exactly 2 boxes
        (81.0, 24, 40.0),    # cap just above 2 boxes
        (120.0, 24, 40.0),   # cap == exactly 3 boxes
    ]
    for cap, n, wt in A_specs:
        boxes = make_heavy(n, wt)
        pallet = Pallet(length=1200, width=1000, height=1500, max_weight=cap)
        cr = run_case(f"A cap={cap:g} n={n} wt={wt:g}", boxes, pallet,
                      PackerConfig(), max_pallets=20, t=5.0)
        results.append(cr)
        all_pass &= report(cr)
        # Adversarial extra: per-pallet count must respect floor(cap/wt).
        max_per = int(math.floor(cap / wt + EPS))
        for i, pw in enumerate(cr.pallet_ws):
            cnt = round(pw / wt)
            if cap < float("inf") and cnt > max_per:
                print(f"    !!! pallet {i} holds {cnt} boxes > "
                      f"floor(cap/wt)={max_per}")
                all_pass = False
        # If cap < one box, EVERYTHING must be unpacked, zero pallets.
        if cap < wt - EPS:
            if cr.n_pallets > 0 and any(w > EPS for w in cr.pallet_ws):
                print(f"    !!! cap<one-box but {cr.n_pallets} pallet(s) loaded")
                all_pass = False
            else:
                print(f"    ok: cap<one-box => all {cr.n_unpacked} unpacked")

    # ----------------------------------------------------------------------
    # GROUP B: heterogeneous heavy mix, full rotation, tight cap.
    # Mimics IND8 automotive: weight binds well before volume.
    # ----------------------------------------------------------------------
    print("\n--- GROUP B: heterogeneous heavy mix, tight caps ---")
    for seed_off, cap in enumerate([300.0, 500.0, 700.0]):
        rng = random.Random(1000 + seed_off)
        boxes = []
        for i in range(50):
            l = rng.randint(200, 450)
            w = rng.randint(150, 400)
            h = rng.randint(100, 350)
            wt = rng.uniform(8.0, 30.0)
            boxes.append(Box(id=f"B{i:03d}", length=l, width=w, height=h,
                             weight=wt, allowed_rotations=ALL_ROTATIONS,
                             max_load_on_top=float("inf")))
        total = sum(b.weight for b in boxes)
        pallet = Pallet(length=1200, width=1000, height=1500, max_weight=cap)
        cr = run_case(f"B cap={cap:g} total={total:.0f}", boxes, pallet,
                      PackerConfig(), max_pallets=30, t=6.0, seed=7 + seed_off)
        results.append(cr)
        ok = report(cr)
        all_pass &= ok
        # weight LB on pallet count
        lb = math.ceil(total / cap)
        placed_w = sum(cr.pallet_ws)
        print(f"    total_input={total:.0f}  placed_weight={placed_w:.0f}  "
              f"weight-LB pallets={lb}  used pallets={cr.n_pallets}")

    # ----------------------------------------------------------------------
    # GROUP C: weight cap + load-bearing together (realistic). The cstr path
    # has mode-delegation rules (mode5->cstr4); make sure weight still holds.
    # ----------------------------------------------------------------------
    print("\n--- GROUP C: weight cap + finite max_load_on_top (full cstr path) ---")
    for seed_off, cap in enumerate([400.0, 600.0]):
        boxes = []
        # uniform stackable beverage-like, but heavy enough to bind weight
        for i in range(40):
            boxes.append(Box(id=f"C{i:03d}", length=400, width=300, height=250,
                             weight=18.0, allowed_rotations=THIS_SIDE_UP,
                             max_load_on_top=54.0))  # 3 on top
        total = sum(b.weight for b in boxes)
        pallet = Pallet(length=1200, width=1000, height=1500, max_weight=cap)
        cr = run_case(f"C cap={cap:g} total={total:.0f} (mlot finite)", boxes,
                      pallet, PackerConfig(), max_pallets=30, t=6.0,
                      seed=11 + seed_off)
        results.append(cr)
        all_pass &= report(cr)

    # ----------------------------------------------------------------------
    # GROUP D: SINGLE-PALLET cap (max_pallets=1) — overflow must go to unpacked,
    # NOT exceed the cap. This is the most adversarial: solver wants to maximize
    # utilization on the one allowed pallet, weight is the only thing stopping it.
    # ----------------------------------------------------------------------
    # NOTE: a SEPARATE (orthogonal) defect surfaces here — the kwarg
    # max_pallets=1 is NOT honored by the v2-hybrid-polish path (it returns the
    # full multi-pallet v2 PalletPacker result). That is a max_pallets contract
    # bug, NOT a weight-cap bug: every spilled pallet still respects the weight
    # cap. We record it via maxpallets_violations but do NOT let it fail the
    # WEIGHT verdict, which is this dimension's mandate.
    print("\n--- GROUP D: max_pallets=1, weight forces unpacked ---")
    maxpallets_violations = []
    for cap in [200.0, 400.0, 1000.0]:
        boxes = make_heavy(40, 40.0)  # 1600kg total
        pallet = Pallet(length=1200, width=1000, height=1500, max_weight=cap)
        cr = run_case(f"D single-pallet cap={cap:g} (1600kg input)", boxes,
                      pallet, PackerConfig(), max_pallets=1, t=6.0)
        results.append(cr)
        all_pass &= report(cr)  # weight verdict only
        if cr.n_pallets > 1:
            print(f"    [orthogonal] max_pallets=1 NOT honored: {cr.n_pallets} "
                  f"pallets returned (weight cap still respected). "
                  f"Cause: v2-hybrid-polish (driver.py:608) returns full v2 result.")
            maxpallets_violations.append((cap, cr.n_pallets))
        # The one pallet must be <= cap, surplus boxes unpacked.
        expect_unp = len(boxes) - int(math.floor(cap / 40.0 + EPS))
        if cr.n_unpacked < expect_unp:
            # fewer unpacked than the weight cap allows means an overrun MUST
            # have happened (caught above) — but flag explicitly.
            print(f"    note: unpacked={cr.n_unpacked} < weight-forced "
                  f"min {expect_unp} (suspicious unless geometry also binds)")

    # ----------------------------------------------------------------------
    # GROUP E: CONTROL — max_weight=inf must make weight irrelevant.
    # Same heavy boxes, infinite cap: no weight error possible, and the solver
    # should pack far more per pallet than the finite-cap runs did.
    # ----------------------------------------------------------------------
    print("\n--- GROUP E: max_weight=inf => weight ignored (control) ---")
    boxes = make_heavy(20, 1000.0)  # 20 tonnes total, but cap infinite
    pallet_inf = Pallet(length=1200, width=1000, height=1500,
                        max_weight=float("inf"))
    cr_inf = run_case("E inf-cap heavy (20x1000kg)", boxes, pallet_inf,
                      PackerConfig(), max_pallets=1, t=6.0)
    results.append(cr_inf)
    all_pass &= report(cr_inf)
    # With inf cap the validator must NEVER flag weight, and these boxes
    # (300^3) easily fit >1 on a pallet by volume; expect them packed densely.
    if cr_inf.validator_weight_errs:
        print("    !!! inf cap produced a weight error — impossible, real bug")
        all_pass = False
    # Sanity: compare to a tight finite cap run with same boxes; finite should
    # pack fewer per pallet.
    pallet_tight = Pallet(length=1200, width=1000, height=1500, max_weight=3000.0)
    cr_tight = run_case("E' same boxes cap=3000", boxes, pallet_tight,
                        PackerConfig(), max_pallets=20, t=6.0)
    results.append(cr_tight)
    all_pass &= report(cr_tight)
    maxw_inf = max(cr_inf.pallet_ws) if cr_inf.pallet_ws else 0.0
    print(f"    inf-cap max pallet weight={maxw_inf:.0f}  "
          f"(cap=3000 max pallet weight={max(cr_tight.pallet_ws) if cr_tight.pallet_ws else 0:.0f})")
    if maxw_inf <= 3000.0 + EPS and len(boxes) > 3:
        print("    note: inf-cap did not exceed 3000 on a single pallet — "
              "geometry/height may bind before weight; not a defect by itself")

    # ----------------------------------------------------------------------
    # GROUP F: ADVERSARIAL EDGE — fractional / non-integer cap & zero-weight.
    # ----------------------------------------------------------------------
    print("\n--- GROUP F: edge caps (fractional, near-exact, zero-weight) ---")
    # F1: cap exactly equals total weight of a subset that tiles the floor.
    boxes = [Box(id=f"F{i:03d}", length=300, width=250, height=200,
                 weight=12.5, allowed_rotations=ALL_ROTATIONS,
                 max_load_on_top=float("inf")) for i in range(30)]
    palF = Pallet(length=1200, width=1000, height=1500, max_weight=100.0)  # 8 boxes exactly
    crF1 = run_case("F1 cap=100 wt=12.5 (8 exactly)", boxes, palF,
                    PackerConfig(), max_pallets=20, t=5.0)
    results.append(crF1)
    all_pass &= report(crF1)
    for i, pw in enumerate(crF1.pallet_ws):
        if pw > 100.0 + EPS:
            print(f"    !!! pallet {i} weight {pw} > 100")
            all_pass = False

    # F2: fractional cap, fractional weights (real-world kg).
    rng = random.Random(55)
    boxes = [Box(id=f"G{i:03d}", length=rng.randint(200, 400),
                 width=rng.randint(200, 350), height=rng.randint(150, 300),
                 weight=round(rng.uniform(3.3, 9.7), 2),
                 allowed_rotations=ALL_ROTATIONS, max_load_on_top=float("inf"))
             for i in range(45)]
    palF2 = Pallet(length=1200, width=1000, height=1500, max_weight=137.77)
    crF2 = run_case("F2 fractional cap=137.77 fractional weights", boxes, palF2,
                    PackerConfig(), max_pallets=30, t=6.0)
    results.append(crF2)
    all_pass &= report(crF2)

    # F3: zero-weight boxes with a finite cap (weight constraint present but
    # never binds) — must not spuriously reject and must remain valid.
    boxes = [Box(id=f"Z{i:03d}", length=250, width=250, height=250, weight=0.0,
                 allowed_rotations=ALL_ROTATIONS, max_load_on_top=float("inf"))
             for i in range(30)]
    palF3 = Pallet(length=1200, width=1000, height=1500, max_weight=10.0)
    crF3 = run_case("F3 zero-weight boxes, finite cap=10", boxes, palF3,
                    PackerConfig(), max_pallets=5, t=5.0)
    results.append(crF3)
    all_pass &= report(crF3)
    # zero-weight: cap never binds, so a single pallet should accept many.
    if crF3.n_unpacked == len(boxes):
        print("    !!! zero-weight boxes all unpacked under finite cap — "
              "weight check wrongly rejecting massless boxes")
        all_pass = False

    # ----------------------------------------------------------------------
    # GROUP G: official weight-binding industry cases (IND4, IND8, IND9, IND2).
    # ----------------------------------------------------------------------
    print("\n--- GROUP G: industry weight-binding cases ---")
    from benchmarks.industry import cases as industry_cases
    cs = industry_cases()
    for idx in (1, 3, 7, 8):  # IND2 pharma, IND4 beverage, IND8 auto, IND9 docs
        c = cs[idx]
        r = brkga_pack_v35(c.boxes, c.pallet, c.config, time_limit_s=8.0,
                           max_pallets=15, seed=42, population_size=200,
                           n_populations=3, patience=150, verbose=False,
                           n_modes=6)
        cap = c.pallet.max_weight
        pw = pallet_weights(r)
        ovr = weight_overruns(r, cap)
        errs = validate(r, c.pallet, c.config)
        werrs = [e for e in errs if "total weight" in e]
        acct_ok, acct_msg = accounting_ok(r, c.boxes)
        cr = CaseResult(name=c.name, cap=cap, n_pallets=len(r.pallets),
                        n_unpacked=len(r.unpacked), pallet_ws=pw, overruns=ovr,
                        validator_errs=len(errs), validator_weight_errs=len(werrs),
                        acct_ok=acct_ok, acct_msg=acct_msg)
        results.append(cr)
        all_pass &= report(cr)
        if errs and not werrs:
            print(f"    (note: {len(errs)} non-weight validator errs: {errs[0]})")

    # ----------------------------------------------------------------------
    # SUMMARY
    # ----------------------------------------------------------------------
    print("\n" + "=" * 78)
    n_total = len(results)
    n_fail = sum(1 for r in results if not r.passed)
    n_ovr = sum(len(r.overruns) for r in results)
    n_werr = sum(r.validator_weight_errs for r in results)
    n_acct = sum(1 for r in results if not r.acct_ok)
    worst = None
    for r in results:
        for i, w in r.overruns:
            slack = w - r.cap
            if worst is None or slack > worst[1]:
                worst = (r.name, slack, w, r.cap)
    print(f"CASES: {n_total}  FAILED(weight): {n_fail}  "
          f"weight-cap overruns(independent): {n_ovr}  "
          f"validator-weight-errs: {n_werr}  accounting-fails: {n_acct}")
    if worst:
        print(f"WORST OVERRUN: {worst[0]}: weight {worst[2]:.2f} > cap "
              f"{worst[3]:g} (slack +{worst[1]:.2f} kg)")
    else:
        print("WORST OVERRUN: none — no pallet ever exceeded its weight cap.")
    if maxpallets_violations:
        print(f"ORTHOGONAL (non-weight) defect: max_pallets=1 not honored in "
              f"{len(maxpallets_violations)} case(s): {maxpallets_violations} "
              f"(weight cap still respected in every spilled pallet).")
    weight_pass = (n_ovr == 0 and n_werr == 0 and n_acct == 0)
    print(f"\n=== WEIGHT-MANDATE VERDICT: {'PASS' if weight_pass else 'FAIL'} ===")
    print(f"=== OVERALL (incl. orthogonal max_pallets check): "
          f"{'PASS' if all_pass else 'FAIL'} ===")
    return 0 if weight_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
