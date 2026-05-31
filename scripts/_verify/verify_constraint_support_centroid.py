"""verify_constraint_support_centroid.py — adversarial verification of
SUPPORT_RATIO + require_centroid_supported (no-floating-box stability).

Mandate: construct instances where partial-overhang placements would raise
density (tempting the engine to float boxes). Sweep support_ratio in
{0.5, 0.8, 1.0} x require_centroid_supported {True, False}. validate() must
report no 'floating' violation beyond the configured support_ratio. Verify
support_ratio=1.0 truly forbids ANY overhang; lower ratios still respected.
Independently check the centroid rule (the validator does NOT check it).

KEY ARCHITECTURE FACT exploited here:
  support_ratio + require_centroid are ONLY enforced when has_constraints=True
  (precompute_constraint_arrays: any finite max_load_on_top, any
  requires_full_support, or finite pallet.max_weight). So every adversarial
  instance below sets a finite max_load_on_top OR a finite pallet weight to
  ARM the constraint path. We ALSO add a control: support_ratio set but
  has_constraints=False, to demonstrate/measure the silent-disable trap.

Run inside Docker:
    docker run --rm -e OMP_NUM_THREADS=1 -e PYTHONHASHSEED=0 \
        -v "$(pwd)":/app -w /app pallet-packer:dev \
        python scripts/_verify/verify_constraint_support_centroid.py
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dataclasses import replace

import numpy as np

from pallet_packer import Box, Pallet, PackerConfig, ALL_ROTATIONS, validate
from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit
from pallet_packer._brkga_core.precompute import precompute_constraint_arrays

EPS = 1e-6


# ---------------------------------------------------------------------------
# Independent geometric re-checks (do NOT trust the engine).
# ---------------------------------------------------------------------------
def support_ratio_of(p, others):
    """Fraction of p's footprint resting on supporters directly beneath."""
    footprint = p.dx * p.dy
    if footprint <= 0:
        return 1.0
    supported = 0.0
    for q in others:
        if q is p:
            continue
        if abs(p.z - q.z2) > EPS:
            continue
        ox = max(0.0, min(p.x2, q.x2) - max(p.x, q.x))
        oy = max(0.0, min(p.y2, q.y2) - max(p.y, q.y))
        supported += ox * oy
    return supported / footprint


def centroid_supported(p, others):
    """True iff p's footprint centroid (cx, cy) lies over >=1 supporter beneath.

    Mirrors the engine rule in jit_constraints._check_load_on_top_njit:
    centroid (cand_x + dx/2, cand_y + dy/2) must lie within at least one
    supporter's XY footprint that touches z = p.z (i.e. supporter top == p.z).
    Floor boxes (z<=0) are trivially supported.
    """
    if p.z <= EPS:
        return True
    cx = p.x + p.dx / 2.0
    cy = p.y + p.dy / 2.0
    for q in others:
        if q is p:
            continue
        if abs(p.z - q.z2) > EPS:
            continue
        if (q.x - EPS <= cx <= q.x2 + EPS) and (q.y - EPS <= cy <= q.y2 + EPS):
            return True
    return False


def worst_overhang(result):
    """Max fraction of any non-floor box footprint that is unsupported."""
    worst = 0.0
    worst_box = None
    for st in result.pallets:
        ps = st.placements
        for p in ps:
            if p.z <= EPS:
                continue
            sr = support_ratio_of(p, ps)
            oh = 1.0 - sr
            if oh > worst:
                worst = oh
                worst_box = (st.pallet_id, p.box.id, sr)
    return worst, worst_box


def centroid_violations(result):
    """List of (pallet, box, cx, cy) for non-floor boxes whose centroid is
    NOT over any supporter beneath them."""
    out = []
    for st in result.pallets:
        ps = st.placements
        for p in ps:
            if p.z <= EPS:
                continue
            if not centroid_supported(p, ps):
                out.append((st.pallet_id, p.box.id,
                            p.x + p.dx / 2.0, p.y + p.dy / 2.0, p.z))
    return out


# ---------------------------------------------------------------------------
# Instance builders. All set a finite cap to ARM the constraint path.
# ---------------------------------------------------------------------------
def stacking_case(name, n_boxes, L, W, H, bx, by, bz, weight=5.0,
                  mlot=200.0, pallet_w=1e9, rots=None, pallet_h=None):
    """Many identical boxes that must stack (forces non-floor placements).
    Finite mlot arms has_constraints=True so support/centroid are enforced."""
    rots = rots or ALL_ROTATIONS
    boxes = [Box(id=f"{name}-{i}", length=bx, width=by, height=bz,
                 weight=weight, max_load_on_top=mlot,
                 allowed_rotations=rots)
             for i in range(n_boxes)]
    pal = Pallet(length=L, width=W, height=pallet_h or H, max_weight=pallet_w)
    return name, boxes, pal


def mixed_size_case(name, L, W, H, pallet_w=1e9):
    """Heterogeneous sizes — tempts the engine into partial-overhang stacks
    where a big box sits on a smaller one (low support ratio) to gain density."""
    boxes = []
    # big base-ish boxes
    for i in range(8):
        boxes.append(Box(id=f"{name}-BIG-{i}", length=400, width=380, height=200,
                         weight=8.0, max_load_on_top=300.0))
    # medium
    for i in range(12):
        boxes.append(Box(id=f"{name}-MED-{i}", length=300, width=200, height=180,
                         weight=4.0, max_load_on_top=150.0))
    # small (these could perch on edges of larger boxes)
    for i in range(20):
        boxes.append(Box(id=f"{name}-SML-{i}", length=180, width=140, height=160,
                         weight=2.0, max_load_on_top=80.0))
    pal = Pallet(length=L, width=W, height=H, max_weight=pallet_w)
    return name, boxes, pal


def noninteger_case(name, pallet_w=1e9):
    """Non-integer real-world dims: int-decode vs float-validate mismatch risk.
    Decoder rounds to int; validator uses float. With support_ratio=1.0 a box
    rounded UP in the decoder could appear over-hanging to the float validator."""
    boxes = []
    for i in range(30):
        boxes.append(Box(id=f"{name}-{i}",
                         length=137.6, width=98.4, height=123.9,
                         weight=3.0, max_load_on_top=120.0))
    pal = Pallet(length=1000.5, width=800.3, height=1200.7, max_weight=pallet_w)
    return name, boxes, pal


def run_solve(boxes, pallet, cfg, seed=42, t=4.0, max_pallets=4):
    return brkga_pack_v35(
        boxes, pallet, cfg,
        time_limit_s=t, max_pallets=max_pallets,
        population_size=120, n_populations=3, patience=120,
        local_search_budget_s=1.0, seed=seed, verbose=False, n_modes=6,
    )


# ---------------------------------------------------------------------------
def main():
    warmup_jit()
    print("=== verify_constraint_support_centroid ===\n")

    # Backend sanity.
    from pallet_packer._brkga_core import dispatch
    print(f"[backend] decode_njit_mode module: {dispatch.decode_njit_mode.__module__}")
    print(f"[backend] _BATCH_AVAILABLE = {dispatch._BATCH_AVAILABLE}\n")

    overall_ok = True
    findings = []
    n_cases = 0
    n_floating = 0
    n_centroid_viol = 0
    n_overhang_at_sr1 = 0
    worst_floating = (0.0, None, None)   # (margin_over_ratio, case, box)
    worst_overhang_global = 0.0

    # Build the adversarial instance set.
    builders = [
        stacking_case("tall_stack", 40, 800, 600, 1400, 200, 150, 120),
        stacking_case("cube_tower", 60, 600, 600, 1500, 150, 150, 150),
        stacking_case("flat_tile", 50, 1000, 800, 1000, 250, 200, 100),
        stacking_case("skinny_tall", 30, 500, 500, 1500, 120, 110, 300),
        stacking_case("weight_cap", 50, 800, 800, 1400, 180, 170, 150,
                      weight=20.0, pallet_w=600.0),  # pallet cap arms cstr
        mixed_size_case("mixed_a", 1000, 800, 1400),
        mixed_size_case("mixed_b", 1200, 1000, 1500, pallet_w=500.0),
        noninteger_case("nonint"),
    ]

    # support_ratio x require_centroid sweep.
    sr_values = [0.5, 0.8, 1.0]
    rc_values = [True, False]

    for cname, boxes, pallet in builders:
        # Confirm the constraint path is actually armed for this instance.
        _, _, _, _, has_cstr = precompute_constraint_arrays(boxes, pallet)
        if not has_cstr:
            print(f"[WARN] {cname}: has_constraints=False — support NOT enforced!")
        for sr in sr_values:
            for rc in rc_values:
                n_cases += 1
                cfg = PackerConfig(
                    support_ratio=sr,
                    require_centroid_supported=rc,
                    # turn OFF cog envelope so it doesn't confound the support
                    # measurement (we test cog elsewhere); also off load-bearing
                    # double-count is fine — keep defaults sensible.
                    cog_envelope_fraction=1.0,   # disables cog_active
                    enforce_load_bearing=True,
                    max_pallets=4,
                    optimize="max_util",
                )
                r = run_solve(boxes, pallet, cfg)
                errs = validate(r, pallet, cfg)
                placed = sum(len(s.placements) for s in r.pallets)

                # (a) validator floating violations
                float_errs = [e for e in errs if "floating" in e]
                other_errs = [e for e in errs if "floating" not in e]

                # (b) independent support measurement
                woh, wob = worst_overhang(r)
                worst_overhang_global = max(worst_overhang_global, woh)
                # achieved min support ratio across all non-floor boxes
                min_sr = 1.0 - woh

                # (c) independent centroid check (only meaningful when rc=True
                #     AND constraints armed)
                cviol = centroid_violations(r) if (rc and has_cstr) else []

                tag = f"{cname:11s} sr={sr} rc={int(rc)}"
                status = "ok"
                if float_errs:
                    status = "FLOATING"
                    n_floating += 1
                    overall_ok = False
                    margin = (sr - min_sr)
                    if margin > worst_floating[0]:
                        worst_floating = (margin, tag, float_errs[0])
                    findings.append(f"{tag}: {len(float_errs)} floating; "
                                    f"min_sr_achieved={min_sr:.3f} < cfg {sr}; "
                                    f"e.g. {float_errs[0]}")
                if other_errs:
                    status = "OTHER_ERR"
                    overall_ok = False
                    findings.append(f"{tag}: non-floating validator errors: "
                                    f"{other_errs[:2]}")
                if has_cstr and cviol:
                    # Centroid rule should hold when rc=True and armed.
                    n_centroid_viol += len(cviol)
                    overall_ok = False
                    status = "CENTROID"
                    findings.append(f"{tag}: {len(cviol)} centroid violations; "
                                    f"e.g. box {cviol[0][1]} centroid "
                                    f"({cviol[0][2]:.1f},{cviol[0][3]:.1f}) z={cviol[0][4]}")

                # (d) support_ratio=1.0 MUST forbid ANY overhang (when armed).
                if has_cstr and abs(sr - 1.0) < EPS and woh > EPS:
                    n_overhang_at_sr1 += 1
                    overall_ok = False
                    status = "OVERHANG@1.0"
                    findings.append(f"{tag}: sr=1.0 but worst overhang "
                                    f"{woh:.4f} (box {wob})")

                print(f"  [{status:11s}] {tag}  placed={placed:3d} "
                      f"pallets={len(r.pallets)} min_sr={min_sr:.3f} "
                      f"float_err={len(float_errs)} cviol={len(cviol)} "
                      f"other_err={len(other_errs)}")

    # ----- Control: support_ratio set but has_constraints=False (the trap) ---
    print("\n=== CONTROL: support_ratio with has_constraints=False ===")
    # Pure-geometric: infinite mlot, infinite pallet weight, zero weight.
    geo_boxes = [Box(id=f"G-{i}", length=150, width=150, height=150,
                     weight=0.0, max_load_on_top=float("inf"))
                 for i in range(60)]
    geo_pal = Pallet(length=600, width=600, height=1500, max_weight=float("inf"))
    _, _, _, _, has_cstr_geo = precompute_constraint_arrays(geo_boxes, geo_pal)
    print(f"  has_constraints={has_cstr_geo} (expect False)")
    cfg_geo = PackerConfig(support_ratio=1.0, require_centroid_supported=True,
                           cog_envelope_fraction=1.0, enforce_load_bearing=False,
                           max_pallets=1, optimize="max_util")
    rg = run_solve(geo_boxes, geo_pal, cfg_geo, max_pallets=1)
    errs_g = validate(rg, geo_pal, cfg_geo)
    fe_g = [e for e in errs_g if "floating" in e]
    woh_g, wob_g = worst_overhang(rg)
    placed_g = sum(len(s.placements) for s in rg.pallets)
    print(f"  geom solve: placed={placed_g} worst_overhang={woh_g:.4f} "
          f"validator_floating={len(fe_g)}")
    if not has_cstr_geo and fe_g:
        # This is the documented trap: support_ratio silently disabled while
        # the validator still enforces it. Report as a real defect surface.
        overall_ok = False
        findings.append(
            f"TRAP: has_constraints=False disables support enforcement, yet "
            f"validate() flags {len(fe_g)} floating boxes for sr=1.0 (worst "
            f"overhang {woh_g:.3f}). A purely-geometric config that requests "
            f"support_ratio=1.0 produces validator-invalid results.")
        print(f"  *** TRAP CONFIRMED: {len(fe_g)} validator floating errors "
              f"despite support requested ***")
    elif not has_cstr_geo and not fe_g:
        print("  (no floating in this particular geom layout — trap latent "
              "but not triggered here)")

    # ----- Targeted minimal centroid stress -----
    print("\n=== TARGETED: centroid rule under tight stacking (armed) ===")
    # Boxes that, if perched off-centre on a single supporter, would float.
    cboxes = [Box(id=f"C-{i}", length=200, width=200, height=100,
                  weight=2.0, max_load_on_top=50.0) for i in range(48)]
    cpal = Pallet(length=620, width=620, height=900, max_weight=1e9)
    for sr in (0.5, 1.0):
        for rc in (True, False):
            cfg = PackerConfig(support_ratio=sr, require_centroid_supported=rc,
                               cog_envelope_fraction=1.0, max_pallets=1,
                               optimize="max_util")
            r = run_solve(cboxes, cpal, cfg, max_pallets=1, t=4.0)
            errs = validate(r, cpal, cfg)
            fe = [e for e in errs if "floating" in e]
            cviol = centroid_violations(r) if rc else []
            woh, _ = worst_overhang(r)
            placed = sum(len(s.placements) for s in r.pallets)
            bad = bool(fe) or bool(cviol) or (abs(sr - 1.0) < EPS and woh > EPS)
            if bad:
                overall_ok = False
                findings.append(f"centroid-stress sr={sr} rc={int(rc)}: "
                                f"float={len(fe)} cviol={len(cviol)} woh={woh:.4f}")
            print(f"  sr={sr} rc={int(rc)}: placed={placed} float={len(fe)} "
                  f"cviol={len(cviol)} worst_overhang={woh:.4f} "
                  f"{'BAD' if bad else 'ok'}")

    # ----- MINIMAL 2-box repro of the int-decode vs float-validate defect ---
    print("\n=== MINIMAL REPRO: non-integer height under sr=1.0 (2 boxes) ===")
    from pallet_packer import NO_ROTATION
    print(f"  int(round(98.4)) = {int(round(98.4))}  (engine dz=98, float dz=98.4)")
    mboxes = [
        Box(id="A", length=200.0, width=200.0, height=98.4, weight=1.0,
            max_load_on_top=50.0, allowed_rotations=NO_ROTATION),
        Box(id="B", length=200.0, width=200.0, height=98.4, weight=1.0,
            max_load_on_top=50.0, allowed_rotations=NO_ROTATION),
    ]
    mpal = Pallet(length=200.0, width=200.0, height=400.0, max_weight=1e9)
    mcfg = PackerConfig(support_ratio=1.0, require_centroid_supported=True,
                        cog_envelope_fraction=1.0, max_pallets=1,
                        optimize="max_util")
    rm = brkga_pack_v35(mboxes, mpal, mcfg, time_limit_s=2.0, max_pallets=1,
                        population_size=40, n_populations=2, patience=40,
                        local_search_budget_s=0.5, seed=42, verbose=False, n_modes=6)
    me = validate(rm, mpal, mcfg)
    for st in rm.pallets:
        for p in st.placements:
            print(f"    {p.box.id}: z=[{p.z:.3f},{p.z2:.3f}] dz={p.dz:.3f}")
    print(f"  validator errors: {len(me)}  -> {me}")
    if me:
        overall_ok = False
        findings.append(
            "MINIMAL: 2 boxes h=98.4 stacked at sr=1.0 -> engine rounds dz to "
            "98, places B at z=98; validator (float dz=98.4) reports A/B "
            "overlap by 0.4mm AND B floating. Root cause: dims_all uses "
            "int(round(dim)) while validate() uses original float Box dims.")
        print("  *** DEFECT REPRODUCED (int-decode vs float-validate) ***")

    # ----- Summary -----
    print("\n=== SUMMARY ===")
    print(f"  sweep cases run        : {n_cases}")
    print(f"  validator-floating viol: {n_floating}")
    print(f"  centroid violations    : {n_centroid_viol}")
    print(f"  overhang@sr=1.0 cases  : {n_overhang_at_sr1}")
    print(f"  worst overhang (global): {worst_overhang_global:.4f}")
    if worst_floating[1]:
        print(f"  worst floating         : {worst_floating[1]} -> {worst_floating[2]}")
    if findings:
        print("\n  FINDINGS:")
        for f in findings[:20]:
            print(f"   - {f}")
    print(f"\n=== RESULT: {'PASS' if overall_ok else 'FAIL'} ===")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
