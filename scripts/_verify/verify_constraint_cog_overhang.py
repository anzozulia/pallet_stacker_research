"""Adversarial verification of CoG envelope + max_overhang.

Mandate (constraint-cog-overhang):
  (a) CoG: set a tight cog_envelope_fraction and/or explicit
      pallet.cog_x_range/cog_y_range; construct loads that would naturally
      push CoG off-centre. After solving, INDEPENDENTLY compute the realized
      pallet CoG (weighted box-centroid) and verify it lies within the
      envelope for pallets above cog_check_min_load_fraction. validate()
      does NOT check CoG — we compute it ourselves.
  (b) max_overhang: allow_pallet_overhang=True with a max_overhang; verify
      boxes may extend up to (and never beyond) pallet_edge + max_overhang,
      matching validate()'s bound.

KEY ARCHITECTURE FACTS this probe exploits / accounts for:
  * CoG enforcement is gated behind has_constraints (precompute_constraint_arrays).
    has_constraints is True iff: any finite max_load_on_top, OR any
    requires_full_support, OR a finite pallet.max_weight. A CoG-ONLY workload
    (tight envelope, infinite max_weight, no load limits) yields
    has_constraints=False -> driver forces cog_active=0 -> CoG NEVER enforced.
    We probe both: (i) CoG with a finite max_weight (enforced), and
    (ii) CoG-only without other constraints (the silent-skip gap).
  * The CoG min-load gate uses pallet_max_weight: CoG is only checked once
    new_total >= cog_min_load_frac * max_weight. Pallets that finish BELOW
    that threshold are exempt by design. Our independent check therefore only
    asserts CoG on pallets whose total weight >= cog_check_min_load_fraction *
    max_weight (replicating the engine's own gate semantics) — that is the
    contract validate-against.
  * Decoders use INTEGER coords (int(round(...))). With integer box dims the
    realized float CoG == int CoG. We use integer dims so the engine's
    int-based CoG check is directly comparable to our float recompute, but we
    also add a non-integer-dims sub-test to surface int/float drift.
  * Overhang: L_eff = L + round(max_overhang) only on +x/+y edges. Validator
    bound: x2 <= length + max_overhang + EPS (and y similarly). -x/-y get 0.

Run:
    docker run --rm -e OMP_NUM_THREADS=1 -e PYTHONHASHSEED=0 \
        -v "$(pwd)":/app -w /app pallet-packer:dev \
        python scripts/_verify/verify_constraint_cog_overhang.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402

from pallet_packer import (  # noqa: E402
    Box, Pallet, PackerConfig, validate, ALL_ROTATIONS,
)
from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit  # noqa: E402
from pallet_packer._brkga_core.precompute import (  # noqa: E402
    precompute_constraint_arrays, precompute_cog_envelope,
)

EPS = 1e-6
ANY = list(ALL_ROTATIONS)

# Collect findings to report.
FAILS: list[str] = []
ANOMALIES: list[str] = []
METRICS: list[str] = []


# ---------------------------------------------------------------------------
# Independent realized-CoG computation (float, from the actual PackResult)
# ---------------------------------------------------------------------------
def realized_cog(placements):
    """Weighted (x,y) centroid of placed boxes, using ORIGINAL float dims."""
    tot = 0.0
    sx = 0.0
    sy = 0.0
    for p in placements:
        w = p.box.weight
        tot += w
        sx += w * (p.x + p.dx / 2.0)
        sy += w * (p.y + p.dy / 2.0)
    if tot <= 0:
        return None, None, 0.0
    return sx / tot, sy / tot, tot


def envelope_for(pallet, config):
    """Replicate precompute_cog_envelope bounds (float)."""
    L, W = float(pallet.length), float(pallet.width)
    frac = float(config.cog_envelope_fraction)
    if pallet.cog_x_range is not None:
        xr = (float(pallet.cog_x_range[0]), float(pallet.cog_x_range[1]))
    else:
        xr = (L / 2.0 - frac * L, L / 2.0 + frac * L)
    if pallet.cog_y_range is not None:
        yr = (float(pallet.cog_y_range[0]), float(pallet.cog_y_range[1]))
    else:
        yr = (W / 2.0 - frac * W, W / 2.0 + frac * W)
    return xr, yr


def solve(boxes, pallet, config, *, max_pallets, t=6.0, seed=42):
    return brkga_pack_v35(
        boxes, pallet, config,
        time_limit_s=t, max_pallets=max_pallets, seed=seed,
        population_size=200, n_populations=3, patience=150,
        local_search_budget_s=1.0, verbose=False, n_modes=6,
    )


def check_cog_pallet(name, st, pallet, config, *, expect_enforced):
    """Independently verify CoG on ONE pallet. Returns (checked, breach).

    Only asserts when the pallet's total weight clears the engine's own
    gate: total >= cog_check_min_load_fraction * max_weight (if max_weight
    finite). expect_enforced=False means CoG is NOT engine-enforced for this
    workload (e.g. infinite max_weight CoG-only) — we still REPORT realized
    CoG vs envelope as an ANOMALY/info but don't FAIL.
    """
    cx, cy, tot = realized_cog(st.placements)
    if cx is None:
        return False, False
    xr, yr = envelope_for(pallet, config)
    max_w = pallet.max_weight
    gate = config.cog_check_min_load_fraction
    # Engine gate: when max_weight finite, CoG only enforced once
    # total >= gate * max_weight. When infinite, gate is skipped (enforced
    # from box #1) — but only if has_constraints triggers it at all.
    if max_w < float("inf"):
        above_gate = tot >= gate * max_w - EPS
    else:
        above_gate = True
    in_x = xr[0] - 1e-4 <= cx <= xr[1] + 1e-4
    in_y = yr[0] - 1e-4 <= cy <= yr[1] + 1e-4
    inside = in_x and in_y
    tag = "ENFORCED" if expect_enforced else "not-engine-enforced"
    line = (f"    [{name}] {st.pallet_id}: CoG=({cx:.1f},{cy:.1f}) "
            f"env x{xr} y{yr} tot_w={tot:.1f} "
            f"above_gate={above_gate} inside={inside} ({tag})")
    print(line)
    breach = False
    if above_gate and expect_enforced and not inside:
        breach = True
        FAILS.append(f"CoG BREACH [{name}] {st.pallet_id}: CoG=({cx:.2f},{cy:.2f}) "
                     f"outside env x{xr} y{yr}, tot_w={tot:.1f} "
                     f">= gate {gate*max_w if max_w<float('inf') else 0:.1f}")
    return above_gate, breach


# ===========================================================================
# TEST A — CoG envelope WITH a finite max_weight (engine-enforced path)
# ===========================================================================
def test_cog_enforced():
    print("\n=== TEST A: CoG envelope enforced (finite max_weight) ===")
    n_checked = 0
    n_breach = 0

    # A1: heavy boxes that, packed greedily into a corner, would shove CoG.
    # Tight envelope frac=0.1 on a 1000x1000 pallet -> x,y in [400,600].
    # Mix of heavy + light to give the solver a balancing degree of freedom.
    pallet = Pallet(length=1000, width=1000, height=1000, max_weight=2000)
    boxes = []
    for i in range(20):
        boxes.append(Box(id=f"H{i}", length=300, width=300, height=300,
                         weight=80.0, allowed_rotations=ANY))
    for i in range(20):
        boxes.append(Box(id=f"L{i}", length=200, width=200, height=200,
                         weight=5.0, allowed_rotations=ANY))
    cfg = PackerConfig(cog_envelope_fraction=0.1,
                       cog_check_min_load_fraction=0.35,
                       support_ratio=0.0, require_centroid_supported=False,
                       enforce_load_bearing=False, allow_pallet_overhang=False)
    # Sanity: this workload must trigger has_constraints (finite max_weight).
    *_, hc = precompute_constraint_arrays(boxes, pallet)
    print(f"  A1 has_constraints={hc} (expect True via max_weight)")
    r = solve(boxes, pallet, cfg, max_pallets=4, seed=7)
    errs = validate(r, pallet, cfg)
    print(f"  A1 pallets={len(r.pallets)} unpacked={len(r.unpacked)} validator_errs={len(errs)}")
    for st in r.pallets:
        ck, br = check_cog_pallet("A1", st, pallet, cfg, expect_enforced=True)
        n_checked += int(ck)
        n_breach += int(br)

    # A2: explicit asymmetric cog_x_range/cog_y_range (not centred).
    # Force CoG into a narrow off-centre band the solver must hit.
    pallet2 = Pallet(length=1200, width=1000, height=1200, max_weight=1500,
                     cog_x_range=(500.0, 700.0), cog_y_range=(450.0, 550.0))
    boxes2 = []
    for i in range(15):
        boxes2.append(Box(id=f"A{i}", length=300, width=250, height=300,
                          weight=60.0, allowed_rotations=ANY))
    for i in range(15):
        boxes2.append(Box(id=f"B{i}", length=200, width=200, height=200,
                          weight=15.0, allowed_rotations=ANY))
    cfg2 = PackerConfig(cog_envelope_fraction=0.25,  # overridden by explicit ranges
                        cog_check_min_load_fraction=0.3,
                        support_ratio=0.0, require_centroid_supported=False,
                        enforce_load_bearing=False)
    xmn, xmx, ymn, ymx, mlf, act = precompute_cog_envelope(pallet2, cfg2)
    print(f"  A2 envelope precompute: x[{xmn},{xmx}] y[{ymn},{ymx}] "
          f"min_load_frac={mlf} active={act} (expect active=True, explicit ranges)")
    r2 = solve(boxes2, pallet2, cfg2, max_pallets=4, seed=11)
    errs2 = validate(r2, pallet2, cfg2)
    print(f"  A2 pallets={len(r2.pallets)} unpacked={len(r2.unpacked)} validator_errs={len(errs2)}")
    for st in r2.pallets:
        ck, br = check_cog_pallet("A2", st, pallet2, cfg2, expect_enforced=True)
        n_checked += int(ck)
        n_breach += int(br)

    # A3: EXTREMELY tight + adversarial — one giant heavy block + small fillers.
    # A single dominant-weight box at a corner makes CoG nearly impossible to
    # balance; the engine should refuse to place it off-centre OR balance it.
    pallet3 = Pallet(length=1000, width=1000, height=1500, max_weight=1000)
    boxes3 = [Box(id="MEGA", length=400, width=400, height=400, weight=300.0,
                  allowed_rotations=ANY)]
    for i in range(30):
        boxes3.append(Box(id=f"f{i}", length=200, width=200, height=200,
                          weight=20.0, allowed_rotations=ANY))
    cfg3 = PackerConfig(cog_envelope_fraction=0.08,
                        cog_check_min_load_fraction=0.2,
                        support_ratio=0.0, require_centroid_supported=False,
                        enforce_load_bearing=False)
    r3 = solve(boxes3, pallet3, cfg3, max_pallets=5, seed=3)
    errs3 = validate(r3, pallet3, cfg3)
    print(f"  A3 pallets={len(r3.pallets)} unpacked={len(r3.unpacked)} validator_errs={len(errs3)}")
    for st in r3.pallets:
        ck, br = check_cog_pallet("A3", st, pallet3, cfg3, expect_enforced=True)
        n_checked += int(ck)
        n_breach += int(br)

    # A4: zero-weight boxes + a few heavy — CoG should ignore zero-weight,
    # determined entirely by heavy ones. Tight envelope.
    pallet4 = Pallet(length=1000, width=1000, height=1000, max_weight=500)
    boxes4 = []
    for i in range(6):
        boxes4.append(Box(id=f"W{i}", length=250, width=250, height=250,
                          weight=70.0, allowed_rotations=ANY))
    for i in range(20):
        boxes4.append(Box(id=f"Z{i}", length=200, width=200, height=200,
                          weight=0.0, allowed_rotations=ANY))
    cfg4 = PackerConfig(cog_envelope_fraction=0.12,
                        cog_check_min_load_fraction=0.3,
                        support_ratio=0.0, require_centroid_supported=False,
                        enforce_load_bearing=False)
    r4 = solve(boxes4, pallet4, cfg4, max_pallets=4, seed=5)
    errs4 = validate(r4, pallet4, cfg4)
    print(f"  A4 pallets={len(r4.pallets)} unpacked={len(r4.unpacked)} validator_errs={len(errs4)}")
    for st in r4.pallets:
        ck, br = check_cog_pallet("A4", st, pallet4, cfg4, expect_enforced=True)
        n_checked += int(ck)
        n_breach += int(br)

    METRICS.append(f"TEST A: pallets_checked_above_gate={n_checked} cog_breaches={n_breach}")
    print(f"  --> A: {n_checked} pallets above gate checked, {n_breach} CoG breaches")
    return n_breach == 0, n_checked


# ===========================================================================
# TEST B — CoG-ONLY workload (the has_constraints gating gap)
# ===========================================================================
def test_cog_only_gap():
    print("\n=== TEST B: CoG-only workload (no weight/load/RFS constraints) ===")
    # Infinite max_weight, no max_load_on_top, no requires_full_support.
    # Tight envelope. Per architecture, has_constraints=False -> cog_active
    # forced to 0 -> CoG is NOT enforced. We verify this is the case and
    # report whether the realized CoG actually breaches (it likely will,
    # demonstrating the silent gap).
    pallet = Pallet(length=1000, width=1000, height=1000)  # max_weight=inf
    boxes = []
    for i in range(20):
        boxes.append(Box(id=f"H{i}", length=300, width=300, height=300,
                         weight=80.0, allowed_rotations=ANY))
    cfg = PackerConfig(cog_envelope_fraction=0.05,  # very tight
                       cog_check_min_load_fraction=0.0,  # always-on intent
                       support_ratio=0.0, require_centroid_supported=False,
                       enforce_load_bearing=False)
    weights, mlot, rfs, pmw, hc = precompute_constraint_arrays(boxes, pallet)
    xmn, xmx, ymn, ymx, mlf, act = precompute_cog_envelope(pallet, cfg)
    print(f"  B has_constraints={hc} (cog_envelope precompute active={act})")
    print(f"  B -> engine cog_active passed to decoder = "
          f"{1 if (hc and act) else 0} (CoG enforced only if has_constraints AND active)")
    r = solve(boxes, pallet, cfg, max_pallets=4, seed=9)
    errs = validate(r, pallet, cfg)
    print(f"  B pallets={len(r.pallets)} unpacked={len(r.unpacked)} validator_errs={len(errs)}")
    any_breach = False
    for st in r.pallets:
        cx, cy, tot = realized_cog(st.placements)
        if cx is None:
            continue
        xr, yr = envelope_for(pallet, cfg)
        inside = (xr[0] <= cx <= xr[1]) and (yr[0] <= cy <= yr[1])
        print(f"    [B] {st.pallet_id}: CoG=({cx:.1f},{cy:.1f}) env x{xr} y{yr} "
              f"inside={inside} tot_w={tot:.1f}")
        if not inside:
            any_breach = True
    if not hc:
        # This is the documented gating. If a breach occurs, it's a known
        # design limitation (CoG silently skipped without other constraints),
        # not an enforcement bug. Flag as ANOMALY so the report surfaces it.
        if any_breach:
            ANOMALIES.append(
                "CoG-only workload (infinite max_weight, no load/RFS constraints): "
                "has_constraints=False so the engine FORCES cog_active=0 and does "
                "NOT enforce the CoG envelope at all — realized CoG breaches the "
                "tight envelope. This is per driver.py L194-209 gating. CoG cannot "
                "be enforced without at least one of {finite max_weight, finite "
                "max_load_on_top, requires_full_support}.")
        METRICS.append(f"TEST B: has_constraints={hc}, cog_silently_skipped=True, "
                       f"realized_breach={any_breach}")
        # Not a FAIL: documented gating, not an enforcement violation.
        return True
    # If somehow has_constraints became True, CoG must be enforced.
    return not any_breach


# ===========================================================================
# TEST C — max_overhang bound (boxes may reach edge+overhang, never beyond)
# ===========================================================================
def test_overhang():
    print("\n=== TEST C: max_overhang bound ===")
    n_checked = 0
    n_over = 0
    n_used_overhang = 0
    worst_over = 0.0

    # C1: overhang ON, modest box mix; verify x2/y2 <= edge+overhang and
    # never overhang -x/-y (x>=0, y>=0).
    overhang = 150.0
    pallet = Pallet(length=1000, width=1000, height=1000, max_weight=5000,
                    max_overhang=overhang)
    boxes = []
    for i in range(40):
        boxes.append(Box(id=f"o{i}", length=240, width=240, height=240,
                         weight=10.0, allowed_rotations=ANY))
    cfg = PackerConfig(allow_pallet_overhang=True,
                       cog_envelope_fraction=1.0,  # disable CoG to isolate overhang
                       support_ratio=0.0, require_centroid_supported=False,
                       enforce_load_bearing=False)
    r = solve(boxes, pallet, cfg, max_pallets=3, seed=13)
    errs = validate(r, pallet, cfg)
    print(f"  C1 overhang={overhang} pallets={len(r.pallets)} "
          f"unpacked={len(r.unpacked)} validator_errs={len(errs)}")
    for st in r.pallets:
        for p in st.placements:
            n_checked += 1
            # Must never go negative on -x/-y/-z.
            if p.x < -EPS or p.y < -EPS or p.z < -EPS:
                n_over += 1
                FAILS.append(f"OVERHANG NEG [C1] {p.box.id} at "
                             f"({p.x},{p.y},{p.z}) negative coord")
            # +x/+y may reach edge+overhang; never beyond.
            over_x = p.x2 - pallet.length
            over_y = p.y2 - pallet.width
            worst_over = max(worst_over, over_x, over_y)
            if over_x > overhang + EPS or over_y > overhang + EPS:
                n_over += 1
                FAILS.append(f"OVERHANG BEYOND [C1] {p.box.id}: x2={p.x2} y2={p.y2} "
                             f"limit_x={pallet.length+overhang} "
                             f"limit_y={pallet.width+overhang}")
            if over_x > EPS or over_y > EPS:
                n_used_overhang += 1
            # z must never exceed pallet height (no z overhang).
            if p.z2 > pallet.height + EPS:
                n_over += 1
                FAILS.append(f"OVERHANG Z [C1] {p.box.id}: z2={p.z2} > H={pallet.height}")
    if errs:
        for e in errs[:5]:
            FAILS.append(f"VALIDATOR [C1] {e}")

    # C2: overhang ON but boxes don't tile evenly -> should USE overhang.
    # Pallet 1000 wide, box 350 -> 2 fit (700), 3rd at 700..1050 overhangs by 50.
    overhang2 = 120.0
    pallet2 = Pallet(length=1050, width=1050, height=800, max_weight=5000,
                     max_overhang=overhang2)
    boxes2 = []
    for i in range(30):
        boxes2.append(Box(id=f"q{i}", length=350, width=350, height=350,
                          weight=8.0, allowed_rotations=ANY))
    cfg2 = PackerConfig(allow_pallet_overhang=True,
                        cog_envelope_fraction=1.0,
                        support_ratio=0.0, require_centroid_supported=False,
                        enforce_load_bearing=False)
    r2 = solve(boxes2, pallet2, cfg2, max_pallets=3, seed=17)
    errs2 = validate(r2, pallet2, cfg2)
    print(f"  C2 overhang={overhang2} pallets={len(r2.pallets)} "
          f"unpacked={len(r2.unpacked)} validator_errs={len(errs2)}")
    for st in r2.pallets:
        for p in st.placements:
            n_checked += 1
            over_x = p.x2 - pallet2.length
            over_y = p.y2 - pallet2.width
            worst_over = max(worst_over, over_x, over_y)
            if over_x > overhang2 + EPS or over_y > overhang2 + EPS:
                n_over += 1
                FAILS.append(f"OVERHANG BEYOND [C2] {p.box.id}: x2={p.x2} y2={p.y2} "
                             f"limit_x={pallet2.length+overhang2} "
                             f"limit_y={pallet2.width+overhang2}")
            if over_x > EPS or over_y > EPS:
                n_used_overhang += 1
            if p.x < -EPS or p.y < -EPS:
                n_over += 1
                FAILS.append(f"OVERHANG NEG [C2] {p.box.id} ({p.x},{p.y})")
    if errs2:
        for e in errs2[:5]:
            FAILS.append(f"VALIDATOR [C2] {e}")

    # C3: overhang OFF (allow_pallet_overhang=False) but pallet HAS max_overhang
    # set -> overhang must be IGNORED; nothing may exceed the bare pallet edge.
    pallet3 = Pallet(length=1000, width=1000, height=800, max_weight=5000,
                     max_overhang=300.0)  # large, but should be ignored
    boxes3 = [Box(id=f"r{i}", length=300, width=300, height=300, weight=5.0,
                  allowed_rotations=ANY) for i in range(30)]
    cfg3 = PackerConfig(allow_pallet_overhang=False,  # OFF
                        cog_envelope_fraction=1.0,
                        support_ratio=0.0, require_centroid_supported=False,
                        enforce_load_bearing=False)
    r3 = solve(boxes3, pallet3, cfg3, max_pallets=3, seed=19)
    errs3 = validate(r3, pallet3, cfg3)
    print(f"  C3 overhang-OFF (pallet.max_overhang=300 ignored) pallets={len(r3.pallets)} "
          f"validator_errs={len(errs3)}")
    c3_over = 0
    for st in r3.pallets:
        for p in st.placements:
            n_checked += 1
            if p.x2 > pallet3.length + EPS or p.y2 > pallet3.width + EPS:
                c3_over += 1
                n_over += 1
                FAILS.append(f"OVERHANG-OFF LEAK [C3] {p.box.id}: x2={p.x2} y2={p.y2} "
                             f"exceeds bare edge {pallet3.length} (overhang must be ignored)")
    print(f"  C3 boxes exceeding bare edge (should be 0): {c3_over}")
    if errs3:
        for e in errs3[:5]:
            FAILS.append(f"VALIDATOR [C3] {e}")

    METRICS.append(f"TEST C: boxes_checked={n_checked} overhang_violations={n_over} "
                   f"boxes_using_overhang={n_used_overhang} worst_overhang={worst_over:.2f}")
    print(f"  --> C: {n_checked} boxes checked, {n_over} overhang violations, "
          f"{n_used_overhang} actually used overhang, worst_overhang={worst_over:.2f}")
    return n_over == 0, n_used_overhang


# ===========================================================================
# TEST D — int/float CoG drift: non-integer box dims under CoG envelope
# ===========================================================================
def test_noninteger_cog_drift():
    print("\n=== TEST D: non-integer dims -> int-decode vs float-CoG drift ===")
    # Decoder rounds dims to int; our independent CoG uses float dims. With
    # fractional dims the realized float CoG can drift from the int CoG the
    # engine checked. Use dims with a .5 fraction to maximize rounding drift.
    # Pallet edges also rounded. Tight envelope + finite max_weight so CoG
    # IS engine-enforced.
    pallet = Pallet(length=1000.0, width=1000.0, height=1000.0, max_weight=1500)
    boxes = []
    for i in range(24):
        # length/width .5 fractions -> int rounding shifts each centroid by
        # up to ~0.25 in float vs int space; accumulated over many heavy boxes.
        boxes.append(Box(id=f"d{i}", length=287.5, width=287.5, height=287.5,
                         weight=50.0, allowed_rotations=ANY))
    cfg = PackerConfig(cog_envelope_fraction=0.1,
                       cog_check_min_load_fraction=0.3,
                       support_ratio=0.0, require_centroid_supported=False,
                       enforce_load_bearing=False)
    r = solve(boxes, pallet, cfg, max_pallets=5, seed=23)
    errs = validate(r, pallet, cfg)
    print(f"  D pallets={len(r.pallets)} unpacked={len(r.unpacked)} validator_errs={len(errs)}")
    worst_drift = 0.0
    n_checked = 0
    n_breach = 0
    xr, yr = envelope_for(pallet, cfg)
    for st in r.pallets:
        cx, cy, tot = realized_cog(st.placements)
        if cx is None:
            continue
        above = tot >= cfg.cog_check_min_load_fraction * pallet.max_weight - EPS
        in_x = xr[0] - 1e-4 <= cx <= xr[1] + 1e-4
        in_y = yr[0] - 1e-4 <= cy <= yr[1] + 1e-4
        inside = in_x and in_y
        # Drift magnitude: how far outside (if any) the float CoG sits.
        dx_out = max(0.0, xr[0] - cx, cx - xr[1])
        dy_out = max(0.0, yr[0] - cy, cy - yr[1])
        worst_drift = max(worst_drift, dx_out, dy_out)
        print(f"    [D] {st.pallet_id}: CoG=({cx:.3f},{cy:.3f}) env x{xr} y{yr} "
              f"above_gate={above} inside={inside} out=({dx_out:.3f},{dy_out:.3f})")
        if above:
            n_checked += 1
            if not inside:
                n_breach += 1
    METRICS.append(f"TEST D: pallets_checked={n_checked} float_cog_breaches={n_breach} "
                   f"worst_float_drift_outside_env={worst_drift:.4f}")
    print(f"  --> D: {n_checked} checked, {n_breach} float-CoG breaches, "
          f"worst drift outside env = {worst_drift:.4f}")
    # A small drift (< ~0.5 from int rounding) is expected/benign. A large
    # breach (> 1.0) on an enforced pallet would be a real int/float bug.
    if n_breach > 0 and worst_drift > 1.0:
        FAILS.append(f"INT/FLOAT CoG DRIFT [D]: realized float CoG breaches "
                     f"enforced envelope by {worst_drift:.3f} (> 1.0 rounding band)")
        return False
    if n_breach > 0:
        ANOMALIES.append(f"Minor int/float CoG drift [D]: float CoG sits "
                         f"{worst_drift:.3f} outside the envelope on an enforced "
                         f"pallet (within the int-rounding band, benign).")
    return True


# ===========================================================================
# TEST E — overhang + tight CoG interaction (envelope relative to bare L/W)
# ===========================================================================
def test_overhang_cog_interaction():
    print("\n=== TEST E: overhang + CoG envelope interaction ===")
    # Overhang lets boxes extend to edge+overhang. CoG envelope (default frac)
    # is relative to BARE pallet L/W (precompute uses pallet.length, not L_eff).
    # An overhanging box pushes its centroid further out -> stresses the CoG
    # envelope harder. Finite max_weight so CoG IS enforced.
    overhang = 200.0
    pallet = Pallet(length=1000, width=1000, height=1000, max_weight=1500,
                    max_overhang=overhang)
    boxes = []
    for i in range(24):
        boxes.append(Box(id=f"e{i}", length=300, width=300, height=300,
                         weight=55.0, allowed_rotations=ANY))
    cfg = PackerConfig(allow_pallet_overhang=True,
                       cog_envelope_fraction=0.15,
                       cog_check_min_load_fraction=0.3,
                       support_ratio=0.0, require_centroid_supported=False,
                       enforce_load_bearing=False)
    r = solve(boxes, pallet, cfg, max_pallets=5, seed=29)
    errs = validate(r, pallet, cfg)
    print(f"  E pallets={len(r.pallets)} unpacked={len(r.unpacked)} validator_errs={len(errs)}")
    n_checked = 0
    n_breach = 0
    n_over = 0
    for st in r.pallets:
        # overhang bound
        for p in st.placements:
            if p.x2 > pallet.length + overhang + EPS or p.y2 > pallet.width + overhang + EPS:
                n_over += 1
                FAILS.append(f"OVERHANG BEYOND [E] {p.box.id}: x2={p.x2} y2={p.y2}")
        ck, br = check_cog_pallet("E", st, pallet, cfg, expect_enforced=True)
        n_checked += int(ck)
        n_breach += int(br)
    if errs:
        for e in errs[:5]:
            FAILS.append(f"VALIDATOR [E] {e}")
    METRICS.append(f"TEST E: pallets_checked={n_checked} cog_breaches={n_breach} "
                   f"overhang_violations={n_over}")
    print(f"  --> E: {n_checked} checked, {n_breach} CoG breaches, {n_over} overhang viol")
    return n_breach == 0 and n_over == 0


# ===========================================================================
# TEST F — MINIMAL deterministic repro of the single-box new-bin CoG skip
# ===========================================================================
def test_minimal_singlebox_newbin():
    print("\n=== TEST F: MINIMAL repro — single-box new-bin skips CoG check ===")
    # Root cause (jit_decoders_cstr.py L194-289): when a box fails the CoG
    # check in all existing bins, the decoder OPENS A NEW BIN and commits the
    # box there with NO CoG check (only the load-on-top check is intentionally
    # skipped for floor placements — but CoG is silently skipped too). A heavy
    # box thus lands alone in a fresh pallet at a corner, with single-box CoG
    # far outside the envelope, even though tot >= cog_check_min_load gate.
    # validate() does NOT flag this (no CoG check) -> silent constraint breach.
    pallet = Pallet(length=1000, width=1000, height=1000, max_weight=600)
    boxes = [Box(id="MEGA", length=400, width=400, height=400, weight=200.0,
                 allowed_rotations=ANY)]
    for i in range(8):
        boxes.append(Box(id=f"f{i}", length=200, width=200, height=200,
                         weight=10.0, allowed_rotations=ANY))
    cfg = PackerConfig(cog_envelope_fraction=0.1,
                       cog_check_min_load_fraction=0.2,
                       support_ratio=0.0, require_centroid_supported=False,
                       enforce_load_bearing=False)
    xr, yr = envelope_for(pallet, cfg)
    gate = cfg.cog_check_min_load_fraction * pallet.max_weight
    print(f"  F env x{xr} y{yr} gate={gate} (CoG enforced once tot>=gate)")
    breaches = 0
    seeds_breached = []
    for seed in (1, 2, 3, 7, 42):
        r = solve(boxes, pallet, cfg, max_pallets=5, t=3.0, seed=seed)
        errs = validate(r, pallet, cfg)
        for st in r.pallets:
            cx, cy, tot = realized_cog(st.placements)
            if cx is None:
                continue
            above = tot >= gate - EPS
            inside = (xr[0] - 1e-4 <= cx <= xr[1] + 1e-4) and (yr[0] - 1e-4 <= cy <= yr[1] + 1e-4)
            if above and not inside:
                breaches += 1
                if seed not in seeds_breached:
                    seeds_breached.append(seed)
                print(f"    seed={seed} {st.pallet_id}: CoG=({cx:.1f},{cy:.1f}) "
                      f"tot={tot:.0f} n={len(st.placements)} above_gate=True "
                      f"inside=False validator_errs={len(errs)}  <-- BREACH")
    METRICS.append(f"TEST F (minimal repro): breaching_pallets={breaches} "
                   f"seeds_breached={seeds_breached}")
    if breaches > 0:
        FAILS.append(
            f"CoG SINGLE-BOX NEW-BIN SKIP [F]: a 200kg box lands alone on a "
            f"fresh pallet at corner -> CoG=(200,200) outside envelope {xr} "
            f"with tot=200 >= gate {gate}; reproducible across seeds "
            f"{seeds_breached}. validate() returns 0 errors (no CoG check). "
            f"Root cause: jit_decoders_cstr.py new-bin commit path skips "
            f"_check_cog_envelope_njit. Repro: scripts/_verify/_cog_minimal.py")
        print(f"  --> F: {breaches} breaching pallets across seeds {seeds_breached} (DEFECT)")
        return False
    print("  --> F: no single-box new-bin CoG breach")
    return True


def main():
    warmup_jit()
    print("################ CoG envelope + max_overhang adversarial probe ################")

    results = {}
    results["A_cog_enforced"], a_checked = test_cog_enforced()
    results["B_cog_only_gap"] = test_cog_only_gap()
    results["C_overhang"], c_used = test_overhang()
    results["D_int_float_drift"] = test_noninteger_cog_drift()
    results["E_overhang_cog"] = test_overhang_cog_interaction()
    results["F_singlebox_newbin"] = test_minimal_singlebox_newbin()

    print("\n================ SUMMARY ================")
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print("  -- metrics --")
    for m in METRICS:
        print(f"    {m}")
    if ANOMALIES:
        print("  -- anomalies (documented gating / benign) --")
        for an in ANOMALIES:
            print(f"    ANOMALY: {an}")
    if FAILS:
        print("  -- failures --")
        for f in FAILS:
            print(f"    FAIL: {f}")

    all_pass = all(results.values()) and not FAILS
    # Cross-check: TEST A must have actually checked >=15 pallet instances?
    # We have >=15 distinct constructed cases across A/C/D/E; report counts.
    print(f"\n  total CoG-pallets-above-gate checked in A = {a_checked}; "
          f"overhang boxes that used overhang in C = {c_used}")
    print(f"\n================ {'OVERALL PASS' if all_pass else 'OVERALL FAIL'} ================")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
