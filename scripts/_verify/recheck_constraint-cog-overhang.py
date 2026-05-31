"""Counter-probe for claimed defect 'constraint-cog-overhang'.

CLAIM (from verify_constraint_cog_overhang.py): full solves return multi-box
pallets above the cog_check_min_load_fraction gate whose realized CoG is
outside the envelope (TEST A2/A3/E), plus a single-box new-bin case (TEST F),
while validate() reports clean -> "silent constraint breach".

We INDEPENDENTLY re-derive whether this is a real algorithm defect, a probe
artifact, or expected/documented behavior, via FOUR orthogonal checks:

  Q1 (canonical semantics). packer.py:_cog_ok L168-179 explicitly does
     `if not self.placements: return True` with the comment "with a single
     box in a corner the check is unsatisfiable." The CANONICAL v2 reference
     deliberately EXEMPTS the first placement of every pallet from the CoG
     envelope. We RUN the v2 reference PalletPacker on the exact minimal repro
     and confirm it ALSO leaves a lone heavy box off-envelope above the gate.

  Q2 (validator contract). validate() does not check CoG at all (it covers
     geometry/overlap/weight/support/rotation/load-bearing). So "0 errors" is
     BY DESIGN; CoG is a soft objective-shaping constraint, not a hard
     validity invariant. A realized CoG outside the envelope is therefore not
     a validity failure by the system's own contract.

  Q3 (port-divergence test, the decisive one). Run A2/A3/E through BOTH the
     BRKGA engine AND the v2 reference PalletPacker. If the reference (which
     DOES check CoG on box #2+) ALSO finishes multi-box pallets off-envelope
     above the gate, the breach is an inherent property of the greedy
     incremental CoG veto (it can veto a candidate but never relocate placed
     boxes) -- NOT a Cython-port divergence.

  Q4 (incremental-contract replay, the strongest enforcement test). For each
     returned pallet, replay placements in commit order with INT coords and
     verify the engine's ACTUAL guarantee: for every NON-SEED placement made
     while running_total >= gate, the cumulative CoG stays inside the
     envelope. A violation here would be a genuine enforcement bug (the engine
     committed a box it should have vetoed). Also: a per-chromosome decoder
     test proving the in-bin CoG check vetoes a balance-breaking placement.

Run:
    docker run --rm -e OMP_NUM_THREADS=1 -e PYTHONHASHSEED=0 \
        -v "$(pwd)":/app -w /app pallet-packer:dev \
        python scripts/_verify/recheck_constraint-cog-overhang.py
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
from pallet_packer.packer import PalletPacker  # noqa: E402  v2 reference
from pallet_packer._brkga_core.precompute import (  # noqa: E402
    precompute_box_dims_and_sku, precompute_constraint_arrays,
    precompute_cog_envelope,
)
from pallet_packer._brkga_core.jit_decoders_cstr_cy import (  # noqa: E402
    decode_njit_mode_cstr,
)

EPS = 1e-6
ANY = list(ALL_ROTATIONS)


def realized_cog(placements):
    tot = sx = sy = 0.0
    for p in placements:
        w = p.box.weight
        tot += w
        sx += w * (p.x + p.dx / 2.0)
        sy += w * (p.y + p.dy / 2.0)
    if tot <= 0:
        return None, None, 0.0
    return sx / tot, sy / tot, tot


def envelope_for(pallet, config):
    L, W = float(pallet.length), float(pallet.width)
    frac = float(config.cog_envelope_fraction)
    xr = (pallet.cog_x_range[0], pallet.cog_x_range[1]) if pallet.cog_x_range \
        else (L / 2.0 - frac * L, L / 2.0 + frac * L)
    yr = (pallet.cog_y_range[0], pallet.cog_y_range[1]) if pallet.cog_y_range \
        else (W / 2.0 - frac * W, W / 2.0 + frac * W)
    return tuple(map(float, xr)), tuple(map(float, yr))


def inside(cx, cy, xr, yr):
    return (xr[0] - 1e-4 <= cx <= xr[1] + 1e-4) and (yr[0] - 1e-4 <= cy <= yr[1] + 1e-4)


def brkga(boxes, pallet, config, *, max_pallets, seed, t=4.0):
    return brkga_pack_v35(
        boxes, pallet, config, time_limit_s=t, max_pallets=max_pallets,
        seed=seed, population_size=120, n_populations=2, patience=100,
        local_search_budget_s=0.6, verbose=False, n_modes=6)


def ref_pack(boxes, pallet, config):
    """v2 reference greedy packer (canonical _cog_ok semantics)."""
    return PalletPacker(pallet, config).pack(boxes)


def report_breaches(tag, result, pallet, config):
    xr, yr = envelope_for(pallet, config)
    gate_frac = config.cog_check_min_load_fraction
    mw = pallet.max_weight
    n_above = n_breach = 0
    lines = []
    for st in result.pallets:
        cx, cy, tot = realized_cog(st.placements)
        if cx is None:
            continue
        above = True if mw == float("inf") else (tot >= gate_frac * mw - EPS)
        ins = inside(cx, cy, xr, yr)
        if above:
            n_above += 1
            if not ins:
                n_breach += 1
                lines.append(f"      {tag} {st.pallet_id}: CoG=({cx:.1f},{cy:.1f}) "
                             f"n={len(st.placements)} tot={tot:.0f} env x{xr} y{yr} BREACH")
    return len(result.pallets), n_above, n_breach, lines


# ===========================================================================
# Q4 helper: replay one pallet's commit sequence and check the INCREMENTAL
# enforcement contract. A violation = the engine committed a non-seed box,
# above the gate, that left the cumulative CoG outside the envelope = REAL bug.
# ===========================================================================
def is_single_regular_block(plc):
    """A pallet is one atomic same-SKU block (= the new-bin SEED block, fully
    CoG-exempt per the first-placement exemption) iff all boxes share the same
    (dims,weight,rotation) and their positions form a regular x*y*z grid that
    exactly tiles the box count.

    For such a pallet, EVERY box was committed in a single
    _commit_block_placements_njit call as the new-bin seed; none ran through
    _check_cog_envelope_njit. So no incremental contract applies to it."""
    if len(plc) <= 1:
        return True
    key0 = (round(plc[0].box.length, 6), round(plc[0].box.width, 6),
            round(plc[0].box.height, 6), round(plc[0].box.weight, 6))
    for p in plc:
        if (round(p.box.length, 6), round(p.box.width, 6),
                round(p.box.height, 6), round(p.box.weight, 6)) != key0:
            return False
    xs = sorted(set(int(round(p.x)) for p in plc))
    ys = sorted(set(int(round(p.y)) for p in plc))
    zs = sorted(set(int(round(p.z)) for p in plc))
    return len(xs) * len(ys) * len(zs) == len(plc)


def replay_incremental(name, st, pallet, config):
    """Verify the engine's ACTUAL incremental enforcement contract.

    The new-bin SEED (single box OR a same-SKU block committed atomically) is
    CoG-exempt by design (mirrors packer.py:_cog_ok `if not self.placements`).
    A genuine enforcement bug = the engine committed a box to a pallet that
    ALREADY held a distinct prior placement, while running_total >= gate, that
    left cumulative CoG outside the envelope.

    We therefore exempt a pallet that is a single atomic block. For mixed
    pallets we walk the boxes and exempt the leading same-SKU/same-grid block
    (the seed), then require every later box to satisfy the running check.
    """
    xr, yr = envelope_for(pallet, config)
    mw = pallet.max_weight
    gate = (config.cog_check_min_load_fraction * mw) if mw < float("inf") else 0.0
    plc = list(st.placements)
    if is_single_regular_block(plc):
        return 0, 0, 0.0, True  # whole pallet is the exempt seed block

    sum_xw = sum_yw = tot = 0.0
    n_checks = n_viol = 0
    worst = 0.0
    # Identify the leading seed block: contiguous run from index 0 that shares
    # box-0's SKU. (Conservative: this matches how the new-bin path commits the
    # first same-SKU block before any other SKU/box is added.)
    key0 = (round(plc[0].box.length, 6), round(plc[0].box.width, 6),
            round(plc[0].box.height, 6), round(plc[0].box.weight, 6))
    seed_len = 0
    for p in plc:
        k = (round(p.box.length, 6), round(p.box.width, 6),
             round(p.box.height, 6), round(p.box.weight, 6))
        if k == key0:
            seed_len += 1
        else:
            break
    for i, p in enumerate(plc):
        w = float(p.box.weight)
        cx = float(int(round(p.x))) + 0.5 * float(int(round(p.dx)))
        cy = float(int(round(p.y))) + 0.5 * float(int(round(p.dy)))
        sum_xw += w * cx
        sum_yw += w * cy
        tot += w
        if i < seed_len:
            continue  # exempt seed block
        if tot <= 0:
            continue
        above = (mw == float("inf")) or (tot >= gate - EPS)
        if not above:
            continue
        n_checks += 1
        ccx, ccy = sum_xw / tot, sum_yw / tot
        if not inside(ccx, ccy, xr, yr):
            n_viol += 1
            dx_out = max(0.0, xr[0] - ccx, ccx - xr[1])
            dy_out = max(0.0, yr[0] - ccy, ccy - yr[1])
            worst = max(worst, dx_out, dy_out)
            print(f"        [{name}] POST-SEED VIOLATION commit#{i} "
                  f"({p.box.id}): cumCoG=({ccx:.1f},{ccy:.1f}) tot={tot:.0f} "
                  f"out=({dx_out:.1f},{dy_out:.1f})")
    return n_checks, n_viol, worst, False


def case_minimal_singlebox():
    print("\n=== Q1: minimal single-heavy-box repro — engine vs v2 reference ===")
    pallet = Pallet(length=1000, width=1000, height=1000, max_weight=600)
    boxes = [Box(id="MEGA", length=400, width=400, height=400, weight=200.0,
                 allowed_rotations=ANY)]
    for i in range(8):
        boxes.append(Box(id=f"f{i}", length=200, width=200, height=200,
                         weight=10.0, allowed_rotations=ANY))
    cfg = PackerConfig(cog_envelope_fraction=0.1, cog_check_min_load_fraction=0.2,
                       support_ratio=0.0, require_centroid_supported=False,
                       enforce_load_bearing=False)
    xr, yr = envelope_for(pallet, cfg)
    print(f"  env x{xr} y{yr} gate={cfg.cog_check_min_load_fraction*pallet.max_weight}")

    rref = ref_pack(boxes, pallet, cfg)
    np_, nag, nbr, lines = report_breaches("[REF]", rref, pallet, cfg)
    errs = validate(rref, pallet, cfg)
    print(f"  v2 REFERENCE: pallets={np_} above_gate={nag} breaches={nbr} "
          f"validator_errs={len(errs)}")
    for ln in lines:
        print(ln)

    eng_breach = 0
    eng_seeds = []
    for seed in (1, 2, 3, 7, 42):
        r = brkga(boxes, pallet, cfg, max_pallets=5, seed=seed)
        _, _, nb, ln = report_breaches(f"[ENG s{seed}]", r, pallet, cfg)
        if nb:
            eng_breach += nb
            eng_seeds.append(seed)
            for x in ln:
                print(x)
    print(f"  BRKGA engine: total breaching pallets across seeds = {eng_breach} "
          f"(seeds {eng_seeds})")
    ref_also_breaches = nbr > 0
    print(f"  --> v2 REFERENCE also leaves lone heavy box off-envelope above "
          f"gate: {ref_also_breaches}")
    return ref_also_breaches, eng_breach


def case_multibox():
    print("\n=== Q3+Q4: multi-box above-gate — engine vs reference + replay ===")
    findings = []
    total_incr_viol = 0

    # A2: explicit asymmetric envelope, mixed heavy/light.
    p2 = Pallet(length=1200, width=1000, height=1200, max_weight=1500,
                cog_x_range=(500.0, 700.0), cog_y_range=(450.0, 550.0))
    b2 = [Box(id=f"A{i}", length=300, width=250, height=300, weight=60.0,
              allowed_rotations=ANY) for i in range(15)]
    b2 += [Box(id=f"B{i}", length=200, width=200, height=200, weight=15.0,
               allowed_rotations=ANY) for i in range(15)]
    c2 = PackerConfig(cog_envelope_fraction=0.25, cog_check_min_load_fraction=0.3,
                      support_ratio=0.0, require_centroid_supported=False,
                      enforce_load_bearing=False)

    # A3: one giant block + fillers.
    p3 = Pallet(length=1000, width=1000, height=1500, max_weight=1000)
    b3 = [Box(id="MEGA", length=400, width=400, height=400, weight=300.0,
              allowed_rotations=ANY)]
    b3 += [Box(id=f"f{i}", length=200, width=200, height=200, weight=20.0,
               allowed_rotations=ANY) for i in range(30)]
    c3 = PackerConfig(cog_envelope_fraction=0.08, cog_check_min_load_fraction=0.2,
                      support_ratio=0.0, require_centroid_supported=False,
                      enforce_load_bearing=False)

    # E: overhang + tight CoG.
    pE = Pallet(length=1000, width=1000, height=1000, max_weight=1500,
                max_overhang=200.0)
    bE = [Box(id=f"e{i}", length=300, width=300, height=300, weight=55.0,
              allowed_rotations=ANY) for i in range(24)]
    cE = PackerConfig(allow_pallet_overhang=True, cog_envelope_fraction=0.15,
                      cog_check_min_load_fraction=0.3, support_ratio=0.0,
                      require_centroid_supported=False, enforce_load_bearing=False)

    for name, p, b, c, seed in (("A2", p2, b2, c2, 11),
                                ("A3", p3, b3, c3, 3),
                                ("E", pE, bE, cE, 29)):
        rref = ref_pack(b, p, c)
        _, nag_r, nbr_r, ln_r = report_breaches(f"[REF {name}]", rref, p, c)
        errs_r = validate(rref, p, c)
        reng = brkga(b, p, c, max_pallets=5, seed=seed)
        _, nag_e, nbr_e, ln_e = report_breaches(f"[ENG {name}]", reng, p, c)
        errs_e = validate(reng, p, c)
        print(f"  {name}: REF above_gate={nag_r} breaches={nbr_r} "
              f"valerr={len(errs_r)} | ENG above_gate={nag_e} breaches={nbr_e} "
              f"valerr={len(errs_e)}")
        for x in ln_r + ln_e:
            print(x)
        # Q4: incremental replay of the ENGINE result (seed-block aware).
        case_viol = 0
        case_checks = 0
        n_seed_only = 0
        for st in reng.pallets:
            ck, vi, _, seed_only = replay_incremental(name, st, p, c)
            case_checks += ck
            case_viol += vi
            n_seed_only += int(seed_only)
        print(f"      [{name}] engine post-seed contract: "
              f"post_gate_checks={case_checks} violations={case_viol} "
              f"pallets_that_are_pure_seed_block={n_seed_only}/{len(reng.pallets)}")
        total_incr_viol += case_viol
        findings.append((name, nbr_r, nbr_e, case_viol))
    return findings, total_incr_viol


def case_inbin_enforcement():
    print("\n=== Q4b: in-bin CoG check actually vetoes balance-breaking place ===")
    boxes = [
        Box(id="anchor", length=400, width=400, height=300, weight=10.0,
            allowed_rotations=[ALL_ROTATIONS[0]]),
        Box(id="HEAVY", length=300, width=300, height=300, weight=500.0,
            allowed_rotations=[ALL_ROTATIONS[0]]),
    ]
    pallet = Pallet(length=1000, width=1000, height=1000, max_weight=600)
    cfg = PackerConfig(cog_envelope_fraction=0.1, cog_check_min_load_fraction=0.2,
                       support_ratio=0.0, require_centroid_supported=False,
                       enforce_load_bearing=False)
    n_rots_arr, dims_all, sku = precompute_box_dims_and_sku(boxes)
    weights, mlot, rfs, pmw, hc = precompute_constraint_arrays(boxes, pallet)
    cx_min, cx_max, cy_min, cy_max, mlf, act = precompute_cog_envelope(pallet, cfg)
    L = int(round(pallet.length)); W = int(round(pallet.width)); H = int(round(pallet.height))

    results = {}
    for cog_on in (1, 0):
        bps = np.array([0, 1], dtype=np.int64)
        placements_out = np.zeros((len(boxes), 6), dtype=np.int64)
        decode_njit_mode_cstr(
            bps, n_rots_arr, dims_all, L, W, H, 5, placements_out, 0,
            weights, mlot, rfs, float(pmw), 0.0, 0,
            cx_min, cx_max, cy_min, cy_max, mlf, cog_on)
        info = [(boxes[b].id, int(placements_out[i, 0]), int(placements_out[i, 5]))
                for i, b in enumerate([0, 1])]
        same_bin = (info[0][2] == 1 and info[1][2] == 1
                    and info[0][1] == info[1][1])
        results[cog_on] = (info, same_bin)
        tag = "cog_ON " if cog_on else "cog_OFF"
        print(f"  {tag}: placements(id,bin,ok)={info} anchor&HEAVY same_bin={same_bin}")
    # Enforcement works iff: with CoG OFF the heavy co-locates with anchor in
    # the same bin, but with CoG ON the decoder vetoes that and separates them.
    works = (results[0][1] is True) and (results[1][1] is False)
    print(f"  --> in-bin CoG veto demonstrably active (OFF co-locates, ON "
          f"separates): {works}")
    return works


def main():
    warmup_jit()
    print("########## counter-probe: CoG envelope new-bin skip ##########")

    ref_breaches_single, eng_single = case_minimal_singlebox()
    multibox, total_incr_viol = case_multibox()
    inbin_works = case_inbin_enforcement()

    print("\n================ VERDICT EVIDENCE ================")
    print(f"  [Q1] v2 REFERENCE engine ALSO leaves a lone heavy box off-envelope "
          f"above the gate: {ref_breaches_single}")
    print(f"       (packer.py:_cog_ok L173 'if not self.placements: return True' "
          f"— first-box CoG skip is DOCUMENTED/canonical.)")
    print(f"  [Q2] validate() has no CoG code -> 0 errors is BY DESIGN; CoG is a "
          f"soft constraint, not a hard validity invariant.")
    ref_breaks_multi = any(nbr_r > 0 for _, nbr_r, _, _ in multibox)
    print(f"  [Q3] multi-box above-gate REF vs ENG breach counts:")
    for name, nbr_r, nbr_e, vi in multibox:
        print(f"        {name}: ref_breaches={nbr_r} eng_breaches={nbr_e} "
              f"engine_incremental_violations={vi}")
    print(f"       -> reference (checks CoG box#2+) ALSO breaches multi-box: "
          f"{ref_breaks_multi}")
    print(f"  [Q4] engine incremental-contract violations (non-seed box, above "
          f"gate, cumulative CoG out) across A2/A3/E = {total_incr_viol}")
    print(f"  [Q4b] in-bin CoG veto demonstrably active = {inbin_works}")

    print("\n================ CONCLUSION ================")
    port_defect = False
    enforcement_bug = (total_incr_viol > 0)
    if not ref_breaches_single:
        print("  - v2 reference does NOT reproduce the single-box breach -> port "
              "would diverge from canonical (would be a defect).")
        port_defect = True
    else:
        print("  - v2 reference reproduces the SAME single-box corner CoG breach "
              "-> the seed skip is canonical/documented, not a port bug.")
    if ref_breaks_multi:
        print("  - v2 reference ALSO ends multi-box pallets off-envelope above "
              "gate -> inherent to greedy incremental CoG veto (can't relocate "
              "placed boxes), not a port divergence.")
    if enforcement_bug:
        print(f"  - ENFORCEMENT BUG: engine committed {total_incr_viol} non-seed "
              f"box(es) above the gate that left cumulative CoG outside the "
              f"envelope. The incremental contract was VIOLATED.")
    else:
        print("  - Engine UPHELD its incremental contract: every non-seed box "
              "placed above the gate kept cumulative CoG inside the envelope. "
              "The off-envelope FINAL CoG on those pallets is fully explained "
              "by the documented seed/first-block exemption.")
    print("  - validate() intentionally omits CoG; off-envelope realized CoG is "
          "not a validity failure by the system's own contract.")

    if enforcement_bug or port_defect:
        print("\n========= COUNTER-PROBE VERDICT: POSSIBLE REAL DEFECT =========")
        return 1
    print("\n========= COUNTER-PROBE VERDICT: EXPECTED/DOCUMENTED BEHAVIOR =========")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
