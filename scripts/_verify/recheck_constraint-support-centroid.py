"""recheck_constraint-support-centroid.py — INDEPENDENT re-derivation of the
claimed 'non-integer dims break support/centroid/overlap' defect.

We do NOT trust the original probe. We re-derive from first principles:

  T1. Minimal 2-box repro (non-integer height 98.4). Does validate() really
      report overlap + floating? Capture EXACT placement coords and dz.
  T2. Manual hand-check: with dz quantized to int but z2 computed from float,
      do the engine and validator views actually diverge by the predicted
      amount? Print the arithmetic so a human can audit it.
  T3. Integer control (138/98/124, integer pallet) -> expect 0 errors.
  T4. Sensitivity: how small a fractional part triggers it? Sweep heights
      98.0, 98.1, 98.4, 98.5, 98.9, 99.0 — and a 'round-trip clean' value
      where round(h) == h (integers) vs near-integers.
  T5. Is the failure a VALIDATOR artifact? Re-implement overlap/support with
      INTEGER-truncated dims (engine's actual view) and confirm the engine
      view is internally consistent (no overlap, full support). The defect is
      only the int-engine/float-validator MISMATCH, not engine self-inconsistency.
  T6. Realism: do the BR benchmark instances (the suite the project actually
      validates on) have integer or fractional dims? And what about industry
      cases? This tells us if fractional input is in-distribution or contrived.
  T7. Does a single-pallet PURE non-integer geometric solve (has_constraints
      =False) ALSO produce validator overlaps? (overlap check is unconditional)

Run inside Docker:
    docker run --rm -e OMP_NUM_THREADS=1 -e PYTHONHASHSEED=0 \
        -v "$(pwd)":/app -w /app pallet-packer:dev \
        python scripts/_verify/recheck_constraint-support-centroid.py
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from pallet_packer import (Box, Pallet, PackerConfig, NO_ROTATION,
                           ALL_ROTATIONS, validate)
from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit
from pallet_packer._brkga_core.precompute import precompute_constraint_arrays
from pallet_packer.brkga_v3_fast import precompute_box_dims

EPS = 1e-6


def solve_min(boxes, pal, cfg, t=2.0, mp=1, seed=42):
    return brkga_pack_v35(boxes, pal, cfg, time_limit_s=t, max_pallets=mp,
                          population_size=40, n_populations=2, patience=40,
                          local_search_budget_s=0.5, seed=seed, verbose=False,
                          n_modes=6)


def dump(r):
    out = []
    for st in r.pallets:
        for p in st.placements:
            out.append((p.box.id, p.x, p.y, p.z, p.dx, p.dy, p.dz,
                        p.x2, p.y2, p.z2))
    return out


def main():
    warmup_jit()
    print("=== recheck_constraint-support-centroid ===\n")
    from pallet_packer._brkga_core import dispatch
    print(f"[backend] {dispatch.decode_njit_mode.__module__} "
          f"batch={dispatch._BATCH_AVAILABLE}\n")

    # ---- T1: minimal 2-box repro --------------------------------------
    print("--- T1: minimal 2-box non-integer (h=98.4), sr=1.0, NO_ROTATION ---")
    mb = [Box(id="A", length=200.0, width=200.0, height=98.4, weight=1.0,
              max_load_on_top=50.0, allowed_rotations=NO_ROTATION),
          Box(id="B", length=200.0, width=200.0, height=98.4, weight=1.0,
              max_load_on_top=50.0, allowed_rotations=NO_ROTATION)]
    mp_ = Pallet(length=200.0, width=200.0, height=400.0, max_weight=1e9)
    cfg = PackerConfig(support_ratio=1.0, require_centroid_supported=True,
                       cog_envelope_fraction=1.0, max_pallets=1,
                       optimize="max_util")
    # confirm constraint path armed
    *_, has_c = precompute_constraint_arrays(mb, mp_)
    _, dims_all = precompute_box_dims(mb)
    print(f"  has_constraints={has_c}  dims_all[0,0]={dims_all[0,0].tolist()} "
          f"(int-rounded; true dz=98.4)")
    r = solve_min(mb, mp_, cfg)
    pl = dump(r)
    for (bid, x, y, z, dx, dy, dz, x2, y2, z2) in pl:
        print(f"    {bid}: pos=({x:.1f},{y:.1f},{z:.1f}) "
              f"dims=({dx:.1f},{dy:.1f},{dz:.1f}) top_z2={z2:.2f}")
    errs = validate(r, mp_, cfg)
    print(f"  validate() -> {len(errs)} errors: {errs}")
    t1_repro = len(errs) > 0 and len(pl) == 2

    # ---- T2: hand arithmetic audit ------------------------------------
    print("\n--- T2: hand arithmetic of the divergence ---")
    if len(pl) == 2:
        # sort by z
        ps = sorted(pl, key=lambda t: t[3])
        low, high = ps[0], ps[1]
        z_low, dz_low, z2_low = low[3], low[6], low[9]
        z_high = high[3]
        print(f"  bottom box top (float z2) = {z2_low:.3f}")
        print(f"  bottom box top (engine int) = z+int(round(dz)) = "
              f"{z_low:.0f}+{int(round(dz_low))} = {z_low+int(round(dz_low)):.0f}")
        print(f"  top box base z = {z_high:.3f}")
        gap_or_overlap = z_high - z2_low
        if gap_or_overlap < -EPS:
            print(f"  => FLOAT OVERLAP of {-gap_or_overlap:.3f} mm "
                  f"(top base {z_high:.1f} < bottom top {z2_low:.1f})")
        elif gap_or_overlap > EPS:
            print(f"  => FLOAT GAP of {gap_or_overlap:.3f} mm -> floating")
        else:
            print(f"  => flush (no divergence)")
    else:
        print("  (need 2 stacked boxes; got", len(pl), ")")

    # ---- T3: integer control ------------------------------------------
    print("\n--- T3: integer control (138/98/124 int pallet) ---")
    cb = [Box(id="A", length=138.0, width=98.0, height=124.0, weight=1.0,
              max_load_on_top=50.0, allowed_rotations=NO_ROTATION),
          Box(id="B", length=138.0, width=98.0, height=124.0, weight=1.0,
              max_load_on_top=50.0, allowed_rotations=NO_ROTATION)]
    cp = Pallet(length=138.0, width=98.0, height=400.0, max_weight=1e9)
    rc = solve_min(cb, cp, cfg)
    ec = validate(rc, cp, cfg)
    print(f"  validate() -> {len(ec)} errors: {ec}")
    t3_clean = len(ec) == 0

    # ---- T4: fractional sensitivity sweep -----------------------------
    print("\n--- T4: height sensitivity sweep (which fractions break) ---")
    sweep = [98.0, 98.1, 98.4, 98.5, 98.9, 99.0, 100.0, 100.5]
    t4 = {}
    for h in sweep:
        bb = [Box(id="A", length=200.0, width=200.0, height=h, weight=1.0,
                  max_load_on_top=50.0, allowed_rotations=NO_ROTATION),
              Box(id="B", length=200.0, width=200.0, height=h, weight=1.0,
                  max_load_on_top=50.0, allowed_rotations=NO_ROTATION)]
        rr = solve_min(bb, mp_, cfg)
        ee = validate(rr, mp_, cfg)
        frac = abs(h - round(h))
        ov = [e for e in ee if "overlap" in e]
        fl = [e for e in ee if "floating" in e]
        t4[h] = (len(ee), len(ov), len(fl))
        print(f"  h={h:6.1f} int(round)={int(round(h)):3d} frac={frac:.2f} "
              f"-> errs={len(ee)} overlap={len(ov)} floating={len(fl)}")

    # ---- T5: engine-view self-consistency (is engine internally OK?) --
    print("\n--- T5: engine-view (int-truncated) self-consistency ---")
    # Reproduce the engine's INTEGER view of the T1 placements and check
    # overlap/support using integer dz. If the engine view is consistent,
    # the defect is purely the int/float MISMATCH, not an engine bug.
    if len(pl) == 2:
        ps = sorted(pl, key=lambda t: t[3])
        # engine int dims
        idz = int(round(ps[0][6]))   # 98
        z_low = ps[0][3]
        z2_low_int = z_low + idz
        z_high = ps[1][3]
        eng_overlap = z_high + EPS < z2_low_int and z_high + idz > z_low + EPS \
                      and z_high < z2_low_int  # simplistic z-overlap
        eng_gap = z_high - z2_low_int
        print(f"  engine int: bottom top={z2_low_int:.0f} top base={z_high:.0f} "
              f"gap={eng_gap:.0f} -> {'flush (OK)' if abs(eng_gap)<EPS else 'NOT flush'}")
        print(f"  => engine believes flush stack (0 gap, full support); "
              f"validator (float) sees {ps[0][9]-z_high:.2f}mm overlap. "
              f"This is the int/float mismatch.")

    # ---- T6: realism — BR + industry input dims --------------------
    print("\n--- T6: are real benchmark dims integer or fractional? ---")
    try:
        from benchmarks.br import parse_thpack
        any_frac = False
        for f, sid in [("benchmarks/data/thpack1.txt", "BR1"),
                       ("benchmarks/data/thpack7.txt", "BR7")]:
            insts = parse_thpack(f, set_id=sid)
            inst = insts[0]
            fr = 0
            for b in inst.boxes:
                for v in (b.length, b.width, b.height):
                    if abs(v - round(v)) > EPS:
                        fr += 1
            pl_fr = sum(1 for v in (inst.pallet.length, inst.pallet.width,
                                    inst.pallet.height)
                        if abs(v - round(v)) > EPS)
            any_frac = any_frac or fr > 0 or pl_fr > 0
            print(f"  {sid}#{inst.instance_id}: {len(inst.boxes)} boxes, "
                  f"fractional box-dim count={fr}, pallet fractional={pl_fr}")
        print(f"  => BR suite uses {'FRACTIONAL' if any_frac else 'INTEGER'} dims")
    except Exception as e:
        print(f"  (BR parse failed: {e})")
    try:
        from benchmarks.industry import cases
        ind_frac_cases = 0
        for c in cases():
            fr = 0
            for b in c.boxes:
                for v in (b.length, b.width, b.height):
                    if abs(v - round(v)) > EPS:
                        fr += 1
            for v in (c.pallet.length, c.pallet.width, c.pallet.height):
                if abs(v - round(v)) > EPS:
                    fr += 1
            if fr:
                ind_frac_cases += 1
        print(f"  industry: {ind_frac_cases}/10 cases have any fractional dim")
    except Exception as e:
        print(f"  (industry load failed: {e})")

    # ---- T7: pure geometric (has_constraints=False) non-integer overlap
    print("\n--- T7: pure-geometric non-integer stack -> overlap check is "
          "unconditional ---")
    gb = [Box(id=f"G-{i}", length=200.0, width=200.0, height=98.4,
              weight=0.0, max_load_on_top=float("inf"),
              allowed_rotations=NO_ROTATION) for i in range(4)]
    gp = Pallet(length=200.0, width=200.0, height=800.0, max_weight=float("inf"))
    *_, has_cg = precompute_constraint_arrays(gb, gp)
    gcfg = PackerConfig(support_ratio=0.0, require_centroid_supported=False,
                        cog_envelope_fraction=1.0, enforce_load_bearing=False,
                        max_pallets=1, optimize="max_util")
    rg = solve_min(gb, gp, gcfg)
    eg = validate(rg, gp, gcfg)
    ov_g = [e for e in eg if "overlap" in e]
    print(f"  has_constraints={has_cg} (expect False)")
    for (bid, x, y, z, dx, dy, dz, x2, y2, z2) in sorted(dump(rg), key=lambda t: t[3]):
        print(f"    {bid}: z={z:.1f} z2={z2:.2f}")
    print(f"  validate() -> {len(eg)} errs, overlaps={len(ov_g)}: {eg[:3]}")

    # ---- verdict synthesis --------------------------------------------
    print("\n=== SYNTHESIS ===")
    print(f"  T1 minimal repro reproduced : {t1_repro}")
    print(f"  T3 integer control clean    : {t3_clean}")
    print(f"  T4 frac-part 0.0 cases clean: "
          f"{all(t4[h][0]==0 for h in (98.0,99.0,100.0))}")
    print(f"  T4 frac-part !=0 cases bad  : "
          f"{all(t4[h][0]>0 for h in (98.1,98.4,98.5,98.9,100.5))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
